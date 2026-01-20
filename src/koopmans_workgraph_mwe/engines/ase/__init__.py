from typing import overload
import shutil
from pathlib import Path
from koopmans_workgraph_mwe.engines.engine import Engine
from koopmans_workgraph_mwe.calculators.pw import PwScfInputs, PwScfOutputs, PwNscfInputs, PwNscfOutputs, PwBandsInputs, PwBandsOutputs
from koopmans_workgraph_mwe.files import File, LocalFile
from koopmans_workgraph_mwe.status import Status
from ase.calculators.espresso import EspressoProfile, Espresso
from ase.calculators.calculator import CalculationFailed
from ase.spectrum.band_structure import BandStructure

class AseEngine(Engine):

    @overload
    def _run_pw(self, inputs: PwScfInputs, uid: str) -> PwScfOutputs: ...

    @overload
    def _run_pw(self, inputs: PwNscfInputs, uid: str) -> PwNscfOutputs: ...

    @overload
    def _run_pw(self, inputs: PwBandsInputs, uid: str) -> PwBandsOutputs: ...

    def _run_pw(self, inputs, uid):
        # Create a profile and calculator
        profile = EspressoProfile(command=self.pw_command, pseudo_dir="pseudopotentials")
        ase_calc = Espresso(directory=uid, profile=profile)

        # Set up the input parameters
        dct = inputs.parameters.model_dump(exclude_none=True, exclude_defaults=True)
        ase_calc.parameters['input_data'] = dct

        # Set up the k-points
        ase_calc.parameters['kpts'] = inputs.kpoints

        # Set up the pseudopotentials
        pseudopotentials = {}
        for atom in inputs.atoms:
            if atom.symbol not in pseudopotentials:
                pseudopotentials[atom.symbol] = f"{atom.symbol}.upf"
        ase_calc.parameters['pseudopotentials'] = pseudopotentials

        # Copy over the pseudopotentials
        pseudo_folder = File(parent_process_uid=uid, path='pseudopotentials')
        self.mkdir(pseudo_folder, parents=True, exist_ok=True)
        for pseudo in set(pseudopotentials.values()):
            src = LocalFile(path=Path(__file__).parent
                            / 'pseudopotentials' / inputs.pseudopotential_family / pseudo)
            self.link_file(src, pseudo_folder / pseudo, overwrite=True)

        ase_calc.prefix = 'espresso'

        error_message = None
        error_type = None
        try:
            ase_calc.calculate(atoms=inputs.atoms, properties='energy', system_changes=None)
            status = Status.COMPLETED
        except CalculationFailed as error:
            status = Status.FAILED
            error_message = str(error)
            error_type = type(error)
        output_dict = {'status': status, 'error_message': error_message, 'error_type': error_type}

        # Store the total energy

        # Store the fermi level
        fermi_level: list[float] = ase_calc.results.get('fermi_level', [])
        if not isinstance(fermi_level, list):
            fermi_level = [fermi_level]
        output_dict['fermi_level'] = fermi_level

        # Store the eigenvalues
        eigenvalues=ase_calc.results.get('eigenvalues', None)
        output_dict['eigenvalues'] = eigenvalues

        # Store calculation-specific outputs
        if isinstance(inputs, PwScfInputs):
            output_dict['total_energy'] = ase_calc.results['energy']
            output_class = PwScfOutputs
        elif isinstance(inputs, PwNscfInputs):
            output_class = PwNscfOutputs
        elif isinstance(inputs, PwBandsInputs):
            output_dict['band_structure'] = BandStructure(path=inputs.kpoints,
                                           energies=eigenvalues,
                                           reference=max(fermi_level) if fermi_level else 0.0)
            output_class = PwBandsOutputs
        else:
            raise ValueError("Unknown input type")

        # Store the outdir
        output_dict['outdir'] = File(parent_process_uid=uid, path=inputs.parameters.control.outdir)

        # Store the walltime
        output_dict['walltime'] = ase_calc.results.get('walltime', '0.0s')

        outputs = output_class(**output_dict)
        
        return outputs

    def _run_pw_scf(self, inputs: PwScfInputs, uid: str) -> PwScfOutputs:
        return self._run_pw(inputs, uid)

    def _run_pw_nscf(self, inputs: PwNscfInputs, uid: str) -> PwNscfOutputs:
        return self._run_pw(inputs, uid)

    def _run_pw_bands(self, inputs: PwBandsInputs, uid: str) -> PwBandsOutputs:
        return self._run_pw(inputs, uid)
    
    def file_exists(self, path: File) -> bool:
        explicit_path = _file_to_path(path)
        return explicit_path.exists() or explicit_path.is_symlink()

    def write_to_file(self, content: str, path: File):
        """Write content to a file."""
        with open(_file_to_path(path), 'w') as f:
            f.write(content)
    
    def delete_file(self, path: File):
        explicit_path = _file_to_path(path)
        if explicit_path.is_dir():
            shutil.rmtree(explicit_path)
        else:
            explicit_path.unlink()
    
    def _copy_file(self, src: File | LocalFile, dest: File):
        src_path = _file_to_path(src)
        dest_path = _file_to_path(dest)
        if src_path.is_dir():
            shutil.copytree(src_path, dest_path)
        else:
            shutil.copy(src_path, dest_path)
            
    def _link_file(self, src: File | LocalFile, dest: File, recursive: bool = False):
        """Link a file from src to dest."""
        src_path = _file_to_path(src)
        dest_path = _file_to_path(dest)
        relative_path = src_path.resolve().relative_to(dest_path.parent.resolve(), walk_up=True)
        if self.is_dir(src):
            dest_path.symlink_to(relative_path, target_is_directory=True)
        else:
            dest_path.symlink_to(relative_path)

    def is_dir(self, path: File | LocalFile) -> bool:
        explicit_path = _file_to_path(path)
        return explicit_path.is_dir()
    
    def mkdir(self, path: File, parents: bool = False, exist_ok: bool = False):
        """Create a directory at the given path."""
        directory = Path(path.parent_process_uid) / Path(path.path)
        directory.mkdir(parents=parents, exist_ok=exist_ok)
    
def _file_to_path(file: File | LocalFile) -> Path:
    return Path(file.parent_process_uid) / file.path if isinstance(file, File) else file.path