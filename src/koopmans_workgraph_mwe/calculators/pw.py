"""pw calculator module for koopmans."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Unpack

import numpy as np
import numpy.typing as npt
from ase.calculators.calculator import CalculationFailed
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.dft.kpoints import BandPath
from ase.spectrum.band_structure import BandStructure
from pint import Quantity

from koopmans_workgraph_mwe.commands import CommandsConfig
from koopmans_workgraph_mwe.files import (
    DirectoryDict,
    LinkDict,
    SingleFileDict,
    directory,
    single_file,
)
from koopmans_workgraph_mwe.kpoints import ExplicitKpoint
from koopmans_workgraph_mwe.parameters.pw import PwInputParametersDict
from koopmans_workgraph_mwe.status import Status
from koopmans_workgraph_mwe.utils import remove_null_from_obj

from .calculator import CalculatorInputsDict, CalculatorOutputsDict

class _PwInputsDict(CalculatorInputsDict):
    parameters: PwInputParametersDict
    pseudopotential_family: str
    outdir: LinkDict | None

class PwInputsDict(_PwInputsDict):
    kpoints: BandPath | list[ExplicitKpoint]

class PwScfInputsDict(_PwInputsDict):
    kpoints: list[ExplicitKpoint]

class PwNscfInputsDict(_PwInputsDict):
    kpoints: list[ExplicitKpoint]

class PwBandsInputsDict(_PwInputsDict):
    kpoints: BandPath


class _PwOutputsDict(CalculatorOutputsDict):
    eigenvalues: npt.NDArray[np.float64] | None
    fermi_level: list[float] | None
    outdir: DirectoryDict

class PWOutputsDict(_PwOutputsDict):
    total_energy: float | None
    band_structure: BandStructure | None

class PwScfOutputsDict(_PwOutputsDict):
    total_energy: float


def pw_scf_outputs(kind: str='calculator_output', **kwargs):
    return PwScfOutputsDict(kind=kind, **kwargs)

class PwNscfOutputsDict(_PwOutputsDict):
    pass


def pw_nscf_outputs(kind: str='calculator_output', **kwargs):
    return PwNscfOutputsDict(kind=kind, **kwargs)

class PwBandsOutputsDict(_PwOutputsDict):
    band_structure: BandStructure


def pw_bands_outputs(kind: str='calculator_output', **kwargs):
    return PwBandsOutputsDict(kind=kind, **kwargs)


def _run_pw_with_ase(
    uid: str,
    commands: CommandsConfig,
    link_file: Callable[[SingleFileDict | DirectoryDict, SingleFileDict | DirectoryDict, bool], None],
    **kwargs: Unpack[_PwInputsDict],
) -> PWOutputsDict:
    # Create a profile and calculator
    profile = EspressoProfile(command=commands['pw'], pseudo_dir="pseudopotentials")  # type: ignore[no-untyped-call]
    ase_calc = Espresso(directory=uid, profile=profile)  # type: ignore[no-untyped-call]

    # Set up the input parameters
    ase_calc.parameters['input_data'] = remove_null_from_obj(kwargs["parameters"])

    if isinstance(kwargs['kpoints'], BandPath):
        kpts = kwargs['kpoints']
    else:
        kpts = np.array(kwargs['kpoints'], dtype=np.float64)
    ase_calc.parameters['kpts'] = kpts

    # Set up the pseudopotentials
    pseudopotentials: dict[str, str] = {}
    for atom in kwargs["atoms"]:
        if atom.symbol not in pseudopotentials:
            pseudopotentials[atom.symbol] = f"{atom.symbol}.upf"
    ase_calc.parameters['pseudopotentials'] = pseudopotentials

    # Copy over the pseudopotentials
    pseudo_folder = Path(uid) / "pseudopotentials"
    pseudo_folder.mkdir(parents=True, exist_ok=True)
    for pseudo in set(pseudopotentials.values()):
        src = single_file(uid=str(Path(__file__).parents[1] / 'pseudopotentials' / kwargs['pseudopotential_family'] / pseudo))
        dest = single_file(uid=str(pseudo_folder / pseudo))
        link_file(src, dest, True)

    error_message: str | None = None
    error_type: type[BaseException] | None = None
    try:
        ase_calc.calculate(atoms=kwargs["atoms"], properties=['energy'], system_changes=None)  # type: ignore[no-untyped-call]
        status = Status.COMPLETED
    except CalculationFailed as error:
        status = Status.FAILED
        error_message = str(error)
        error_type = type(error)
    outputs: dict[str, Any] = {'status': status, 'error_message': error_message, 'error_type': error_type}

    # Store the total energy
    outputs['total_energy'] = ase_calc.results.get('energy', None)

    # Store the fermi level
    fermi_level_result: Any = ase_calc.results.get('fermi_level', [])
    fermi_level: list[float] = fermi_level_result if isinstance(fermi_level_result, list) else [fermi_level_result]
    outputs['fermi_level'] = fermi_level

    # Store the eigenvalues
    outputs['eigenvalues'] = ase_calc.results.get('eigenvalues', None)

    # Store the outdir
    outputs['outdir'] = directory(uid = uid + '/' + kwargs['parameters']['control']['outdir'])

    # Store the band structure
    if isinstance(kpts, BandPath):
        reference = max(fermi_level) if fermi_level else 0.0
        try:
            outputs['band_structure'] = BandStructure(path=kpts, energies=outputs['eigenvalues'], reference=reference)  # type: ignore[no-untyped-call]
        except:
            raise ValueError()

    # Store the walltime
    outputs['walltime'] = ase_calc.results.get('walltime', '0.0s')

    return outputs


def run_scf(
    uid: str,
    commands: CommandsConfig,
    link_file: Callable[[SingleFileDict | DirectoryDict, SingleFileDict | DirectoryDict, bool], None],
    **kwargs: Unpack[PwScfInputsDict],
) -> PwScfOutputsDict:
    outputs = _run_pw_with_ase(uid, commands, link_file, **kwargs)
    return pw_scf_outputs(**outputs)


def run_nscf(
    uid: str,
    commands: CommandsConfig,
    link_file: Callable[[SingleFileDict | DirectoryDict, SingleFileDict | DirectoryDict, bool], None],
    **kwargs: Unpack[PwNscfInputsDict],
) -> PwNscfOutputsDict:
    outputs = _run_pw_with_ase(uid, commands, link_file, **kwargs)
    return pw_nscf_outputs(**outputs)


def run_bands(
    uid: str,
    commands: CommandsConfig,
    link_file: Callable[[SingleFileDict | DirectoryDict, SingleFileDict | DirectoryDict, bool], None],
    **kwargs: Unpack[PwBandsInputsDict],
) -> PwBandsOutputsDict:
    outputs = _run_pw_with_ase(uid, commands, link_file, **kwargs)
    return pw_bands_outputs(**outputs)

