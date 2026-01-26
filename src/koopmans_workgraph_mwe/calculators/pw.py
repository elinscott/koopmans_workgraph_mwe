"""pw calculator module for koopmans."""

import uuid
from pathlib import Path
from typing import Any, Unpack, NotRequired

import numpy as np
import numpy.typing as npt
from ase.calculators.calculator import CalculationFailed
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.dft.kpoints import BandPath
from ase.spectrum.band_structure import BandStructure

from koopmans_workgraph_mwe.commands import CommandsConfig
from koopmans_workgraph_mwe.files import DirectoryDict, directory_factory
from koopmans_workgraph_mwe.kpoints import ExplicitKpoint
from koopmans_workgraph_mwe.parameters.pw import PwInputParametersDict
from koopmans_workgraph_mwe.status import Status
from koopmans_workgraph_mwe.utils import remove_null_from_obj
from koopmans_workgraph_mwe.os.local import link_path

from .calculator import CalculatorInputsDict, CalculatorOutputsDict

class _PwInputsDict(CalculatorInputsDict):
    parameters: PwInputParametersDict

class PwInputsDict(_PwInputsDict):
    kpoints: BandPath | list[ExplicitKpoint]
    outdir: NotRequired[DirectoryDict]

class PwScfInputsDict(_PwInputsDict):
    kpoints: list[ExplicitKpoint]
    outdir: NotRequired[DirectoryDict]

class PwNscfInputsDict(_PwInputsDict):
    kpoints: list[ExplicitKpoint]
    outdir: DirectoryDict

class PwBandsInputsDict(_PwInputsDict):
    kpoints: BandPath
    outdir: DirectoryDict


class _PwOutputsDict(CalculatorOutputsDict):
    eigenvalues: npt.NDArray[np.float64] | None
    fermi_level: list[float] | None
    outdir: DirectoryDict

class PWOutputsDict(_PwOutputsDict):
    total_energy: float | None
    band_structure: BandStructure | None

class PwScfOutputsDict(_PwOutputsDict):
    total_energy: float


class PwNscfOutputsDict(_PwOutputsDict):
    pass


class PwBandsOutputsDict(_PwOutputsDict):
    band_structure: BandStructure


def _run_pw_with_ase(
    uid: str | None = None,
    commands: CommandsConfig | None = None,
    **kwargs: Unpack[PwInputsDict],
) -> PWOutputsDict:
    # Generate a uid if not provided
    uid = uid if uid is not None else str(uuid.uuid4())

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
    outputs['outdir'] = directory_factory(parent_uid=uid, path=kwargs['parameters']['control']['outdir'])

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
    uid: str | None = None,
    commands: CommandsConfig | None = None,
    **kwargs: Unpack[PwScfInputsDict],
) -> PwScfOutputsDict:
    outputs = _run_pw_with_ase(uid, commands, **kwargs)
    return PwScfOutputsDict(**outputs)


def run_nscf(
    uid: str | None = None,
    commands: CommandsConfig | None = None,
    **kwargs: Unpack[PwNscfInputsDict],
) -> PwNscfOutputsDict:
    outputs = _run_pw_with_ase(uid, commands, **kwargs)
    return PwNscfOutputsDict(**outputs)


def run_bands(
    uid: str | None = None,
    commands: CommandsConfig | None = None,
    **kwargs: Unpack[PwBandsInputsDict],
) -> PwBandsOutputsDict:
    outputs = _run_pw_with_ase(uid, commands, **kwargs)
    return PwBandsOutputsDict(**outputs)

