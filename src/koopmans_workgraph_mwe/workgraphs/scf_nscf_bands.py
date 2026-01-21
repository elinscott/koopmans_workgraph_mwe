from aiida_workgraph import task
from koopmans_workgraph_mwe.workflows.scf_nscf_bands import run_scf_nscf_bands_unwrapped, PwWorkflowOutputs
from pydash import set_

from koopmans_workgraph_mwe.calculators.pw import PwInputParameters, PwScfOutputs, PwNscfInputs, PwNscfOutputs, PwBandsInputs, PwBandsOutputs
from ase import Atoms
from ase.dft.kpoints import BandPath
from ase.spectrum.band_structure import BandStructure
from typing import Annotated, Any
from numpy import typing as npt
import numpy as np
from pydantic import BeforeValidator
from koopmans_workgraph_mwe.pydantic_config import FlexibleStr, BaseModel
from koopmans_workgraph_mwe.serialization import deserialize_numpy
from koopmans_workgraph_mwe.kpoints import Kpoints
from koopmans_workgraph_mwe.files import LinkedFile, File
from koopmans_workgraph_mwe.status import Status

"""
The following implementations are very verbose. Ideally they'd be replaced by e.g.
@task
def run_pw_nscf(inputs: PwNscfInputs) -> PwNscfOutputs:
    return PwNscfOutputs(...)
"""

@task(
    inputs = ["atoms", "parameters", "pseudopotential_family", "kpoints"]
)
def run_pw_scf(
        atoms: Atoms,
        parameters: PwInputParameters,
        pseudopotential_family: FlexibleStr,
        kpoints: Annotated[npt.NDArray[np.float64], BeforeValidator(deserialize_numpy)]
    ) -> PwScfOutputs:
    outdir = File(parent_process_uid="placeholder", path=parameters.control.outdir)
    return dict(
        walltime="0s",
        fermi_level=[],
        total_energy=-10.0,
        outdir=outdir.model_dump(mode='json'),
        status=Status.COMPLETED.value,
        error_message=None,
        error_type=None,
        eigenvalues=None,
    )

@task(
    inputs = ["atoms", "parameters", "pseudopotential_family", "kpoints", "outdir"]
)
def run_pw_nscf(
    atoms: Atoms,
    parameters: PwInputParameters,
    pseudopotential_family: FlexibleStr,
    kpoints: Annotated[npt.NDArray[np.float64], BeforeValidator(deserialize_numpy)],
    outdir: LinkedFile
) -> PwNscfOutputs:
    return dict(
        walltime="0s",
        fermi_level=[],
        outdir=outdir,
        status=Status.COMPLETED.value,
        error_message=None,
        error_type=None,
        eigenvalues=None,
    )

@task(
    inputs = ["atoms", "parameters", "pseudopotential_family", "kpoints", "outdir"]
)
def run_pw_bands(
    atoms: Atoms,
    parameters: PwInputParameters,
    pseudopotential_family: FlexibleStr,
    kpoints: BandPath,
    outdir: LinkedFile
) -> PwBandsOutputs:
    from koopmans_workgraph_mwe.serialization import serialize_ase_bandstructure
    # Create a valid mock band structure with proper shape: (nspins, nkpts, nbands)
    bandpath = atoms.cell.bandpath()
    nkpts = len(bandpath.kpts)
    mock_energies = np.zeros((1, nkpts, 4))  # 1 spin, nkpts k-points, 4 bands
    band_structure = BandStructure(path=bandpath, energies=mock_energies)
    return dict(
        walltime="0s",
        fermi_level=[],
        outdir=outdir,
        band_structure=band_structure,
        status=Status.COMPLETED.value,
        error_message=None,
        error_type=None,
        eigenvalues=None,
        vbm=None,
        cbm=None,
    )

@task(
    inputs = ["model", "key"],
    outputs = ["model"]
)
def run_set_to_none(model: BaseModel, key: str):
    model_copy = model.model_copy()
    set_(model_copy, key, None)
    return {"model": model_copy}

@task(
    inputs = ["kpoints"],
    outputs = ["result"]
)
def kpoints_to_explicit_grid(kpoints: Kpoints) -> npt.NDArray[np.float64]:
    return kpoints.explicit_grid

class AiiDATasks:
    run_pw_scf = run_pw_scf
    run_pw_nscf = run_pw_nscf
    run_pw_bands = run_pw_bands
    run_set_to_none = run_set_to_none
    kpoints_to_explicit_grid = kpoints_to_explicit_grid

@task.graph(
    inputs = ["atoms", "pw_parameters", "pseudopotential_family", "kpoints"]
)
def aiida_run_scf_nscf_bands(
    atoms: Atoms,
    pw_parameters: PwInputParameters,
    pseudopotential_family: FlexibleStr,
    kpoints: Kpoints
) -> PwWorkflowOutputs:
    return run_scf_nscf_bands_unwrapped(
        atoms=atoms,
        pw_parameters=pw_parameters,
        pseudopotential_family=pseudopotential_family,
        kpoints=kpoints,
        engine=AiiDATasks)