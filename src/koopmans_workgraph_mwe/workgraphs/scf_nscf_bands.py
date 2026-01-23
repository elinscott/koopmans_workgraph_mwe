from collections.abc import Callable
from typing import Any, Literal, Protocol, TypedDict, TypeVar, Unpack, cast

from aiida_workgraph import task as aiida_task
from ase import Atoms
from ase.dft.kpoints import BandPath
from ase.spectrum.band_structure import BandStructure
from pydantic import Field

from koopmans_workgraph_mwe.calculators.pw import run_bands, run_nscf, run_scf
from koopmans_workgraph_mwe.engines.localhost import LocalhostEngine
from koopmans_workgraph_mwe.files import link
from koopmans_workgraph_mwe.kpoints import Kpoints, KpointsModel
from koopmans_workgraph_mwe.normalize.pw import normalize_bands, normalize_nscf, normalize_scf
from koopmans_workgraph_mwe.parameters.pw import PwInputParametersDict, PwInputParametersModel
from koopmans_workgraph_mwe.pydantic_config import BaseModel
from koopmans_workgraph_mwe.requirements.pw import (
    BANDS_REQUIREMENTS,
    NSCF_REQUIREMENTS,
    SCF_REQUIREMENTS,
    require_all,
)


class PwWorkflowInputsDict(TypedDict):
    atoms: Atoms
    kpoints: Kpoints
    pw_parameters: PwInputParametersDict
    pseudopotential_family: str
    kind: Literal['node_inputs']


class PwWorkflowInputs(BaseModel):
    atoms: Atoms
    kpoints: KpointsModel
    pw_parameters: PwInputParametersModel = Field(default_factory=PwInputParametersModel)
    pseudopotential_family: str
    kind: Literal['node_inputs'] = 'node_inputs'


class PwWorkflowOutputsDict(TypedDict):
    total_energy: float
    band_structure: BandStructure
    kind: Literal['node_outputs']


def pw_workflow_outputs_factory(total_energy: float, band_structure: BandStructure) -> PwWorkflowOutputsDict:
    return PwWorkflowOutputsDict(
        total_energy=total_energy,
        band_structure=band_structure,
        kind='node_outputs',
    )


class PwWorkflowOutputs(BaseModel):
    total_energy: float
    band_structure: BandStructure
    kind: Literal['node_outputs'] = 'node_outputs'


def kpoints_to_bandpath(kpoints: Kpoints) -> BandPath | None:
    """Extract the bandpath from a Kpoints object."""
    return kpoints.get('path')


R = TypeVar('R')


class TaskWrapper(Protocol):
    def __call__(self, func: Callable[..., R]) -> Callable[..., R]: ...


def run_scf_nscf_bands_core(
    task: TaskWrapper,
    **kwargs: Unpack[PwWorkflowInputsDict],
) -> PwWorkflowOutputsDict:
    scf_parameters = task(normalize_scf)(parameters=kwargs['pw_parameters'])
    task(require_all)(parameters=scf_parameters, requirements=SCF_REQUIREMENTS)

    scf_outputs = task(run_scf)(
        metadata={'call_link_label': 'pw-scf'},
        atoms=kwargs['atoms'],
        parameters=scf_parameters,
        pseudopotential_family=kwargs['pseudopotential_family'],
        kpoints=kwargs['kpoints']['explicit_grid']
    )

    nscf_parameters = task(normalize_nscf)(parameters=kwargs['pw_parameters'])
    task(require_all)(parameters=nscf_parameters, requirements=NSCF_REQUIREMENTS)
    nscf_outputs = task(run_nscf)(
        metadata={'call_link_label': 'pw-nscf'},
        atoms=kwargs['atoms'],
        parameters=nscf_parameters,
        pseudopotential_family=kwargs['pseudopotential_family'],
        outdir=link(scf_outputs['outdir'], recursive_symlink=True),
        kpoints=kwargs['kpoints']['explicit_grid']
    )

    bands_parameters = task(normalize_bands)(parameters=kwargs['pw_parameters'])
    task(require_all)(parameters=bands_parameters, requirements=BANDS_REQUIREMENTS)
    bands_outputs = task(run_bands)(
        metadata={'call_link_label': 'pw-bands'},
        atoms=kwargs['atoms'],
        parameters=bands_parameters,
        pseudopotential_family=kwargs['pseudopotential_family'],
        outdir=link(nscf_outputs['outdir'], recursive_symlink=True),
        kpoints=kwargs['kpoints']['path']
    )

    return pw_workflow_outputs_factory(
        total_energy=scf_outputs['total_energy'],
        band_structure=bands_outputs['band_structure'],
    )


def adapt_pw_input(kwargs: dict[str, Any]) -> PwWorkflowInputsDict:
    model = PwWorkflowInputs.model_validate(kwargs)
    return cast(PwWorkflowInputsDict, model.model_dump())


def adapt_pw_output(output: PwWorkflowOutputsDict) -> PwWorkflowOutputs:
    return PwWorkflowOutputs.model_validate(output)


def run_scf_nscf_bands(engine: LocalhostEngine, **kwargs: Any) -> PwWorkflowOutputs:
    sanitised_kwargs = adapt_pw_input(kwargs)
    output = run_scf_nscf_bands_core(engine.task, **sanitised_kwargs)
    return adapt_pw_output(output)


@aiida_task.graph
def run_scf_nscf_bands_with_aiida(**kwargs: Unpack[PwWorkflowInputsDict]) -> PwWorkflowOutputsDict:
    return run_scf_nscf_bands_core(aiida_task.task, **kwargs)
