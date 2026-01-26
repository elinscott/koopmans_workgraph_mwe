from collections.abc import Callable
from typing import Any, Protocol, TypedDict, TypeVar, Unpack, cast

from aiida_workgraph import task as aiida_task
from ase import Atoms
from ase.dft.kpoints import BandPath
from ase.spectrum.band_structure import BandStructure
from pydantic import Field
from aiida.orm import load_code

from koopmans_workgraph_mwe.calculators.pw import run_bands, run_nscf, run_scf
from koopmans_workgraph_mwe.commands import CommandsConfig
from koopmans_workgraph_mwe.engines.localhost import LocalhostEngine
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


class PwWorkflowInputs(BaseModel):
    atoms: Atoms
    kpoints: KpointsModel
    pw_parameters: PwInputParametersModel = Field(default_factory=PwInputParametersModel)
    pseudopotential_family: str


class PwWorkflowOutputsDict(TypedDict):
    total_energy: float
    band_structure: BandStructure


class PwWorkflowOutputs(BaseModel):
    total_energy: float
    band_structure: BandStructure


def kpoints_to_bandpath(kpoints: Kpoints) -> BandPath | None:
    """Extract the bandpath from a Kpoints object."""
    return kpoints.get('path')


R = TypeVar('R')


class TaskWrapper(Protocol):
    def __call__(self, func: Callable[..., R], name: str | None = None) -> Callable[..., R]: ...


def run_scf_nscf_bands_core(
    task: TaskWrapper,
    **kwargs: Unpack[PwWorkflowInputsDict],
) -> PwWorkflowOutputsDict:
    scf_parameters = task(normalize_scf)(parameters=kwargs['pw_parameters'],
                                         pseudo_family=kwargs['pseudopotential_family'])
    task(require_all)(parameters=scf_parameters, requirements=SCF_REQUIREMENTS)

    scf_outputs = task(run_scf, name='pw-scf')(
        atoms=kwargs['atoms'],
        parameters=scf_parameters,
        kpoints=kwargs['kpoints']['explicit_grid']
    )

    nscf_parameters = task(normalize_nscf)(parameters=kwargs['pw_parameters'],
                                           pseudo_family=kwargs['pseudopotential_family'])
    task(require_all)(parameters=nscf_parameters, requirements=NSCF_REQUIREMENTS)
    nscf_outputs = task(run_nscf, name='pw-nscf')(
        atoms=kwargs['atoms'],
        parameters=nscf_parameters,
        outdir=scf_outputs['outdir'],
        kpoints=kwargs['kpoints']['explicit_grid']
    )

    bands_parameters = task(normalize_bands)(parameters=kwargs['pw_parameters'],
                                             pseudo_family=kwargs['pseudopotential_family'])
    task(require_all)(parameters=bands_parameters, requirements=BANDS_REQUIREMENTS)
    bands_outputs = task(run_bands, name='pw-bands')(
        atoms=kwargs['atoms'],
        parameters=bands_parameters,
        outdir=nscf_outputs['outdir'],
        kpoints=kwargs['kpoints']['path']
    )

    return PwWorkflowOutputsDict(
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

def get_commands_from_aiida(pw_label: str) -> CommandsConfig:
    """Load command paths from AiiDA codes configured in the profile."""
    pw_code = load_code(pw_label)
    return CommandsConfig(pw=str(pw_code.filepath_executable))


def make_aiida_task_wrapper(commands: CommandsConfig) -> TaskWrapper:
    """Create a task wrapper that injects commands from AiiDA profile."""
    def wrapper(func: Callable[..., R], name: str | None = None) -> Callable[..., R]:
        wrapped = aiida_task.task(func)

        def with_injected_args(**kwargs: Any) -> R:
            if name is not None:
                kwargs.setdefault('metadata', {})['call_link_label'] = name
            if 'commands' in func.__code__.co_varnames:
                kwargs['commands'] = commands
            return wrapped(**kwargs)
        return with_injected_args
    return wrapper


@aiida_task.graph
def run_scf_nscf_bands_with_aiida(
    pw_label: str,
    **kwargs: Unpack[PwWorkflowInputsDict],
) -> PwWorkflowOutputsDict:
    # commands = get_commands_from_aiida(pw_label)
    commands = CommandsConfig(pw='pw.x')  # temporary workaround
    task_wrapper = make_aiida_task_wrapper(commands)
    return run_scf_nscf_bands_core(task_wrapper, **kwargs)
