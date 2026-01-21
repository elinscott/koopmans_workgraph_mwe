from ase import Atoms
from typing import Protocol, Any
from ase.spectrum.band_structure import BandStructure

from koopmans_workgraph_mwe.parameters.pw import PwInputParameters
from koopmans_workgraph_mwe.calculators.pw import PwBandsInputs, PwNscfInputs, PwScfInputs, PwBandsOutputs, PwNscfOutputs, PwScfOutputs
from koopmans_workgraph_mwe.kpoints import Kpoints
from koopmans_workgraph_mwe.pydantic_config import BaseModel, FlexibleStr

class PwWorkflowInputs(BaseModel):
    atoms: Atoms
    kpoints: Kpoints
    pw_parameters: PwInputParameters = {} # Field(default_factory=lambda: PwInputParameters())
    pseudopotential_family: FlexibleStr


class PwWorkflowOutputs(BaseModel):
    total_energy: float
    band_structure: BandStructure


# Define a protocol that contains the required engine methods
class EngineProtocol(Protocol):
    def run_pw_scf(self, inputs: PwScfInputs) -> PwScfOutputs: ...
    def run_pw_nscf(self, inputs: PwNscfInputs) -> PwNscfOutputs: ...
    def run_pw_bands(self, inputs: PwBandsInputs) -> PwBandsOutputs: ...
    def run_set(self, inputs: BaseModel, key: str, value: Any) -> BaseModel: ...


def run_scf_nscf_bands(inputs: PwWorkflowInputs, engine: EngineProtocol) -> PwWorkflowOutputs:
    scf_parameters = engine.run_set(inputs.pw_parameters, 'system.nbnd', None)

    scf_outputs = engine.run_pw_scf(
        atoms = inputs.atoms,
        parameters = scf_parameters,
        pseudopotential_family = inputs.pseudopotential_family,
        kpoints = inputs.kpoints.explicit_grid
    )

    nscf_outputs = engine.run_pw_nscf(
        atoms = inputs.atoms,
        parameters = inputs.pw_parameters,
        kpoints = inputs.kpoints.explicit_grid,
        pseudopotential_family = inputs.pseudopotential_family,
        outdir = scf_outputs.outdir
    )

    bands_outputs = engine.run_pw_bands(
        atoms = inputs.atoms,
        parameters = inputs.pw_parameters,
        kpoints = inputs.kpoints.path,
        pseudopotential_family = inputs.pseudopotential_family,
        outdir = nscf_outputs.outdir
    )

    return PwWorkflowOutputs.model_construct(total_energy = scf_outputs.total_energy,
                                             band_structure = bands_outputs.band_structure)


def run_scf_nscf_bands_unwrapped(engine: EngineProtocol, **kwargs):
    inputs = PwWorkflowInputs.model_validate(kwargs)
    # scf_parameters = engine.run_set(inputs.pw_parameters, 'system.nbnd', None)

    scf_outputs = engine.run_pw_scf(
        atoms = inputs.atoms,
        parameters = inputs.pw_parameters, # scf_parameters,
        pseudopotential_family = inputs.pseudopotential_family,
        kpoints = inputs.kpoints.explicit_grid
    )

    nscf_outputs = engine.run_pw_nscf(
        atoms = inputs.atoms,
        parameters = inputs.pw_parameters,
        kpoints = inputs.kpoints.explicit_grid,
        pseudopotential_family = inputs.pseudopotential_family,
        outdir = scf_outputs.outdir
    )

    bands_outputs = engine.run_pw_bands(
        atoms = inputs.atoms,
        parameters = inputs.pw_parameters,
        kpoints = inputs.kpoints.path,
        pseudopotential_family = inputs.pseudopotential_family,
        outdir = nscf_outputs.outdir
    )

    # Return socket references directly, not wrapped in Pydantic model
    return {'total_energy': scf_outputs.total_energy,
            'band_structure': bands_outputs.band_structure}