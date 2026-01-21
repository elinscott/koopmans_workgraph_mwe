from ase import Atoms
from typing import Protocol
from ase.spectrum.band_structure import BandStructure
from pydantic import validate_call, Field

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


def run_scf_nscf_bands(inputs: PwWorkflowInputs, engine: EngineProtocol) -> PwWorkflowOutputs:
    # Validate inputs
    if not isinstance(inputs, PwWorkflowInputs):
        inputs = PwWorkflowInputs.model_validate(inputs)

    scf_inputs = PwScfInputs.model_validate({'atoms': inputs.atoms,
                                             'parameters': inputs.pw_parameters,
                                             'pseudopotential_family': inputs.pseudopotential_family,
                                             'kpoints': inputs.kpoints.explicit_grid})
    scf_inputs.parameters.system.nbnd = None
    scf_outputs = engine.run_pw_scf(scf_inputs)

    nscf_outputs = engine.run_pw_nscf(
        {
            'atoms': inputs.atoms,
            'parameters': inputs.pw_parameters,
            'kpoints': inputs.kpoints.explicit_grid,
            'pseudopotential_family': inputs.pseudopotential_family,
            'outdir': scf_outputs.outdir
        }
    )

    bands_outputs = engine.run_pw_bands(
        {
            'atoms': inputs.atoms,
            'parameters': inputs.pw_parameters,
            'kpoints': inputs.kpoints.path,
            'pseudopotential_family': inputs.pseudopotential_family,
            'outdir': nscf_outputs.outdir
        }
    )

    return {'total_energy': scf_outputs.total_energy,
            'band_structure': bands_outputs.band_structure}# PwWorkflowOutputs(total_energy=scf_outputs.total_energy, band_structure=bands_outputs.band_structure)
