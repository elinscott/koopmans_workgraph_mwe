from aiida_workgraph import task
from functools import partial
from koopmans_workgraph_mwe.workflows.scf_nscf_bands import run_scf_nscf_bands, PwWorkflowInputs

from koopmans_workgraph_mwe.calculators.pw import PwScfInputs, PwScfOutputs, PwNscfInputs, PwNscfOutputs, PwBandsInputs, PwBandsOutputs

@task()
def run_pw_scf(inputs: PwScfInputs) -> PwScfOutputs:
    pass

@task()
def run_pw_nscf(inputs: PwNscfInputs) -> PwNscfOutputs:
    pass

@task()
def run_pw_bands(inputs: PwBandsInputs) -> PwBandsOutputs:
    pass

class AiiDATasks:
    run_pw_scf = run_pw_scf
    run_pw_nscf = run_pw_nscf
    run_pw_bands = run_pw_bands

@task.graph()
def aiida_run_scf_nscf_bands(inputs: PwWorkflowInputs):
    return run_scf_nscf_bands(inputs, engine=AiiDATasks)
