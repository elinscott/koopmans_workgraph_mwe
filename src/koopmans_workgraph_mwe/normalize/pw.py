from koopmans_workgraph_mwe.parameters.pw import PwInputParametersDict
from pathlib import Path

def normalize_pw(parameters: PwInputParametersDict, pseudo_family: str) -> PwInputParametersDict:
    """Normalize common parameters for PW calculations."""
    parameters['control']['outdir'] = 'tmp'
    parameters['control']['prefix'] = 'espresso'
    parameters['control']['pseudo_dir'] = str(Path(__file__).parents[1].resolve() / 'pseudopotentials' / pseudo_family)
    # For the moment, limited support until we debug the full feature set
    parameters['system'] = {k: v for k, v in parameters['system'].items() if k in ['ibrav', 'ecutwfc', 'nat', 'ntyp']}
    return parameters

def normalize_scf(parameters: PwInputParametersDict, pseudo_family: str) -> PwInputParametersDict:
    # Set control.calculation to 'scf'
    parameters["control"]["calculation"] = "scf"
    return normalize_pw(parameters, pseudo_family)

def normalize_nscf(parameters: PwInputParametersDict, pseudo_family: str) -> PwInputParametersDict:
    parameters["control"]["calculation"] = "nscf"
    parameters["control"]["restart_mode"] = "restart"
    parameters["electrons"].pop("startingwfc", None)
    return normalize_pw(parameters, pseudo_family)

def normalize_bands(parameters: PwInputParametersDict, pseudo_family: str) -> PwInputParametersDict:
    parameters["control"]["calculation"] = "bands"
    parameters["control"]["restart_mode"] = "restart"
    parameters["electrons"].pop("startingwfc", None)
    return normalize_pw(parameters, pseudo_family)