from koopmans_workgraph_mwe.parameters.pw import PwInputParametersDict

def normalize_pw(parameters: PwInputParametersDict) -> PwInputParametersDict:
    """Normalize common parameters for PW calculations."""
    parameters['control']['pseudo_dir'] = 'pseudopotentials'
    parameters['control']['outdir'] = 'tmp'
    parameters['control']['prefix'] = 'espresso'
    # For the moment, limited support until we debug the full feature set
    parameters['system'] = {k: v for k, v in parameters['system'].items() if k in ['ibrav', 'ecutwfc', 'nat', 'ntyp']}
    return parameters

def normalize_scf(parameters: PwInputParametersDict) -> PwInputParametersDict:
    # Set control.calculation to 'scf'
    parameters["control"]["calculation"] = "scf"
    return normalize_pw(parameters)

def normalize_nscf(parameters: PwInputParametersDict) -> PwInputParametersDict:
    parameters["control"]["calculation"] = "nscf"
    parameters["control"]["restart_mode"] = "restart"
    return normalize_pw(parameters)

def normalize_bands(parameters: PwInputParametersDict) -> PwInputParametersDict:
    parameters["control"]["calculation"] = "bands"
    parameters["control"]["restart_mode"] = "restart"
    return normalize_pw(parameters)
