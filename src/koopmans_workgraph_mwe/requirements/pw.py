from collections.abc import Callable
from functools import partial
from typing import Any


def require_calculation_is(parameters: dict[str, Any], value: str) -> None:
    if parameters["control"]["calculation"] != value:
        raise ValueError(f'Calculation type must be "{value}"')

require_calculation_is_scf = partial(require_calculation_is, value='scf')
require_calculation_is_nscf = partial(require_calculation_is, value='nscf')
require_calculation_is_bands = partial(require_calculation_is, value='bands')

def require_restart(parameters: dict[str, Any]) -> None:
    if parameters["control"]["restart_mode"] != "restart":
        raise ValueError('`restart_mode` must be set to "restart"')

SCF_REQUIREMENTS: list[Callable[[dict[str, Any]], None]] = [
    require_calculation_is_scf,
]

NSCF_REQUIREMENTS: list[Callable[[dict[str, Any]], None]] = [
    require_calculation_is_nscf,
    require_restart,
]

BANDS_REQUIREMENTS: list[Callable[[dict[str, Any]], None]] = [
    require_calculation_is_bands,
    require_restart,
]

def require_all(parameters: dict[str, Any], requirements: list[Callable[[dict[str, Any]], None]]) -> None:
    for requirement in requirements:
        requirement(parameters)
