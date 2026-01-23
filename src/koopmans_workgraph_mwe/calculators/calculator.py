from typing import Any, Literal, TypedDict

from ase import Atoms
from pint import Quantity

from koopmans_workgraph_mwe.status import Status


class CalculatorInputsDict(TypedDict):
    """Parent dictionary for calculator inputs."""

    kind: Literal["calculator_input"]
    atoms: Atoms


class CalculatorOutputsDict(TypedDict):
    """Parent dictionary for calculator outputs."""

    kind: Literal["calculator_output"]
    walltime: Quantity[Any]
    status: Status
    error_message: str | None
    error_type: type[BaseException] | None
