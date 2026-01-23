from typing import Any, TypedDict

from ase import Atoms
from pint import Quantity

from koopmans_workgraph_mwe.status import Status


class CalculatorInputsDict(TypedDict):
    """Parent dictionary for calculator inputs."""

    atoms: Atoms


class CalculatorOutputsDict(TypedDict):
    """Parent dictionary for calculator outputs."""

    walltime: Quantity[Any]
    status: Status
    error_message: str | None
    error_type: type[BaseException] | None
