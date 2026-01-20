from typing import Annotated

from ase import Atoms
from pydantic_pint import PydanticPintQuantity
from pint import Quantity

from koopmans_workgraph_mwe.pydantic_config import BaseModel
from koopmans_workgraph_mwe.status import Status



class CalculatorInputs(BaseModel):
    """Parent input model for calculators."""
    atoms: Atoms


class CalculatorOutputs(BaseModel):
    """Parent output model for calculators."""

    walltime: Annotated[Quantity, PydanticPintQuantity("s")]
    status: Status
    error_message: str | None = None
    error_type: type | None = None
