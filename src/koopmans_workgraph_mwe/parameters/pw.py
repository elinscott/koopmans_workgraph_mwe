
from typing import Any, Literal, TypedDict

from pydantic import Field, field_validator
from pydantic_espresso.models.qe_7_4.pw import ControlNamelist as _ControlNamelist
from pydantic_espresso.models.qe_7_4.pw import ElectronsNamelist, SystemNamelist

from koopmans_workgraph_mwe.pydantic_config import BaseModel


class ControlNamelist(_ControlNamelist):
    @field_validator('verbosity', mode='before')
    @classmethod
    def enforce_high_verbosity(cls, v: Any) -> Literal['high']:
        """High verbosity is required to guarantee that all bands will be printed."""
        return 'high'


class PwInputParametersModel(BaseModel):
    control: ControlNamelist = Field(default_factory=lambda: ControlNamelist())
    system: SystemNamelist = Field(default_factory=lambda: SystemNamelist())
    electrons: ElectronsNamelist = Field(default_factory=lambda: ElectronsNamelist())

class PwInputParametersDict(TypedDict):
    control: dict[str, Any]
    system: dict[str, Any]
    electrons: dict[str, Any]