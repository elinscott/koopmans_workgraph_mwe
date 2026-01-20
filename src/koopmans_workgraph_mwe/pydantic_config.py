from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict

from koopmans_workgraph_mwe.serialization import json_encoders

if TYPE_CHECKING:
    from koopmans_workgraph_mwe.files import LinkedFile

default_config = ConfigDict(extra="forbid",
                            arbitrary_types_allowed=True,
                            strict=False,
                            validate_assignment=True,
                            revalidate_instances='never',
                            json_encoders=json_encoders,
                            )


class BaseModel(_BaseModel):
    """Base model with a modified default configuration."""

    model_config = default_config

    def fields_that_are_files(self) -> list[LinkedFile]:
        """Return the fields of this model that are Files."""
        from koopmans_workgraph_mwe.files import LinkedFile
        file_inputs: list[LinkedFile] = []
        for field in type(self).model_fields:
            value = getattr(self, field)
            if isinstance(value, LinkedFile):
                file_inputs.append(value)
        return file_inputs