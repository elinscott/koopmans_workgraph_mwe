from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict

from koopmans_workgraph_mwe.serialization import json_encoders

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model to a dict compatible with aiida-pythonjob's JsonableData."""
        return {
            '@class': self.__class__.__name__,
            '@module': self.__class__.__module__,
            '@data': self.model_dump(mode='json'),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize the model from a dict created by to_dict."""
        if '@data' in data:
            return cls.model_validate(data['@data'])
        return cls.model_validate(data)
