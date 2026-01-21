from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Annotated, Callable, TypeVar

from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict, PlainValidator
from pydantic.fields import FieldInfo

from koopmans_workgraph_mwe.serialization import json_encoders
from node_graph.socket import TaggedValue

if TYPE_CHECKING:
    from koopmans_workgraph_mwe.files import LinkedFile

T = TypeVar('T')


def _validate_str_or_tagged_value(v: object) -> str | TaggedValue:
    """Validate that v is either a str or a TaggedValue wrapping a str."""
    if isinstance(v, TaggedValue):
        if not isinstance(v.__wrapped__, str):
            raise ValueError(f'Expected str, got {type(v.__wrapped__).__name__}')
        return v
    if isinstance(v, str):
        return v
    raise ValueError(f'Expected str or TaggedValue[str], got {type(v).__name__}')


# A string type that accepts both str and TaggedValue[str], keeping TaggedValue intact.
FlexibleStr = Annotated[str, PlainValidator(_validate_str_or_tagged_value)]


def _patch_str_fields_to_flexible(*models: type) -> None:
    """Patch str-annotated fields in models to use FlexibleStr instead."""
    for model in models:
        for field_name, field_info in model.model_fields.items():
            if field_info.annotation is str:
                field_info.annotation = FlexibleStr
        model.model_rebuild(force=True)


# Patch pydantic_espresso models to accept TaggedValue for str fields
from pydantic_espresso.models.qe_7_4.pw import (
    ControlNamelist as _ControlNamelist,
    SystemNamelist as _SystemNamelist,
)
_patch_str_fields_to_flexible(_ControlNamelist, _SystemNamelist)


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
    
    def to_dict(self):
        """Serialize the model to a dict compatible with aiida-pythonjob's JsonableData."""
        return {
            '@class': self.__class__.__name__,
            '@module': self.__class__.__module__,
            '@data': self.model_dump(mode='json'),
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize the model from a dict created by to_dict."""
        if '@data' in data:
            return cls.model_validate(data['@data'])
        return cls.model_validate(data)