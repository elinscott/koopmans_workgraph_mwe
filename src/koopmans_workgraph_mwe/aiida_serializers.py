"""AiiDA serializers for third-party types that don't have to_dict/from_dict methods."""

import importlib
from pathlib import Path
from typing import Any

from aiida import orm
from ase.dft.kpoints import BandPath
from ase.spectrum.band_structure import BandStructure

from koopmans_workgraph_mwe.pydantic_config import BaseModel
from koopmans_workgraph_mwe.serialization import (
    deserialize_ase_bandpath,
    deserialize_ase_bandstructure,
    serialize_ase_bandpath,
    serialize_ase_bandstructure,
)


class BandPathData(orm.Dict):
    """AiiDA Data class for storing ASE BandPath objects."""

    def __init__(self, value: BandPath | dict[str, Any] | None = None, **kwargs: Any) -> None:
        if isinstance(value, BandPath):
            super().__init__(dict=serialize_ase_bandpath(value), **kwargs)  # type: ignore[no-untyped-call]
        elif isinstance(value, dict):
            super().__init__(dict=value, **kwargs)  # type: ignore[no-untyped-call]
        else:
            super().__init__(**kwargs)  # type: ignore[no-untyped-call]

    def get_object(self) -> BandPath:
        """Reconstruct the BandPath from stored data."""
        dct = self.get_dict()  # type: ignore[no-untyped-call]
        return deserialize_ase_bandpath(dct)


def bandpath_to_aiida(bandpath: BandPath, user: orm.User | None = None) -> BandPathData:
    """Convert a BandPath to an AiiDA Data node."""
    return BandPathData(bandpath, user=user)


class BandStructureData(orm.Dict):
    """AiiDA Data class for storing ASE BandStructure objects."""

    def __init__(self, value: BandStructure | dict[str, Any] | None = None, **kwargs: Any) -> None:
        if isinstance(value, BandStructure):
            super().__init__(dict=serialize_ase_bandstructure(value), **kwargs)  # type: ignore[no-untyped-call]
        elif isinstance(value, dict):
            super().__init__(dict=value, **kwargs)  # type: ignore[no-untyped-call]
        else:
            super().__init__(**kwargs)  # type: ignore[no-untyped-call]

    def get_object(self) -> BandStructure:
        """Reconstruct the BandStructure from stored data."""
        dct = self.get_dict()  # type: ignore[no-untyped-call]
        return deserialize_ase_bandstructure(dct)


def bandstructure_to_aiida(bandstructure: BandStructure, user: orm.User | None = None) -> BandStructureData:
    """Convert a BandStructure to an AiiDA Data node."""
    return BandStructureData(bandstructure, user=user)


def path_to_aiida(path: Path, user: orm.User | None = None) -> orm.Str:
    """Convert a Path to an AiiDA Str node."""
    return orm.Str(str(path), user=user)  # type: ignore[no-untyped-call]


class BaseModelData(orm.Dict):
    """AiiDA Data class for storing pydantic BaseModel objects."""

    def __init__(self, value: BaseModel | dict[str, Any] | None = None, **kwargs: Any) -> None:
        if isinstance(value, BaseModel):
            data = {
                '@class': f'{value.__class__.__module__}.{value.__class__.__name__}',
                '@data': value.model_dump(mode='json'),
            }
            super().__init__(dict=data, **kwargs)  # type: ignore[no-untyped-call]
        elif isinstance(value, dict):
            super().__init__(dict=value, **kwargs)  # type: ignore[no-untyped-call]
        else:
            super().__init__(**kwargs)  # type: ignore[no-untyped-call]

    def get_object(self) -> BaseModel:
        """Reconstruct the BaseModel from stored data."""
        data = self.get_dict()
        class_path = data['@class']
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls.model_validate(data['@data'])


def basemodel_to_aiida(model: BaseModel, user: orm.User | None = None) -> BaseModelData:
    """Convert a BaseModel to an AiiDA Data node."""
    return BaseModelData(model, user=user)
