from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.cell import Cell
from ase.dft.kpoints import BandPath
from ase.spectrum.band_structure import BandStructure
from ase.spectrum.doscollection import GridDOSCollection
from ase.spectrum.dosdata import GridDOSData

from koopmans_workgraph_mwe.utils import remove_numpy_from_obj


def serialize_ase_atoms(self: Atoms) -> dict[str, Any]:
    dct: dict[str, Any] = self.todict()  # type: ignore[no-untyped-call]
    return remove_numpy_from_obj(dct)


def deserialize_ase_atoms(value: Any) -> Atoms:
    if isinstance(value, Atoms):
        return value
    elif isinstance(value, dict):
        atoms: Atoms = Atoms.fromdict(value)  # type: ignore[no-untyped-call]
        return atoms
    raise ValueError(f"Cannot deserialize an ASE Atoms from a {type(value)}")


def serialize_ase_cell(cell: Cell) -> dict[str, Any]:
    dct: dict[str, Any] = cell.todict()  # type: ignore[no-untyped-call]
    return remove_numpy_from_obj(dct)


def deserialize_ase_cell(value: Any) -> Cell:
    if isinstance(value, Cell):
        return value
    elif isinstance(value, dict):
        return Cell(**value)  # type: ignore[no-untyped-call]
    raise ValueError(f"Cannot deserialize an ASE Cell from a {type(value)}")


def serialize_ase_bandpath(bandpath: BandPath) -> dict[str, Any]:
    dct: dict[str, Any] = bandpath.todict()  # type: ignore[no-untyped-call]
    dct.pop('labelseq')
    dct['cell'] = serialize_ase_cell(dct['cell'])
    return remove_numpy_from_obj(dct)


def deserialize_ase_bandpath(value: Any) -> BandPath:
    if isinstance(value, BandPath):
        return value
    elif isinstance(value, dict):
        value['cell'] = deserialize_ase_cell(value['cell'])
        return BandPath(**value)  # type: ignore[no-untyped-call]
    raise ValueError(f"Cannot deserialize an ASE BandPath from a {type(value)}")


def serialize_ase_bandstructure(bandstructure: BandStructure) -> dict[str, Any]:
    dct: dict[str, Any] = bandstructure.todict()  # type: ignore[no-untyped-call]
    dct['path'] = serialize_ase_bandpath(dct['path'])
    return remove_numpy_from_obj(dct)


def deserialize_ase_bandstructure(value: Any) -> BandStructure:
    if isinstance(value, BandStructure):
        return value
    elif isinstance(value, dict):
        value['path'] = deserialize_ase_bandpath(value['path'])
        return BandStructure(**value)  # type: ignore[no-untyped-call]
    raise ValueError(f"Cannot deserialize an ASE BandStructure from a {type(value)}")


def serialize_numpy(array: npt.NDArray[np.float64]) -> list[Any]:
    return array.tolist()  # type: ignore[no-any-return]


def deserialize_numpy(value: Any) -> npt.NDArray[np.float64]:
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value, dtype=np.float64)


def serialize_ase_griddosdata(data: GridDOSData) -> dict[str, Any]:
    info: dict[str, Any] = data.info  # type: ignore[has-type]
    return {
        "energies": data.get_energies().tolist(),
        "weights": data.get_weights().tolist(),
        "info": info,
    }


def deserialize_ase_griddosdata(value: Any) -> GridDOSData:
    if isinstance(value, GridDOSData):
        return value
    elif isinstance(value, dict):
        return GridDOSData(**value)
    raise ValueError(f"Cannot deserialize an ASE GridDOSData from a {type(value)}")


def serialize_ase_griddoscollection(dos: GridDOSCollection) -> dict[str, Any]:
    """Serialize an ASE GridDOSCollection by reconstructing the dos_series objects required to initialize it."""
    info_list: list[dict[str, Any]] = dos._info  # type: ignore[unused-ignore, attr-defined]
    dct = {
        "dos_series": [serialize_ase_griddosdata(GridDOSData(dos.get_energies(), data, info))
                       for data, info in zip(dos.get_all_weights(), info_list)]
    }
    return remove_numpy_from_obj(dct)


def deserialize_ase_griddoscollection(value: Any) -> GridDOSCollection:
    """Deserialize an ASE GridDOSCollection from a dictionary."""
    if isinstance(value, GridDOSCollection):
        return value
    elif isinstance(value, dict):
        dos_series = [deserialize_ase_griddosdata(data) for data in value['dos_series']]
        return GridDOSCollection(dos_series=dos_series)
    raise ValueError(f"Cannot deserialize an ASE GridDOSCollection from a {type(value)}")


json_encoders: dict[type[object], Callable[[Any], Any]] = {
    Atoms: serialize_ase_atoms,
    BandPath: serialize_ase_bandpath,
    BandStructure: serialize_ase_bandstructure,
    Cell: serialize_ase_cell,
    np.ndarray: serialize_numpy,
    GridDOSCollection: serialize_ase_griddoscollection,
}
