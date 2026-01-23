"""Defines a "kpoints" class that stores information related to k-points.

Written by Edward Linscott, July 2022
"""


from __future__ import annotations

import copy
import itertools
from typing import Annotated, Any, Literal, TypedDict

import numpy as np
from ase.cell import Cell
from ase.dft.kpoints import BandPath, kpoint_convert, parse_path_string, resolve_kpt_path_string
from pydantic import Field, field_validator, model_validator

from koopmans_workgraph_mwe.pydantic_config import BaseModel

ZeroOrOne = Literal[0, 1]
Fractional = Annotated[float, Field(gt=-1, le=1)]
ExplicitKpoint = tuple[float, float, float, Fractional]
GRID_DESCRIPTION = "A list of three integers specifying the shape of the regular grid of k-points."
OFFSET_DESCRIPTION = "A list of three integers, either zero or one. If one, the regular k-point grid is offset by half a grid step in that dimension."
OFFSET_NSCF_DESCRIPTION = "A list of three numbers, within the interval (-1, 1]. It has the same meaning of offset, but it is applied only to the grid of non-self-consistent pw.x calculations, and permits any fractional offset."
PATH_DESCRIPTION = "An ASE ``BandPath`` object specifying the k-path as defined by the special points of the Bravais lattice."
GAMMA_ONLY_DESCRIPTION = "True if the calculation is only sampling the gamma point."

class Kpoints(TypedDict):
    kind: Literal["kpoints"]
    gamma_only: bool
    grid: tuple[int, int, int]
    offset: tuple[ZeroOrOne, ZeroOrOne, ZeroOrOne]
    offset_nscf: tuple[Fractional, Fractional, Fractional]
    path: BandPath | None
    explicit_grid: list[ExplicitKpoint]

def kpoints_factory(
    gamma_only: bool,
    grid: tuple[int, int, int],
    offset: tuple[ZeroOrOne, ZeroOrOne, ZeroOrOne],
    offset_nscf: tuple[float, float, float],
    path: BandPath | None,
    explicit_grid: list[ExplicitKpoint] | None = None,
) -> Kpoints:
    """Factory function to create a Kpoints TypedDict."""
    return Kpoints(
        kind="kpoints",
        gamma_only=gamma_only,
        grid=grid,
        offset=offset,
        offset_nscf=offset_nscf,
        path=path,
        explicit_grid=explicit_grid or [],
    )

_GAMMA_GRID: tuple[int, int, int] = (1, 1, 1)
_GAMMA_OFFSET: tuple[ZeroOrOne, ZeroOrOne, ZeroOrOne] = (0, 0, 0)
_GAMMA_OFFSET_NSCF: tuple[float, float, float] = (0.0, 0.0, 0.0)


class GammaOnlyKpointsModel(BaseModel):
    """K-point information for a gamma-only calculation."""

    gamma_only: Literal[True] = True
    grid: tuple[int, int, int] = _GAMMA_GRID
    offset: tuple[ZeroOrOne, ZeroOrOne, ZeroOrOne] = _GAMMA_OFFSET
    offset_nscf: tuple[float, float, float] = _GAMMA_OFFSET_NSCF
    path: None = None

    @model_validator(mode="after")
    def enforce_gamma_values(self) -> GammaOnlyKpointsModel:
        """Enforce that gamma-only values are fixed."""
        if self.grid != _GAMMA_GRID:
            raise ValueError(f"grid must be {_GAMMA_GRID} for gamma-only calculations")
        if self.offset != _GAMMA_OFFSET:
            raise ValueError(f"offset must be {_GAMMA_OFFSET} for gamma-only calculations")
        if self.offset_nscf != _GAMMA_OFFSET_NSCF:
            raise ValueError(f"offset_nscf must be {_GAMMA_OFFSET_NSCF} for gamma-only calculations")
        return self


class NonGammaKpointsModel(BaseModel):
    """K-point information for non-gamma-only calculations."""

    grid: tuple[int, int, int] = Field(description=GRID_DESCRIPTION)
    offset: tuple[ZeroOrOne, ZeroOrOne, ZeroOrOne] = Field(default=(0, 0, 0), description=OFFSET_DESCRIPTION)
    offset_nscf: tuple[float, float, float] = Field(default=(0.0, 0.0, 0.0), description=OFFSET_NSCF_DESCRIPTION)
    path: BandPath = Field(description=PATH_DESCRIPTION)
    gamma_only: Literal[False] = Field(False, description=GAMMA_ONLY_DESCRIPTION)
    explicit_grid: list[ExplicitKpoint] = Field(description="An explicit list of k-points with weights.")

    @field_validator("path", mode="before")
    @classmethod
    def coerce_path(cls, value: Any) -> BandPath:
        if isinstance(value, dict):
            value['cell'] = Cell(**value['cell'])  # type: ignore[no-untyped-call]
            return BandPath(**value)  # type: ignore[no-untyped-call]
        elif isinstance(value, BandPath):
            return value
        raise ValueError("path must be a BandPath object or a dictionary representing one")

    @model_validator(mode="before")
    @classmethod
    def compute_explicit_grid(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Compute the explicit grid of k-points based on the grid and offset."""
        if 'explicit_grid' in data:
            return data

        if 'offset' not in data:
            data['offset'] = (0, 0, 0)

        pts = []
        for i, j, k in itertools.product(*[range(x) for x in data['grid']]):
            pt = (
                (i + 0.5 * data['offset'][0]) / data['grid'][0],
                (j + 0.5 * data['offset'][1]) / data['grid'][1],
                (k + 0.5 * data['offset'][2]) / data['grid'][2],
            )
            pts.append(pt)
        # Wrap to (-0.5, 0.5])
        pts_array = np.array(pts, dtype=np.float64)
        pts_array[pts_array > 0.5] -= 1.0

        # Add weights as a fourth column
        weights = np.ones((len(pts_array), 1)) / len(pts_array)
        pts_array = np.concatenate([pts_array, weights], axis=1)
        data['explicit_grid'] = pts_array.tolist()
        return data


KpointsModel = Annotated[
    GammaOnlyKpointsModel | NonGammaKpointsModel,
    Field(discriminator="gamma_only"),
]


def kpoints_from_path_as_string(grid: tuple[int, int, int],
                                path: str,
                                cell: Cell,
                                offset: tuple[ZeroOrOne, ZeroOrOne, ZeroOrOne],
                                offset_nscf: tuple[Fractional, Fractional, Fractional],
                                density: float = 10.0) -> Kpoints:
    """Create a NonGammaKpoints object using a string for the path instead of a `BandPath` object."""
    bandpath = convert_kpath_str_to_bandpath(path, cell, density)
    return kpoints_factory(gamma_only=False, grid=grid, path=bandpath, offset=offset, offset_nscf=offset_nscf)


def convert_kpath_str_to_bandpath(path: str, cell: Cell, density: float | None = None) -> BandPath:
    """Convert a string of high-symmetry k-points to a BandPath object."""
    bandpath: BandPath = cell.bandpath(eps=1e-10)  # type: ignore[no-untyped-call]
    special_points: dict[str, Any] = bandpath.special_points  # type: ignore[no-untyped-call]
    special_points_on_path: set[str] = set([x for y in parse_path_string(path) for x in y])  # type: ignore[no-untyped-call]
    for sp in special_points_on_path:
        if sp not in special_points.keys():
            raise KeyError(
                f'The path provided to convert_kpath_str_to_bandpath contains a special point ({sp}) '
                f'that is not in the set of special points for this cell ({", ".join(special_points.keys())})')
    bp: BandPath = BandPath(cell=cell, path=path, special_points=special_points)  # type: ignore[no-untyped-call]
    if density is not None and len(path) > 1:
        bp = bp.interpolate(density=density)  # type: ignore[no-untyped-call]
    return bp


def kpath_length(path: BandPath) -> float:
    """Calculate the length of a k-path."""
    _, paths = resolve_kpt_path_string(path.path, path.special_points)  # type: ignore[no-untyped-call]
    points = np.concatenate(paths)
    dists = points[1:] - points[:-1]
    lengths: list[float] = [float(np.linalg.norm(d)) for d in kpoint_convert(path.cell, skpts_kc=dists)]  # type: ignore[no-untyped-call]

    i = 0
    for p in paths[:-1]:
        i += len(p)
        lengths[i - 1] = 0.0

    return float(np.sum(lengths))


def kpath_to_dict(path: BandPath) -> dict[str, Any]:
    """Convert a BandPath object to a dictionary."""
    dct: dict[str, Any] = {}
    dct['path'] = path.path
    dct['cell'] = path.cell.todict()  # type: ignore[no-untyped-call]
    if len(path.path) > 1:
        attempts = 0
        density_guess = len(path.kpts) / kpath_length(path)
        density_max: float | None = None
        density_min: float | None = None

        # Super-dumb bisection approach to obtain a density that gives the correct nunber of kpoints.
        # We have to resort to this because the length of paths produced by BandPath().interpolate(density=...)
        # is unreliable
        while attempts < 100:
            dct['density'] = density_guess
            new_path = dict_to_kpath(dct)
            if len(new_path.kpts) == len(path.kpts):
                break
            elif len(new_path.kpts) < len(path.kpts):
                density_min = density_guess
                density_guess = density_min + 1.0
            else:
                density_max = density_guess
                density_guess = density_max - 1.0
            if density_min and density_max:
                density_guess = (density_max + density_min) / 2
            attempts += 1
        if attempts >= 100:
            raise ValueError('Search failed')
    return dct


def dict_to_kpath(dct: dict[str, Any]) -> BandPath:
    """Convert a dictionary to a BandPath object."""
    dct = copy.deepcopy(dct)
    density = dct.pop('density', None)
    cell: Cell = Cell(dct.pop('cell')['array'])  # type: ignore[no-untyped-call]
    bp: BandPath = BandPath(cell=cell, special_points=cell.bandpath(eps=1e-10).special_points,  # type: ignore[no-untyped-call, union-attr]
                    **dct)
    return bp.interpolate(density=density)  # type: ignore[no-untyped-call]
