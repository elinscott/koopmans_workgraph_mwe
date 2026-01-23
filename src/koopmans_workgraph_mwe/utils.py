import contextlib
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any, TypeVar, overload

import numpy as np
from numpy import typing as npt

T = TypeVar('T')

@overload
def remove_numpy_from_obj(obj: npt.NDArray[Any]) -> list[Any]: ...

@overload
def remove_numpy_from_obj(obj: T) -> T: ...

def remove_numpy_from_obj(obj: Any) -> Any:
    """
    Recursively convert NumPy arrays to Python lists inside nested structures.
    Supports dict, list, tuple, set, and nested combinations.
    """
    if isinstance(obj, dict):
        return {k: remove_numpy_from_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [remove_numpy_from_obj(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(remove_numpy_from_obj(v) for v in obj)
    elif isinstance(obj, set):
        return {remove_numpy_from_obj(v) for v in obj}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def remove_null_from_obj(obj: T) -> T:
    null_values: tuple[Any, ...] = (None, '', [], {}, ())
    if isinstance(obj, dict):
        return {k: remove_null_from_obj(v) for k, v in obj.items() if v not in null_values}
    elif isinstance(obj, list):
        return [remove_null_from_obj(v) for v in obj if v not in null_values]
    elif isinstance(obj, tuple):
        return tuple(remove_null_from_obj(v) for v in obj if v not in null_values)
    elif isinstance(obj, set):
        return {remove_null_from_obj(v) for v in obj if v not in null_values}
    else:
        return obj

def chdir_logic(path: Path | str) -> Generator[None, None, None]:
    """Change the working directory.

    Allows for the context "with chdir(path)". All code within this
    context will be executed in the directory "path"
    """
    # Ensure path is a Path object
    if not isinstance(path, Path):
        path = Path(path)

    this_dir = Path.cwd()

    # Create path if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    # Move to the directory
    os.chdir(path)
    try:
        yield
    finally:
        # Return to the original directory
        os.chdir(this_dir)


@contextlib.contextmanager
def chdir(path: Path | str) -> Generator[None, None, None]:
    """Return a context that changes the working directory (returns to the original directory when done)."""
    return chdir_logic(path)
