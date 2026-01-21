from typing import Any
import numpy as np
import contextlib
import os
from pathlib import Path
import inspect
from functools import wraps

def remove_numpy_from_dict(obj: Any) -> Any:
    """
    Recursively convert NumPy arrays to Python lists inside nested structures.
    Supports dict, list, tuple, set, and nested combinations.
    """
    if isinstance(obj, dict):
        return {k: remove_numpy_from_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [remove_numpy_from_dict(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(remove_numpy_from_dict(v) for v in obj)
    elif isinstance(obj, set):
        return {remove_numpy_from_dict(v) for v in obj}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def chdir_logic(path: Path | str):
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
def chdir(path: Path | str):
    """Return a context that changes the working directory (returns to the original directory when done)."""
    return chdir_logic(path)


def with_model_signature(model_cls):
    def decorator(func):
        sig = inspect.signature(model_cls)
        return_annotation = inspect.signature(func).return_annotation

        @wraps(func)
        def wrapper(*args, **kwargs):
            inputs = model_cls(**kwargs)
            return func(inputs)

        wrapper.__signature__ = sig.replace(return_annotation=return_annotation)
        return wrapper
    return decorator
