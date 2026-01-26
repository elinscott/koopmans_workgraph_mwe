from pathlib import Path
import shutil
from koopmans_workgraph_mwe.files import DirectoryDict, SingleFileDict

def _dict_to_path(dct: SingleFileDict | DirectoryDict) -> Path:
    return Path(dct["parent_uid"]) / dct["path"]

def file_exists(path: SingleFileDict | DirectoryDict) -> bool:
    return _dict_to_path(path).exists() or _dict_to_path(path).is_symlink()


def write_to_file(content: str, path: SingleFileDict | DirectoryDict) -> None:
    """Write content to a file."""
    with open(_dict_to_path(path), 'w') as f:
        f.write(content)


def delete_file(path: SingleFileDict | DirectoryDict) -> None:
    explicit_path = _dict_to_path(path)
    if explicit_path.is_dir():
        shutil.rmtree(explicit_path)
    else:
        explicit_path.unlink()


def copy_file(src: SingleFileDict | DirectoryDict, dest: SingleFileDict | DirectoryDict) -> None:
    src_path = _dict_to_path(src)
    dest_path = _dict_to_path(dest)
    if src_path.is_dir():
        shutil.copytree(src_path, dest_path)
    else:
        shutil.copy(src_path, dest_path)


def is_dir(path: SingleFileDict | DirectoryDict) -> bool:
    explicit_path = _dict_to_path(path)
    return explicit_path.is_dir()


def mkdir(path: DirectoryDict, parents: bool = False, exist_ok: bool = False) -> None:
    """Create a directory at the given path."""
    dir_path = _dict_to_path(path)
    dir_path.mkdir(parents=parents, exist_ok=exist_ok)


def link_file(
    src: SingleFileDict | DirectoryDict,
    dest: SingleFileDict | DirectoryDict,
    recursive: bool = False,
) -> None:
    """Link a file from src to dest."""
    src_path = _dict_to_path(src)
    dest_path = _dict_to_path(dest)
    link_path(src_path, dest_path, recursive=recursive)

def link_path(
    src: Path,
    dest: Path,
    recursive: bool = False,
) -> None:
    """Link a file from src to dest."""
    relative_path = src.resolve().relative_to(dest.parent.resolve(), walk_up=True)
    if recursive:
        for child in src.rglob('*'):
            if child.is_dir():
                continue
            child_dest = dest / child.relative_to(src)
            if not child_dest.parent.exists():
                child_dest.parent.mkdir(parents=True, exist_ok=True)
            link_path(child, dest / child.relative_to(src), recursive=False)
    else:
        dest.symlink_to(relative_path)