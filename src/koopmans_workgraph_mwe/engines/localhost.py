import shutil
from pathlib import Path

from koopmans_workgraph_mwe.engines.engine import EngineABC
from koopmans_workgraph_mwe.files import DirectoryDict, SingleFileDict


def file_exists(path: SingleFileDict | DirectoryDict) -> bool:
    return Path(path["uid"]).exists() or Path(path["uid"]).is_symlink()


def write_to_file(content: str, path: SingleFileDict | DirectoryDict) -> None:
    """Write content to a file."""
    with open(path["uid"], 'w') as f:
        f.write(content)


def delete_file(path: SingleFileDict | DirectoryDict) -> None:
    explicit_path = Path(path["uid"])
    if explicit_path.is_dir():
        shutil.rmtree(explicit_path)
    else:
        explicit_path.unlink()


def copy_file(src: SingleFileDict | DirectoryDict, dest: SingleFileDict | DirectoryDict) -> None:
    src_path = Path(src["uid"])
    dest_path = Path(dest["uid"])
    if src_path.is_dir():
        shutil.copytree(src_path, dest_path)
    else:
        shutil.copy(src_path, dest_path)


def is_dir(path: SingleFileDict | DirectoryDict) -> bool:
    explicit_path = Path(path["uid"])
    return explicit_path.is_dir()


def mkdir(path: DirectoryDict, parents: bool = False, exist_ok: bool = False) -> None:
    """Create a directory at the given path."""
    dir_path = Path(path["uid"])
    dir_path.mkdir(parents=parents, exist_ok=exist_ok)


def link_file(
    src: SingleFileDict | DirectoryDict,
    dest: SingleFileDict | DirectoryDict,
    recursive: bool = False,
) -> None:
    """Link a file from src to dest."""
    src_path = Path(src["uid"])
    dest_path = Path(dest["uid"])
    relative_path = src_path.resolve().relative_to(dest_path.parent.resolve(), walk_up=True)
    if is_dir(src):
        dest_path.symlink_to(relative_path, target_is_directory=True)
    else:
        dest_path.symlink_to(relative_path)


class LocalhostEngine(EngineABC):
    """Engine for running calculations on the local machine."""

    def file_exists(self, path: SingleFileDict | DirectoryDict) -> bool:
        return file_exists(path)

    def delete_file(self, path: SingleFileDict | DirectoryDict) -> None:
        delete_file(path)

    def _copy_file(
        self, src: SingleFileDict | DirectoryDict, dest: SingleFileDict | DirectoryDict
    ) -> None:
        copy_file(src, dest)

    def is_dir(self, path: SingleFileDict | DirectoryDict) -> bool:
        return is_dir(path)

    def mkdir(self, path: DirectoryDict, parents: bool = False, exist_ok: bool = False) -> None:
        mkdir(path, parents=parents, exist_ok=exist_ok)

    def _link_file(
        self,
        src: SingleFileDict | DirectoryDict,
        dest: SingleFileDict | DirectoryDict,
        recursive: bool = False,
    ) -> None:
        link_file(src, dest, recursive=recursive)

    def write_to_file(self, content: str, path: SingleFileDict) -> None:
        write_to_file(content, path)
