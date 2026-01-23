from koopmans_workgraph_mwe.engines.engine import EngineABC
from koopmans_workgraph_mwe.files import DirectoryDict, SingleFileDict
from koopmans_workgraph_mwe.os.local import (
    copy_file,
    delete_file,
    file_exists,
    is_dir,
    link_file,
    mkdir,
    write_to_file,)



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
