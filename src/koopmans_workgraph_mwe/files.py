"""A class for representing files in a general way by tethering them to a parent process."""

from typing import Literal, TypedDict
from pathlib import Path

from koopmans_workgraph_mwe.pydantic_config import BaseModel

# from __future__ import annotations
#
# from abc import ABC, abstractmethod
# from collections.abc import Generator
# from pathlib import Path
# from typing import Annotated, Any, Self
#
# from koopmans_workgraph_mwe.pydantic_config import BaseModel
# from pydantic import BeforeValidator


# class FileABC(BaseModel, ABC):
#     """An abstract base class for representing a file."""
#
#     path: Path
#
#     @property
#     def stem(self) -> str:
#         """Return the stem of the file, replicating the behavior of Path.stem."""
#         return self.path.stem
#
#     @property
#     def suffix(self) -> str:
#         """Return the file suffix, replicating the behavior of Path.suffix."""
#         return self.path.suffix
#
#     def rename(self, dst: Self) -> None:
#         """Rename this file, replicating the behavior of Path.rename."""
#         self.path.rename(dst.path)
#
#     @property
#     def name(self) -> str:
#         """Return the name of the file, replicating the behavior of Path.name."""
#         return self.path.name
#
#     @abstractmethod
#     def with_stem(self, stem: str) -> Self:
#         """Return a new object with the given stem."""
#
#     @abstractmethod
#     def with_suffix(self, suffix: str) -> Self:
#         """Return a new File with the given suffix."""
#
#
# class File(FileABC):
#     """An abstract way of representing a file.
#
#     Because a file may not exist locally (specifically, when koopmans is run with AiiDA), we need a way of
#     referring to a file that is more general than an absolute path. This class achieves this by storing a
#     file as a parent_process (which is a Process, Calculator, or some other object that exists in a directory known
#     to koopmans/AiiDA) and a name (which is the path of the file relative to the parent_process's directory).
#
#     We also need to delegate file creation/modification/deletion to the engine.
#     """
#
#     parent_process_uid: str
#
#     def with_stem(self, stem: str) -> File:
#         """Return a new File with the given stem."""
#         new = self.path.with_stem(stem)
#         return File(parent_process_uid=self.parent_process_uid, path=new)
#
#     def with_suffix(self, suffix: str) -> File:
#         """Return a new File with the given suffix."""
#         new = self.path.with_suffix(suffix)
#         return File(parent_process_uid=self.parent_process_uid, path=new)
#
#     def __truediv__(self, other: Any) -> File:
#         assert isinstance(other, Path) or isinstance(other, str)
#         return File(parent_process_uid=self.parent_process_uid, path=self.path / other)
#
#
# class LocalFile(FileABC):
#     """A local file.
#
#     This class simply wraps the Path class.
#     """
#
#     def with_stem(self, stem: str) -> LocalFile:
#         """Return a new object with the given stem."""
#         new = self.path.with_stem(stem)
#         return LocalFile(path=new)
#
#     def with_suffix(self, suffix: str) -> LocalFile:
#         """Return a new File with the given suffix."""
#         new = self.path.with_suffix(suffix)
#         return LocalFile(path=new)
#
#     def __truediv__(self, other: Any) -> LocalFile:
#         assert isinstance(other, Path) or isinstance(other, str)
#         return LocalFile(path=self.path / other)
#
#     @property
#     def parent(self) -> LocalFile:
#         """Return the parent directory of this file."""
#         return LocalFile(path=self.path.parent)
#
#     @property
#     def parents(self) -> Generator[LocalFile, None, None]:
#         """Return all parent directories of this file."""
#         for parent in self.path.parents:
#             yield LocalFile(path=parent)
#
#
# class LinkedFile(BaseModel):
#     """A link from a source to a destination file.
#
#     This can be either a symlink or a copy (see `FileViaSymlink`, `FileViaRecursiveSymlink`, and `FileViaCopy` below).
#     """
#
#     dest: Path
#     src: File | LocalFile
#     symlink: bool = False
#     recursive_symlink: bool = False
#     overwrite: bool = False
#
#     @property
#     def stem(self) -> str:
#         """Return the stem of the destination file, replicating the behavior of Path.stem."""
#         return self.dest.stem
#
#     @property
#     def suffix(self) -> str:
#         """Return the suffix of the destination file, replicating the behavior of Path.suffix."""
#         return self.dest.suffix
#
#     @property
#     def name(self) -> str:
#         """Return the name of the destination file, replicating the behavior of Path.name."""
#         return self.dest.name
#
#
# def as_linkedfile(*, symlink: bool, recursive_symlink: bool, overwrite: bool) -> BeforeValidator:
#     """Return a validator that allows conversion of File to LinkedFile.
#
#     Allows specification of symlink, recursive_symlink, and overwrite parameters.
#     """
#
#     def convert(v: Any) -> LinkedFile | None:
#         if v is None or isinstance(v, LinkedFile):
#             return v
#         if isinstance(v, File | LocalFile):
#             return LinkedFile(dest=v.path, src=v, symlink=symlink, recursive_symlink=recursive_symlink,
#                               overwrite=overwrite)
#         raise TypeError("must be File or LinkedFile")
#     return BeforeValidator(convert)
#
# # Shortcuts for common LinkedFile types
# FileViaSymlink = Annotated[LinkedFile, as_linkedfile(symlink=True, recursive_symlink=False, overwrite=False)]
# FileViaRecursiveSymlink = Annotated[LinkedFile, as_linkedfile(symlink=True, recursive_symlink=True, overwrite=False)]
# FileViaCopy = Annotated[LinkedFile, as_linkedfile(symlink=False, recursive_symlink=False, overwrite=False)]

class SingleFileDict(TypedDict):
    """A TypedDict representing a single file."""

    parent_uid: str
    path: Path
    is_dir: Literal[False]

class SingleFileModel(BaseModel):
    """A Pydantic model representing a single file."""

    parent_uid: str
    path: Path
    is_dir: Literal[False] = False

def single_file_factory(parent_uid: str, path: Path) -> SingleFileDict:
    """Create a SingleFileDict."""
    return SingleFileDict(parent_uid=parent_uid, path=path, is_dir=False)

class DirectoryDict(TypedDict):
    """A TypedDict representing a directory."""

    parent_uid: str
    path: Path
    is_dir: Literal[True]

class DirectoryModel(BaseModel):
    """A Pydantic model representing a directory."""

    parent_uid: str
    path: Path
    is_dir: Literal[True] = True

def directory_factory(parent_uid: str, path: Path) -> DirectoryDict:
    """Create a DirectoryDict."""
    return DirectoryDict(parent_uid=parent_uid, path=path, is_dir=True)
