from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any
from pydantic import ValidationError
from pathlib import Path

from koopmans_workgraph_mwe.commands import CommandsConfig
from koopmans_workgraph_mwe.files import DirectoryDict, SingleFileDict, DirectoryModel, SingleFileModel, directory_factory, single_file_factory
from koopmans_workgraph_mwe.pydantic_config import BaseModel


class EngineABC(BaseModel, ABC):

    uids: list[str] = []
    commands: CommandsConfig

    def _pre_run(
        self,
        name: str,
        inputs: dict[str, Any],
        inputs_model: type[BaseModel] | None = None,
    ) -> str:
        # Assign this calculation a unique identifier
        uid = self.assign_uid(name)

        if inputs_model is not None:
            # Validate and dump inputs
            model = inputs_model.model_validate(inputs)
            self._dump(model, single_file_factory(parent_uid=uid, path=Path('inputs.json')))

        # Create working directory
        # For the moment, always run from scratch
        working_dir = directory_factory(parent_uid=uid, path=Path())
        if self.file_exists(working_dir):
            self.delete_file(working_dir)
        self.mkdir(working_dir, parents=True, exist_ok=True)

        # Copy over any inputs that correspond to files
        for k, inp in inputs.items():
            try:
                model = DirectoryModel.model_validate(inp)
                factory = directory_factory
            except ValidationError:
                try:
                    model = SingleFileModel.model_validate(inp)
                    factory = single_file_factory
                except ValidationError:
                    continue
            
            dest = factory(parent_uid=uid, path=model.path)
            self.link_file(inp, dest, recursive=model.is_dir, overwrite=True)

        return uid

    def _post_run(
        self,
        uid: str,
        outputs: dict[str, Any],
        outputs_model: type[BaseModel] | None,
    ) -> None:
        if outputs_model is not None:
            model = outputs_model.model_validate(outputs)
            self._dump(model, single_file_factory(parent_uid=uid, path=Path('outputs.json')))

    def task(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        input_model: type[BaseModel] | None = None,
        output_model: type[BaseModel] | None = None,
    ) -> Callable[..., Any]:
        task_name = name if name is not None else func.__name__
        def run_task(**kwargs: Any) -> dict[str, Any]:
            uid = self._pre_run(task_name, kwargs, input_model)
            # Provide uid and commands if needed
            if 'uid' in func.__code__.co_varnames:
                kwargs['uid'] = uid
            if 'commands' in func.__code__.co_varnames:
                kwargs['commands'] = self.commands
            outputs: dict[str, Any] = func(**kwargs)
            self._post_run(uid, outputs, output_model)
            return outputs
        return run_task

    @abstractmethod
    def file_exists(self, path: SingleFileDict | DirectoryDict) -> bool:
        """Check if a file exists at the given path."""
        ...

    @abstractmethod
    def delete_file(self, path: SingleFileDict | DirectoryDict) -> None:
        """Delete a file at the given path."""
        ...

    def copy_file(
        self,
        src: SingleFileDict | DirectoryDict,
        dest: SingleFileDict | DirectoryDict,
        overwrite: bool = False,
    ) -> None:
        """Copy a file from src to dest."""
        if overwrite and self.file_exists(dest):
            self.delete_file(dest)
        self._copy_file(src, dest)

    @abstractmethod
    def _copy_file(
        self, src: SingleFileDict | DirectoryDict, dest: SingleFileDict | DirectoryDict
    ) -> None:
        """Copy a file from src to dest."""
        ...

    def link_file(
        self,
        src: SingleFileDict | DirectoryDict,
        dest: SingleFileDict | DirectoryDict,
        recursive: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Link a file from src to dest."""
        if overwrite and self.file_exists(dest):
            self.delete_file(dest)
        self._link_file(src, dest, recursive)

    @abstractmethod
    def _link_file(
        self,
        src: SingleFileDict | DirectoryDict,
        dest: SingleFileDict | DirectoryDict,
        recursive: bool = False,
    ) -> None:
        """Link a file from src to dest."""
        ...

    @abstractmethod
    def is_dir(self, path: SingleFileDict | DirectoryDict) -> bool:
        """Check if the given path is a directory."""
        ...

    @abstractmethod
    def mkdir(self, path: DirectoryDict, parents: bool = False, exist_ok: bool = False) -> None:
        """Create a directory at the given path."""
        ...

    def assign_uid(self, name: str) -> str:
        """Get a unique identifier for the given directory."""
        uid = f"{len(self.uids) + 1:02d}-{name}"
        self.uids.append(uid)
        return uid

    def _dump(self, model: BaseModel, path: SingleFileDict) -> None:
        """Dump a pydantic model to a file."""
        self.write_to_file(model.model_dump_json(indent=2, exclude_none=True), path)

    @abstractmethod
    def write_to_file(self, content: str, path: SingleFileDict) -> None:
        ...
