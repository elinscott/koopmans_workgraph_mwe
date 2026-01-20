from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from koopmans_workgraph_mwe.pydantic_config import BaseModel
from koopmans_workgraph_mwe.status import Status
from koopmans_workgraph_mwe.calculators.pw import PwScfInputs, PwScfOutputs, PwNscfInputs, PwNscfOutputs, PwBandsInputs, PwBandsOutputs
from koopmans_workgraph_mwe.files import File, LocalFile

OutputModel = TypeVar("OutputModel")

class Response(BaseModel, Generic[OutputModel]):
    status: Status
    output: OutputModel | None = None
    error_message: str | None = None
    error_type: type | None = None

class Engine(BaseModel, ABC):

    uids: list[str] = []
    pw_command: str = "pw.x"

    def _pre_run(self, inputs: BaseModel, name: str) -> str:
        # Assign this calculation a unique identifier
        uid = self.assign_uid(name)

        # Create working directory
        # For the moment, always run from scratch
        working_dir = File(parent_process_uid=uid, path='.')
        if self.file_exists(working_dir):
            self.delete_file(working_dir)
        self.mkdir(working_dir, parents=True, exist_ok=True)

        # Dump inputs
        self._dump(inputs, File(parent_process_uid=uid, path='inputs.json'))

        # Copy over any inputs that correspond to files
        files_to_copy = inputs.fields_that_are_files()
        for inp in files_to_copy:
            dest = File(parent_process_uid=uid, path=inp.dest)
            if inp.symlink:
                self.link_file(inp.src, dest, overwrite=inp.overwrite)
            else:
                self.copy_file(inp.src, dest, overwrite=inp.overwrite)
        
        return uid
        
    def _post_run(self, outputs: BaseModel, uid: str) -> None:
        self._dump(outputs, File(parent_process_uid=uid, path='outputs.json'))

    def run_pw_scf(self, inputs: PwScfInputs) -> PwScfOutputs:
        uid = self._pre_run(inputs, 'quantumespresso-pw-scf')
        outputs = self._run_pw_scf(inputs, uid)
        self._post_run(outputs, uid)
        return outputs
    
    def run_pw_nscf(self, inputs: PwNscfInputs) -> PwNscfOutputs:
        uid = self._pre_run(inputs, 'quantumespresso-pw-nscf')
        outputs = self._run_pw_nscf(inputs, uid)
        self._post_run(outputs, uid)
        return outputs
    
    def run_pw_bands(self, inputs: PwBandsInputs) -> PwBandsOutputs:
        uid = self._pre_run(inputs, 'quantumespresso-pw-bands')
        outputs = self._run_pw_bands(inputs, uid)
        self._post_run(outputs, uid)
        return outputs

    @abstractmethod
    def _run_pw_scf(self, inputs: PwScfInputs, uid: str) -> PwScfOutputs:
        ...
    
    @abstractmethod
    def _run_pw_nscf(self, inputs: PwNscfInputs, uid: str) -> PwNscfOutputs:
        ... 
    
    @abstractmethod
    def _run_pw_bands(self, inputs: PwBandsInputs, uid: str) -> PwBandsOutputs:
        ...
    
    @abstractmethod
    def file_exists(self, path: File) -> bool:
        """Check if a file exists at the given path."""
        ...
    
    @abstractmethod
    def delete_file(self, path: File):
        """Delete a file at the given path."""
        ...
    
    def copy_file(self, src: File | LocalFile, dest: File, overwrite: bool = False):
        """Copy a file from src to dest."""
        if overwrite and self.file_exists(dest):
            self.delete_file(dest)
        self._copy_file(src, dest)
        ...

    @abstractmethod
    def _copy_file(self, src: File | LocalFile, dest: File):
        """Copy a file from src to dest."""
        ...

    def link_file(self, src: File | LocalFile, dest: File, recursive: bool = False, overwrite: bool = False):
        """Link a file from src to dest."""
        if overwrite and self.file_exists(dest):
            self.delete_file(dest)
        self._link_file(src, dest, recursive)

    @abstractmethod
    def _link_file(self, src: File | LocalFile, dest: File, recursive: bool = False):
        """Link a file from src to dest."""
        ...
    
    @abstractmethod
    def is_dir(self, path: File | LocalFile) -> bool:
        """Check if the given path is a directory."""
        ...
    
    @abstractmethod
    def mkdir(self, path: File, parents: bool = False, exist_ok: bool = False):
        """Create a directory at the given path."""
        ...

    def assign_uid(self, name: str) -> str:
        """Get a unique identifier for the given directory."""
        uid = f"{len(self.uids) + 1:02d}-{name}"
        self.uids.append(uid)
        return uid

    def _dump(self, model: BaseModel, path: File):
        """Dump a pydantic model to a file."""
        self.write_to_file(model.model_dump_json(indent=2, exclude_none=True), path)
    
    @abstractmethod
    def write_to_file(self, content: str, path: File):
        ...
    