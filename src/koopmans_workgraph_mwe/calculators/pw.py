"""pw calculator module for koopmans."""

from typing import Annotated, ClassVar, Any

import numpy as np
import numpy.typing as npt
from ase.dft.kpoints import BandPath
from ase.spectrum.band_structure import BandStructure
from pydantic import BeforeValidator, field_validator, model_validator, Field

from koopmans_workgraph_mwe.parameters.pw import PwInputParameters
from koopmans_workgraph_mwe.files import File, FileViaRecursiveSymlink
from koopmans_workgraph_mwe.pydantic_config import FlexibleStr
from koopmans_workgraph_mwe.serialization import deserialize_ase_bandstructure, deserialize_numpy

from .calculator import CalculatorInputs, CalculatorOutputs


class BasePwInputs(CalculatorInputs):
    """Generic inputs for calculations with pw.x."""

    parameters: PwInputParameters = {} # Field(default_factory=lambda: PwInputParameters())
    pseudopotential_family: FlexibleStr
    outdir: FileViaRecursiveSymlink | None = None
    calculation: ClassVar[str]

    @model_validator(mode='after')
    def check_outdir_provided_if_required(self):
        """Check if outdir has been provided if using `restart_mode = "restart"`."""
        if self.parameters.control.restart_mode == 'restart' and self.outdir is None:
            raise ValueError('Please provide an `outdir` when using `restart_mode = "restart"`')
        return self

    @field_validator('parameters', mode='before')
    @classmethod
    def set_calculation(cls, value: PwInputParameters) -> PwInputParameters:
        """Ensure that calculation type matches `self.calculation`."""
        value.control.calculation = cls.calculation
        return value
    
    @model_validator(mode='before')
    @classmethod
    def force_outdir_to_tmp(cls, data: dict[str, Any]) -> dict[str, Any]:
        data['parameters'].control.outdir = 'tmp'
        return data


class BasePwOutputs(CalculatorOutputs):
    eigenvalues: Annotated[npt.NDArray[np.float64], BeforeValidator(deserialize_numpy)] | None = None
    fermi_level: list[float]
    outdir: File

class PwScfInputs(BasePwInputs):
    kpoints: Annotated[npt.NDArray[np.float64], BeforeValidator(deserialize_numpy)]
    calculation: ClassVar[str] = 'scf'

class PwScfOutputs(BasePwOutputs):
    total_energy: float

class PwNscfInputs(BasePwInputs):
    kpoints: Annotated[npt.NDArray[np.float64], BeforeValidator(deserialize_numpy)]
    calculation: ClassVar[str] = 'nscf'

    @field_validator('outdir', mode='after')
    def ensure_outdir_provided(cls, value: str | None) -> str:
        """Ensure that outdir is provided for nscf calculations."""
        if value is None:
            raise ValueError('Please provide an `outdir` for nscf calculations')
        return value

class PwNscfOutputs(BasePwOutputs):
    pass

class PwBandsInputs(BasePwInputs):
    kpoints: BandPath
    calculation: ClassVar[str] = 'bands'

    @field_validator('outdir', mode='after')
    def ensure_outdir_provided(cls, value: str | None) -> str:
        """Ensure that outdir is provided for nscf calculations."""
        if value is None:
            raise ValueError('Please provide an `outdir` for nscf calculations')
        return value

class PwBandsOutputs(BasePwOutputs):
    band_structure: Annotated[BandStructure, BeforeValidator(deserialize_ase_bandstructure)] | None = None
    vbm: list[float] | None = None
    cbm: list[float] | None = None
