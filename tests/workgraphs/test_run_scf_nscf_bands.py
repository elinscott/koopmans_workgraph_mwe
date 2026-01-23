from typing import Any

from aiida import load_profile
from ase.build import bulk

from koopmans_workgraph_mwe.engines.localhost import LocalhostEngine
from koopmans_workgraph_mwe.workgraphs.scf_nscf_bands import (
    PwWorkflowInputs,
    run_scf_nscf_bands,
    run_scf_nscf_bands_with_aiida,
)
from koopmans_workgraph_mwe.commands import CommandsConfig


def test_run_scf_nscf_bands(run_within_tmpdir: Any) -> None:
    """Test the SCF + NSCF + Bands workflow using the vanilla localhost engine."""
    atoms = bulk('Si')
    bandpath = atoms.cell.bandpath()

    engine = LocalhostEngine(commands={"pw": "mpirun -n 8 /home/linsco_e/code/q-e/build/bin/pw.x"})

    outputs = run_scf_nscf_bands(
        engine=engine,
        atoms=atoms,
        kpoints={'grid': [2, 2, 2], 'path': bandpath, 'gamma_only': False},
        pseudopotential_family='PseudoDojo/0.5/PBE/SR/standard/upf',
        pw_parameters={'system': {'ecutwfc': 20}},
    )

    assert outputs.total_energy < 0.0
    assert outputs.band_structure is not None

def test_run_scf_nscf_bands_with_aiida(run_within_tmpdir: Any) -> None:
    """Test the SCF + NSCF + Bands workflow using the AiiDA engine."""
    atoms = bulk('Si')
    bandpath = atoms.cell.bandpath()

    # Exterior pydantic validation for the moment
    input_model = PwWorkflowInputs(
        atoms = atoms,
        kpoints = {'grid': [2, 2, 2], 'path': bandpath, 'gamma_only': False},
        pseudopotential_family = 'PseudoDojo/0.5/PBE/SR/standard/upf',
        pw_parameters = {'system': {'ecutwfc': 20}},
    )

    load_profile()
    wg = run_scf_nscf_bands_with_aiida.build(**input_model.model_dump())

    wg.to_html('test.html')

    wg.run()

    assert wg.outputs.total_energy.value < 0.0
    assert wg.outputs.band_structure.value is not None


