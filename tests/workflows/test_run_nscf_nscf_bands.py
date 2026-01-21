import pytest

from ase.build import bulk

from koopmans_workgraph_mwe.engines.ase import AseEngine

def test_scf_nscf_bands_workflow(run_within_tmpdir):
    """Test the SCF + NSCF + Bands workflow."""
    atoms = bulk('Si')
    bandpath = atoms.cell.bandpath()

    engine=AseEngine(pw_command="mpirun -n 8 /home/linsco_e/code/q-e/build/bin/pw.x")

    outputs = engine.run_scf_nscf_bands(
        atoms = atoms,
        kpoints = {'grid':[2,2,2], 'path': bandpath, 'gamma_only': False},
        pseudopotential_family = 'PseudoDojo/0.5/PBE/SR/standard/upf',
        pw_parameters = {'system': {'ecutwfc': 20}}
    )

    assert outputs.total_energy < 0.0
    assert outputs.band_structure is not None
