import pytest

from ase.build import bulk

from koopmans_workgraph_mwe.workgraphs.scf_nscf_bands import aiida_run_scf_nscf_bands
from aiida import load_profile

def test_aiida_scf_nscf_bands(run_within_tmpdir):
    """Test the SCF + NSCF + Bands workflow."""
    atoms = bulk('Si')
    bandpath = atoms.cell.bandpath()

    load_profile()
    wg = aiida_run_scf_nscf_bands.build(
        atoms = atoms,
        kpoints = {'grid':[2,2,2], 'path': bandpath, 'gamma_only': False},
        pseudopotential_family = 'PseudoDojo/0.5/PBE/SR/standard/upf',
        pw_parameters = {'system': {'ecutwfc': 20}}
    )

    wg.run()

    assert wg.outputs.total_energy.value < 0.0
    assert wg.outputs.band_structure.value is not None

    wg.to_html('test.html')
