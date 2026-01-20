import pytest
from koopmans_workgraph_mwe.utils import chdir

@pytest.fixture
def long_tmp_path(tmp_path_factory, request):
    test_name = request.node.name
    safe_name = test_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    return tmp_path_factory.mktemp(safe_name)

@pytest.fixture
def run_within_tmpdir(long_tmp_path):
    with chdir(long_tmp_path):
        yield
