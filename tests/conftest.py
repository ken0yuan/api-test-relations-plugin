import os
from contextlib import contextmanager
from pathlib import Path
import shutil
import sys
import tempfile
import signal
import time
import traceback
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from api_test_relations.cli import PACKAGE, build_identity

# Compile for the test host in a disposable folder, never in the distributed source.
_scratch = tempfile.TemporaryDirectory(prefix='relation-plugin-regression-')
_runtime = Path(_scratch.name)
shutil.copyfile(PACKAGE / 'runtime/relation_identity.c', _runtime / 'relation_identity.c')
shutil.copyfile(PACKAGE / 'runtime/relation_guard.h', _runtime / 'relation_guard.h')
build_identity(_runtime)
sys.path[:0] = [str(_runtime), str(PACKAGE / 'runtime')]


def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker('paired_plugins') and os.environ.get('API_RELATIONS_CALLCHAIN_TESTS')!='1':
            item.add_marker(pytest.mark.skip(reason='Optional paired-plugin regressions are disabled by default'))
        if item.get_closest_marker('integration') and os.environ.get('API_RELATIONS_REAL_TORCH_TESTS')!='1':
            item.add_marker(pytest.mark.skip(reason='Real torch/CUDA integration is explicit opt-in'))


@pytest.fixture
def zombie_process():
    """Keep a real exited child unreaped until the caller checks its evidence."""
    if not hasattr(os, 'fork') or not Path('/proc/self/stat').exists():
        pytest.skip('Requires Linux fork and procfs')

    @contextmanager
    def spawn(action):
        pid = os.fork()
        if pid == 0:
            try:
                action(os.getpid())
            except BaseException:
                traceback.print_exc()
                os._exit(1)
            os._exit(0)
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                stat = Path('/proc', str(pid), 'stat').read_text(encoding='utf-8')
                if stat.rpartition(')')[2].split()[0] == 'Z':
                    break
                time.sleep(.01)
            else:
                pytest.fail('Child did not reach zombie state')
            yield pid
        finally:
            waited, status = os.waitpid(pid, os.WNOHANG)
            if not waited:
                os.kill(pid, signal.SIGKILL)
                _, status = os.waitpid(pid, 0)
            assert os.waitstatus_to_exitcode(status) == 0, 'Fixture child failed'

    return spawn
