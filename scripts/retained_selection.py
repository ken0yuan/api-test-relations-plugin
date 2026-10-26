"""Execution policy only; keep completed observations out of new plugin captures."""
import json
import os
from pathlib import Path
import pytest


def canonical(item, repo):
    _, sep, suffix = item.nodeid.partition('::')
    return 'pytorch/' + Path(item.path).resolve().relative_to(repo).as_posix() + sep + suffix


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(session, config, items):
    path = os.environ.get('RELATION_RESUME_SELECTION')
    if not path:
        return
    selection = json.loads(Path(path).read_text())
    repo = Path(config.getoption('--api-relations-repo')).resolve()
    retained = set(selection['retained'])
    excluded = [item for item in items if canonical(item, repo) in retained]
    items[:] = [item for item in items if canonical(item, repo) not in retained]
    config.hook.pytest_deselected(items=excluded)
    output = Path(config.getoption('--api-relations-out'))
    (output / 'resume-selection.json').write_text(json.dumps({
        'source': path, 'job': selection['job'],
        'deselected': [canonical(item, repo) for item in excluded],
        'selected': [canonical(item, repo) for item in items],
    }, indent=2))
