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
    allowed = set(selection['allowed']) if 'allowed' in selection else None
    if allowed is not None and (len(allowed) != len(selection['allowed']) or allowed & retained):
        raise pytest.UsageError('Invalid overlapping/duplicate instance selection')
    excluded = [item for item in items if canonical(item, repo) in retained]
    unassigned = [item for item in items if canonical(item, repo) not in retained
                  and allowed is not None and canonical(item, repo) not in allowed]
    items[:] = [item for item in items if canonical(item, repo) not in retained
                and (allowed is None or canonical(item, repo) in allowed)]
    if allowed is not None and {canonical(item, repo) for item in items} != allowed:
        raise pytest.UsageError('Assigned full-nodeid missing from collection')
    config.hook.pytest_deselected(items=excluded+unassigned)
    output = Path(config.getoption('--api-relations-out'))
    (output / 'resume-selection.json').write_text(json.dumps({
        'source': path, 'job': selection['job'],
        'deselected': [canonical(item, repo) for item in excluded],
        'selected': [canonical(item, repo) for item in items],
        'unassigned': [canonical(item, repo) for item in unassigned],
        'shard_id': selection.get('shard_id'),
    }, indent=2))


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    path = os.environ.get('RELATION_RESUME_SELECTION')
    if not path:
        return
    selection = json.loads(Path(path).read_text())
    if 'allowed' not in selection:
        return
    repo = Path(session.config.getoption('--api-relations-repo')).resolve()
    actual = [canonical(item, repo) for item in session.items]
    if len(actual) != len(selection['allowed']) or set(actual) != set(selection['allowed']):
        raise pytest.UsageError('Final collected identities differ from assigned shard')
