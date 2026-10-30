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
    selection = json.loads(Path(path).read_text(encoding='utf-8'))
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
    actual = [canonical(item, repo) for item in items]
    if len(actual) != len(set(actual)):
        raise pytest.UsageError('Duplicate full-nodeid in collection')
    missing = sorted(allowed-set(actual)) if allowed is not None else []
    if missing and os.environ.get('RELATION_ALLOW_MISSING_SELECTION') != '1':
        raise pytest.UsageError('Assigned full-nodeid missing from collection')
    config.hook.pytest_deselected(items=excluded+unassigned)
    output = Path(config.getoption('--api-relations-out'))
    (output / 'resume-selection.json').write_text(json.dumps({
        'source': path, 'job': selection['job'],
        'deselected': [canonical(item, repo) for item in excluded],
        'selected': [canonical(item, repo) for item in items],
        'unassigned': [canonical(item, repo) for item in unassigned],
        'shard_id': selection.get('shard_id'),
        'missing': missing,
    }, indent=2), encoding='utf-8')


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    path = os.environ.get('RELATION_RESUME_SELECTION')
    if not path:
        return
    selection = json.loads(Path(path).read_text(encoding='utf-8'))
    if 'allowed' not in selection:
        return
    repo = Path(session.config.getoption('--api-relations-repo')).resolve()
    actual = [canonical(item, repo) for item in session.items]
    if os.environ.get('RELATION_ALLOW_MISSING_SELECTION') == '1':
        output = Path(session.config.getoption('--api-relations-out'))
        receipt = json.loads((output/'resume-selection.json').read_text(encoding='utf-8'))
        if actual != receipt['selected'] or set(actual) | set(receipt['missing']) != set(selection['allowed']):
            raise pytest.UsageError('Final collection changed after exact selection')
        return
    if len(actual) != len(selection['allowed']) or set(actual) != set(selection['allowed']):
        raise pytest.UsageError('Final collected identities differ from assigned shard')
