"""Disjoint full-nodeid scheduling policy; callable observation is untouched."""
import hashlib
import json
import sqlite3
from pathlib import Path


def positive_workers(value):
    number = int(value)
    if number < 1:
        raise ValueError('max-workers must be positive')
    return number


def partition_nodes(module, nodes, max_workers):
    limit = positive_workers(max_workers)
    nodes = list(nodes)
    if len(set(nodes)) != len(nodes):
        raise ValueError('Duplicate instance identity')
    # Spread adjacent op/dtype families over the partitions deterministically.
    ordered = sorted(nodes, key=lambda n: (hashlib.sha256(n.encode()).digest(), n))
    count = min(limit, len(ordered))
    return [{'shard_id': hashlib.sha256(module.encode()).hexdigest()[:16]+'-'+str(i),
             'module': module, 'nodeids': sorted(ordered[i::count])}
            for i in range(count)]


def validate_plan(plan, rows, exclusion):
    seen = set(); ids = set()
    for shard in plan['shards']:
        if shard['shard_id'] in ids or not shard['nodeids']:
            raise ValueError('Duplicate or empty shard')
        ids.add(shard['shard_id'])
        row = rows[shard['module']]
        if exclusion(row):
            raise ValueError('Excluded backend in instance plan')
        for node in shard['nodeids']:
            identity = (shard['module'], node)
            if identity in seen or not node.startswith(row['file']+'::'):
                raise ValueError('Overlapping or foreign instance')
            seen.add(identity)
    expected = {(module, node) for module, nodes in plan['remaining'].items() for node in nodes}
    if seen != expected:
        raise ValueError('Partition does not exactly cover remaining instances')
    return seen


def remaining_nodes(db, module, allowed):
    states = {}
    for start in range(0, len(allowed), 500):
        part = allowed[start:start+500]
        states.update(db.execute('SELECT t.nodeid,t.capture_state FROM tests t JOIN modules m '
                                 'ON m.id=t.module_id WHERE m.module=? AND t.nodeid IN ('+
                                 ','.join('?' for _ in part)+')', (module, *part)))
    if not set(allowed) <= states.keys():
        raise ValueError('Plan contains an unregistered instance')
    return sorted(node for node in allowed if states[node] != 'complete')


def critical_observer_gaps(db):
    return db.execute("SELECT reason FROM gaps WHERE reason LIKE 'collector_exception:%' "
        "OR reason LIKE 'bootstrap_error%' OR reason LIKE '%hook_lost%' "
        "OR reason LIKE '%bypassed%' OR reason LIKE 'checkpoint_write_failed%' "
        "OR reason LIKE 'collection_unavailable:%' OR reason LIKE 'subproc_pool_adapter_unsupported%'").fetchall()


def validate_capture(worker):
    output = Path(worker['output'])
    selection = json.loads((output/'resume-selection.json').read_text(encoding='utf-8'))
    expected = set(worker['allowed_nodeids'])
    missing = selection.get('missing', []) if worker.get('allow_missing_selection') else []
    selected = selection['selected']
    if (set(selected) | set(missing) != expected or set(selected) & set(missing)
            or len(selected)+len(missing) != len(expected)):
        raise ValueError('Actual collection differs from the exact shard selection')
    if expected & set(selection['deselected']):
        raise ValueError('Completed selection overlap')
    for path in Path(worker['shards']).glob('*.sqlite'):
        with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True, timeout=3) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in ('tests', 'phases', 'relations'):
                if table in tables:
                    if any(node not in expected for node, in db.execute('SELECT DISTINCT nodeid FROM '+table)):
                        raise ValueError('Raw owner escaped assigned instance shard')
            if 'gaps' in tables:
                critical = critical_observer_gaps(db)
                if critical:
                    raise ValueError('Observer failure in shard: '+repr(critical[:3]))
    for section in ('launches', 'receipts', 'tasks', 'task-receipts'):
        for path in (Path(worker['shards'])/section).glob('*.json'):
            item = json.loads(path.read_text())
            if item.get('nodeid') not in expected:
                raise ValueError('Process/task owner escaped assigned instance shard')
    return selection


def retry_state(before_remaining, after_remaining, resource_stop=False):
    before, after = set(before_remaining), set(after_remaining)
    if not after <= before:
        raise ValueError('Completed shard identity regressed')
    if not after:
        return 'captured'
    if resource_stop or len(after) < len(before):
        return 'pending'
    return 'continuation_needs_diagnosis'


def dispatch_allowed(resource, active_count, max_workers, capacity_limit, elapsed, estimated_start_mib=0):
    return (active_count < min(positive_workers(max_workers), capacity_limit)
            and elapsed >= 30 and resource['memory_available_mib'] >= max(3072, estimated_start_mib+2048)
            and resource['gpu_free_mib'] is not None and resource['gpu_free_mib'] >= 3072
            and min(resource['host_c_free_gib'], resource['database_volume_free_gib']) >= 10)
