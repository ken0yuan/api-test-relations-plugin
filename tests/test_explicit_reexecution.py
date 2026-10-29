import json
import sqlite3

import pytest

from api_test_relations import single_store
from test_retention import capture
from test_single_store import baseline,add_phases

NODE='pytorch/test/test_one.py::test_one[param]'


def prior(dbpath):
    with sqlite3.connect(dbpath) as db:
        single_store.upgrade_observation_schema(db)
        db.execute('INSERT INTO tests VALUES (2,1,?,?,?,?,?)',(NODE,'pytorch/test/test_one.py','complete','old-version','old-attempt'))
        db.execute("INSERT INTO phases VALUES (1,'setup')")
        db.execute('INSERT INTO alias_groups VALUES (1,?)',('["a"]',))
        db.execute('INSERT INTO observations VALUES (1,2,1,1)')
        db.execute('INSERT INTO observation_evidence VALUES (1,2,1,1,?,?,?)',('old-site','old-capture','old-shard'))


def test_explicit_reexecution_keeps_old_and_new_origins_and_positive_branches(tmp_path):
    target=baseline(tmp_path);prior(target);root=capture(tmp_path);add_phases(root);origin=str(root.resolve())
    result=single_store.append_capture(root,target,'one',allow_reexecution_of=[NODE])
    assert result['cleanup']=='complete'
    with sqlite3.connect(target) as db:
        assert single_store.observations(db,2)=={('a','setup','["a"]'),('a','call','["a"]')}
        assert db.execute('SELECT source_capture FROM observation_evidence WHERE site="old-site"').fetchone()==('old-capture',)
        assert {r[0] for r in db.execute('SELECT source_capture FROM observation_evidence')}=={'old-capture',origin}
        detail=json.loads(db.execute('SELECT detail FROM capture_runs WHERE attempt=?',(origin,)).fetchone()[0])
        assert len(detail['checks'])==6 and detail['authorized_reexecutions']==[NODE]
        assert detail['reexecution_prior'][NODE]['test_record'][-2:]==['old-version','old-attempt']
        assert NODE in single_store.completed(db,'one')
        assert not db.execute('PRAGMA foreign_key_check').fetchall()


def test_default_still_rejects_completed_test_and_preserves_evidence(tmp_path):
    target=baseline(tmp_path);prior(target);root=capture(tmp_path);add_phases(root)
    with pytest.raises(ValueError,match='Completed'):
        single_store.append_capture(root,target,'one')
    assert root.exists()


def test_authorization_for_another_node_does_not_authorize_this_test(tmp_path):
    target=baseline(tmp_path);prior(target);root=capture(tmp_path);add_phases(root)
    with pytest.raises(ValueError):
        single_store.append_capture(root,target,'one',allow_reexecution_of=[NODE+'-other'])
    assert root.exists()


def test_interrupted_reexecution_keeps_completed_metadata_and_raw(tmp_path):
    target=baseline(tmp_path);prior(target);root=capture(tmp_path);add_phases(root)
    with pytest.raises(ValueError,match='Incomplete reexecution'):
        single_store.append_capture(root,target,'one',interrupted=True,allow_reexecution_of=[NODE])
    assert root.exists()
    with sqlite3.connect(target) as db:
        assert db.execute('SELECT source_version,source_attempt,capture_state FROM tests WHERE id=2').fetchone()==('old-version','old-attempt','complete')
