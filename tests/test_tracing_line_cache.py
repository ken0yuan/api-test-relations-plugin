"""Generated traceback metadata must not overrun CPython's tracing cache."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import _relation_identity
import relation_collector


def isolated(source):
    env = dict(os.environ, PYTHONMALLOC='debug', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join((str(Path(_relation_identity.__file__).parent),
                                          str(Path(relation_collector.__file__).parent))))
    run = subprocess.run([sys.executable, '-c', textwrap.dedent(source)], env=env,
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads(run.stdout)


def test_oversized_traceback_table_preserves_mapping_and_saved_replace_identity():
    result = isolated('''
        import dis, gc, json, sys, types
        from types import SimpleNamespace
        from relation_collector import Collector, Registry
        offsets=[]
        def api():
            offsets.append(sys._getframe(1).f_lasti)
            return 17
        catalog={'apis':[{'api_id':'python:torch.'+n,'qualified_name':'torch.'+n,
                           'language':'python','kind':'function'} for n in ('api','alias')]}
        hits=[]; gaps=[]
        collector=Collector(Registry(catalog,SimpleNamespace(api=api,alias=api)),
                            lambda *row:hits.append(row),lambda *row:gaps.append(row))
        ns={}; exec(compile('def original():\\n'+'    x=1\\n'*150+'    return None\\n',
                            'long_source.py','exec'),ns)
        original=ns['original'].__code__
        short=compile('api()','generated_traceback.py','exec')
        descriptor=vars(types.CodeType)['replace']; saved=short.replace
        collector.install(); token=collector.enter(('test::owner','call'))
        try:
            generated=saved(co_linetable=original.co_linetable,
                            co_firstlineno=original.co_firstlineno)
            assert vars(types.CodeType)['replace'] is descriptor and short.replace==saved
            before=(generated.co_code,generated.co_linetable,generated.co_firstlineno,
                    generated.co_name,generated.co_filename,generated.co_exceptiontable)
            for _ in range(20): exec(generated,{'api':api})
            assert len(set(offsets))==1
            expected=[line for start,end,line in generated.co_lines()
                      if start<=offsets[0]<end][0]
            assert before==(generated.co_code,generated.co_linetable,generated.co_firstlineno,
                            generated.co_name,generated.co_filename,generated.co_exceptiontable)
            assert hits==[(('test::owner','call'),('python:torch.alias','python:torch.api'),
                           ('generated_traceback.py',expected))]*20, hits
            assert not gaps,gaps
        finally: collector.leave(token);collector.close()
        gc.collect()
        print(json.dumps({'relations':len(hits),'metadata_unchanged':True,'gaps':gaps}))
    ''')
    assert result == {'relations':20, 'metadata_unchanged':True, 'gaps':[]}


def test_native_replace_error_and_ordinary_trace_events_are_preserved():
    result = isolated('''
        import json,sys,types
        from types import SimpleNamespace
        from relation_collector import Collector,Registry
        def workload(x):
            y=x+1
            return y
        def record(target):
            rows=[]
            def trace(frame,event,arg):
                if frame.f_code is target:rows.append((event,frame.f_lineno))
                return trace
            return rows,trace
        old_rows,old_trace=record(workload.__code__)
        sys.settrace(old_trace);assert workload(3)==4;sys.settrace(None)
        collector=Collector(Registry({'apis':[]},SimpleNamespace()),lambda *a:None,lambda *a:None)
        descriptor=vars(types.CodeType)['replace']; saved=workload.__code__.replace
        try:saved(co_firstlineno='bad')
        except Exception as e:expected=(type(e).__name__,str(e))
        collector.install()
        try:
            new_rows,new_trace=record(workload.__code__)
            sys.settrace(new_trace);assert workload(3)==4;sys.settrace(None)
            try:saved(co_firstlineno='bad')
            except Exception as e:actual=(type(e).__name__,str(e))
            assert actual==expected
            assert descriptor is vars(types.CodeType)['replace']
            assert new_rows==old_rows,(new_rows,old_rows)
        finally:collector.close()
        print(json.dumps({'events':new_rows,'error':actual[0]}))
    ''')
    assert result['error'] == 'TypeError'
    assert [r[0] for r in result['events']] == ['call','line','line','return']
