"""Exact mutation epochs must not weaken binding or structure invalidation."""
from types import ModuleType
import ctypes
import sys
from _relation_identity import make_filter, filter_dirty, namespace_version, binding_filter_stats
from test_relation_collector import collecting, catalog
from relation_collector import Registry


def snapshot(*objects):
    return [make_filter({}, {}, {}, objects, tuple(namespace_version(x) for x in objects))]


def test_clean_epoch_avoids_repeated_dictionary_scans_without_time_sampling():
    objects=tuple(ModuleType('namespace_'+str(i)) for i in range(256))
    state=snapshot(*objects)
    assert not filter_dirty(state)
    before=binding_filter_stats(state)
    for _ in range(2000):
        assert not filter_dirty(state)
    after=binding_filter_stats(state)
    assert after['dictionary_scans']-before['dictionary_scans']<10
    assert after['unchanged_epochs']-before['unchanged_epochs']>1990


def test_changed_binding_remains_dirty_until_new_snapshot():
    root=ModuleType('torch');root.public=object();state=snapshot(root)
    assert not filter_dirty(state)
    root.public=object()
    for _ in range(20):
        assert filter_dirty(state)
    assert not filter_dirty(snapshot(root))


def test_unrelated_dictionary_write_does_not_hide_next_namespace_write():
    root=ModuleType('torch');root.public=object();state=snapshot(root);unrelated={}
    for i in range(10):
        unrelated[i]=i
        assert not filter_dirty(state)
    root.public=object()
    assert filter_dirty(state)


def test_mro_change_checked_even_without_target_dict_change():
    class A:pass
    class B:pass
    class Child(A):pass
    state=snapshot(Child)
    assert not filter_dirty(state)
    Child.__bases__=(B,)
    assert filter_dirty(state)


def test_instance_class_change_checked_even_without_instance_dict():
    class A:__slots__=()
    class B:__slots__=()
    instance=A();state=snapshot(instance)
    assert not filter_dirty(state)
    instance.__class__=B
    assert filter_dirty(state)


def test_existing_instance_dictionary_replacement_checked():
    class Root:pass
    root=Root();root.public=object();other={'public':object()}
    state=snapshot(root)
    assert not filter_dirty(state)
    root.__dict__=other
    assert filter_dirty(state)


def test_rebinding_aliases_and_saved_callable_keep_exact_ownership():
    def one():pass
    def two():pass
    root=ModuleType('torch');root.public=one;root.alias=one
    with collecting(root,catalog(('public','function'),('alias','function'))) as (c,hits,gaps):
        first=c.enter(('first[param]','call'))
        one()
        root.public=two
        two();one()
        c.leave(first)
        second=c.enter(('second[param]','call'))
        del root.alias
        one();two()
        c.leave(second)
    both=('python:torch.alias','python:torch.public')
    assert hits==[(('first[param]','call'),both),
                  (('first[param]','call'),('python:torch.public',)),
                  (('first[param]','call'),both),
                  (('second[param]','call'),both),
                  (('second[param]','call'),('python:torch.public',))]
    assert not gaps


def test_private_trace_frame_updates_do_not_force_full_scans():
    def public():pass
    def foreign():pass
    root=ModuleType('torch');root.public=public
    with collecting(root,catalog(('public','function'))) as (c,hits,gaps):
        token=c.enter(('epoch-work','call'))
        before=binding_filter_stats(c.registry.fast_state)
        for _ in range(2000):foreign()
        after=binding_filter_stats(c.registry.fast_state)
        c.leave(token)
    assert after['dictionary_scans']-before['dictionary_scans']<20
    assert after['unchanged_epochs']-before['unchanged_epochs']>1990
    assert not hits and not gaps


def test_client_trace_mutation_is_not_accounted_as_private_frame_write():
    def public():pass
    def foreign():pass
    root=ModuleType('torch');root.public=public
    def client(frame,event,arg):
        if frame.f_code is foreign.__code__ and event=='call':root.public=foreign
        return client
    with collecting(root,catalog(('public','function'))) as (c,hits,gaps):
        token=c.enter(('client-rebind','call'))
        sys.settrace(client)
        foreign()
        sys.settrace(None)
        c.leave(token)
    assert hits==[(('client-rebind','call'),('python:torch.public',))]
    assert not gaps


def test_native_dictionary_api_mutation_is_not_skipped():
    def public():pass
    def foreign():pass
    root=ModuleType('torch');root.public=public
    setter=ctypes.pythonapi.PyDict_SetItem
    setter.argtypes=[ctypes.py_object,ctypes.py_object,ctypes.py_object]
    setter.restype=ctypes.c_int
    with collecting(root,catalog(('public','function'))) as (c,hits,gaps):
        token=c.enter(('native-rebind','call'))
        assert setter(root.__dict__,'public',foreign)==0
        foreign();c.leave(token)
    assert hits==[(('native-rebind','call'),('python:torch.public',))]
    assert not gaps


def test_refresh_reuses_only_exact_unchanged_prefixes(monkeypatch):
    import inspect
    root=ModuleType('torch');sub=ModuleType('sub');root.sub=sub
    for i in range(40):setattr(sub,'api'+str(i),lambda:None)
    original=inspect.getattr_static;prefix_reads=[]
    def counted(obj,name,*args):
        if obj is root and name=='sub':prefix_reads.append(True)
        return original(obj,name,*args)
    monkeypatch.setattr(inspect,'getattr_static',counted)
    Registry(catalog(*[('sub.api'+str(i),'function') for i in range(40)]),root)
    assert len(prefix_reads)==1


def test_changed_prefix_during_refresh_is_read_again(monkeypatch):
    import inspect
    def one():pass
    def two():pass
    root=ModuleType('torch');a=ModuleType('a');b=ModuleType('b')
    a.first=one;b.second=two;root.sub=a
    original=inspect.getattr_static;reads=[]
    def rebinding(obj,name,*args):
        value=original(obj,name,*args)
        if obj is root and name=='sub':
            reads.append(value)
            root.sub=b
        return value
    monkeypatch.setattr(inspect,'getattr_static',rebinding)
    registry=Registry(catalog(('sub.first','function'),('sub.second','function')),root)
    assert reads==[a,b]
    assert registry.python[one]=={'python:torch.sub.first'}
    assert registry.python[two]=={'python:torch.sub.second'}
