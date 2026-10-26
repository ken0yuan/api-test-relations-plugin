"""Correctness boundaries for native rejection and dependency validation."""
import cProfile
import sys
from types import FunctionType, ModuleType, SimpleNamespace
import pytest
from test_relation_collector import collecting, catalog


@pytest.mark.parametrize('factory',[ModuleType,lambda name:SimpleNamespace()])
def test_same_code_rebinding_is_checked_before_native_rejection(factory):
    def public():pass
    foreign=FunctionType(public.__code__,public.__globals__,'foreign')
    root=factory('torch');root.public=public
    with collecting(root,catalog(('public','function'))) as (c,hits,gaps):
        token=c.enter(('rebind[param]','call'))
        foreign()
        root.public=foreign
        foreign();public()
        del root.public
        foreign()  # Previously bound callable retains its historical identity.
        c.leave(token)
    assert hits==[(('rebind[param]','call'),('python:torch.public',))]*3
    assert gaps==[]


def test_unrelated_dict_and_module_changes_do_not_rebuild_ledger():
    def public():pass
    root=ModuleType('torch');root.public=public
    with collecting(root,catalog(('public','function'))) as (c,hits,gaps):
        refresh=c.registry._refresh;rebuilds=[]
        def counted(*args,**kwargs):
            rebuilds.append(True)
            return refresh(*args,**kwargs)
        c.registry._refresh=counted
        token=c.enter(('changes','call'))
        for i in range(8):
            root.unrelated=i
            sys.modules['_relation_unrelated_probe']=ModuleType('_relation_unrelated_probe')
            public()
        del sys.modules['_relation_unrelated_probe']
        c.leave(token)
    assert len(hits)==8 and not gaps and not rebuilds


def test_late_namespace_and_replaced_parent_are_resolved_mid_phase():
    def one():pass
    def two():pass
    root=ModuleType('torch')
    with collecting(root,catalog(('sub.public','function'))) as (c,hits,gaps):
        token=c.enter(('nested','call'))
        root.sub=ModuleType('sub');root.sub.public=one;one()
        root.sub=SimpleNamespace(public=two);two();one()
        c.leave(token)
    assert hits==[(('nested','call'),('python:torch.sub.public',))]*3
    assert not gaps


def test_inherited_binding_and_mro_replacement_are_tracked():
    class A:
        def public(self):pass
    class B:
        def public(self):pass
    class Child(A):pass
    root=ModuleType('torch');root.Child=Child
    with collecting(root,catalog(('Child.public','method'))) as (c,hits,gaps):
        token=c.enter(('mro','call'))
        obj=Child();obj.public()
        Child.__bases__=(B,);obj.public()
        def replacement(self):pass
        B.public=replacement;obj.public()
        c.leave(token)
    assert hits==[(('mro','call'),('python:torch.Child.public',))]*3
    assert not gaps


def test_native_filter_does_not_filter_client_profiler_events():
    def public():pass
    def foreign():pass
    profiler=cProfile.Profile()
    with collecting(SimpleNamespace(public=public),catalog(('public','function'))) as (c,hits,gaps):
        token=c.enter(('profile','call'))
        profiler.enable();foreign();public();profiler.disable()
        c.leave(token)
    assert hits==[(('profile','call'),('python:torch.public',))] and not gaps
    assert sum(e.callcount for e in profiler.getstats() if e.code is foreign.__code__)==1


def test_compiler_import_does_not_depend_on_sys_modules_length():
    root=ModuleType('torch')
    with collecting(root,catalog(('late','function'))) as (c,hits,gaps):
        token=c.enter(('late','call'))
        def late():pass
        # API binding changes without changing sys.modules; exact function wins.
        root.late=late;late();c.leave(token)
    assert hits==[(('late','call'),('python:torch.late',))] and not gaps


def test_same_value_new_base_dependency_then_rebind():
    class A:
        def public(self):pass
    class B:public=A.public
    class Child(A):pass
    root=ModuleType('torch');root.Child=Child
    with collecting(root,catalog(('Child.public','method'))) as (c,hits,gaps):
        token=c.enter(('topology','call'));obj=Child()
        Child.__bases__=(B,);obj.public()
        def replacement(self):pass
        B.public=replacement;obj.public();c.leave(token)
    assert hits==[(('topology','call'),('python:torch.Child.public',))]*2 and not gaps


def test_instance_class_change_without_dict_mutation():
    class A:
        def public(self):pass
    class B:
        def public(self):pass
    instance=A();root=ModuleType('torch');root.obj=instance
    with collecting(root,catalog(('obj.public','method'))) as (c,hits,gaps):
        token=c.enter(('class-change','call'))
        instance.__class__=B;instance.public();c.leave(token)
    assert hits==[(('class-change','call'),('python:torch.obj.public',))] and not gaps


def test_instance_dict_replacement():
    def one():pass
    def two():pass
    root=SimpleNamespace(public=one)
    # A normal user object permits whole __dict__ replacement.
    class Root:pass
    root=Root();root.public=one
    with collecting(root,catalog(('public','function'))) as (c,hits,gaps):
        token=c.enter(('dict-replace','call'))
        root.__dict__={'public':two};two();c.leave(token)
    assert hits==[(('dict-replace','call'),('python:torch.public',))] and not gaps
