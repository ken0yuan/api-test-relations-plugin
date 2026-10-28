#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <internal/pycore_frame.h>
#include <descrobject.h>

#if PY_MAJOR_VERSION != 3 || PY_MINOR_VERSION != 11
#error "This identity adapter is for CPython 3.11; add a version adapter before using another interpreter."
#endif

static int filter_observation(PyObject *, PyFrameObject *, int, PyObject *);

/* Enter Python observer code with a thread-local compiler evaluation guard.
 * The setter is supplied through a runtime capability, not a source offset.
 * It must be a C callable so its entry cannot itself be compiled.
 */
static int relation_profile(PyObject *state, PyFrameObject *frame, int event, PyObject *arg) {
    static const char *names[] = {"call", "exception", "line", "return", "c_call", "c_exception", "c_return", "opcode"};
    if (event < 0 || event > PyTrace_OPCODE) return 0;
    if (PyTuple_GET_SIZE(state)>2) {
        int needed=filter_observation(PyTuple_GET_ITEM(state,2),frame,event,arg);
        if (needed<=0) return needed;
    }
    PyObject *callback=PyTuple_GET_ITEM(state,0), *guard=PyTuple_GET_ITEM(state,1), *previous=NULL;
    if (guard != Py_None) {
        previous=PyObject_CallOneArg(guard,Py_None);
        if (!previous) return -1;
    }
    PyObject *result=PyObject_CallFunction(callback,"OsO",(PyObject *)frame,names[event],arg ? arg : Py_None);
    PyObject *exc_type=NULL, *exc_value=NULL, *exc_tb=NULL;
    if (!result) PyErr_Fetch(&exc_type,&exc_value,&exc_tb);
    if (previous) {
        PyObject *restored=PyObject_CallOneArg(guard,previous);
        Py_DECREF(previous);
        if (!restored) {
            Py_XDECREF(result);Py_XDECREF(exc_type);Py_XDECREF(exc_value);Py_XDECREF(exc_tb);
            return -1;
        }
        Py_DECREF(restored);
    }
    if (!result) {PyErr_Restore(exc_type,exc_value,exc_tb);return -1;}
    Py_DECREF(result);return 0;
}

static PyObject *set_profile(PyObject *self, PyObject *args) {
    PyObject *callback,*guard;
    if (!PyArg_ParseTuple(args,"OO",&callback,&guard)) return NULL;
    if (!PyCallable_Check(callback) || (guard!=Py_None && !PyCFunction_Check(guard))) {
        PyErr_SetString(PyExc_TypeError,"Expected callback and a native evaluation guard");return NULL;
    }
    PyObject *state=PyTuple_Pack(2,callback,guard);
    if (!state) return NULL;
    PyEval_SetProfile(relation_profile,state);Py_DECREF(state);
    Py_RETURN_NONE;
}

static PyObject *current_profile(PyObject *self, PyObject *unused) {
    PyThreadState *state=PyThreadState_Get();
    if (!state->c_profileobj) Py_RETURN_NONE;
    if (state->c_profilefunc==relation_profile) return Py_NewRef(PyTuple_GET_ITEM(state->c_profileobj,0));
    return Py_NewRef(state->c_profileobj);
}

/* The actual executing function, NOT a guess based on shared bytecode. */
static PyObject *frame_function(PyObject *self, PyObject *arg) {
    if (!PyFrame_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "expected frame");
        return NULL;
    }
    PyFrameObject *frame = (PyFrameObject *)arg;
    if (!frame->f_frame || !frame->f_frame->f_func) Py_RETURN_NONE;
    return Py_NewRef((PyObject *)frame->f_frame->f_func);
}

static PyMethodDef *boundary_original_definition(PyMethodDef *);

/* Descriptor identity also survives storing a bound method before monkeypatching. */
static PyObject *method_definition(PyObject *self, PyObject *arg) {
    if (PyCFunction_Check(arg))
        return PyLong_FromVoidPtr(boundary_original_definition(((PyCFunctionObject *)arg)->m_ml));
    if (Py_IS_TYPE(arg, &PyMethodDescr_Type) || Py_IS_TYPE(arg, &PyClassMethodDescr_Type))
        return PyLong_FromVoidPtr(boundary_original_definition(((PyMethodDescrObject *)arg)->d_method));
    Py_RETURN_NONE;
}

static PyObject *namespace_dict(PyObject *arg) {
    PyObject *dict = NULL;
    if (PyModule_Check(arg)) dict = PyModule_GetDict(arg);
    else if (PyType_Check(arg)) dict = ((PyTypeObject *)arg)->tp_dict;
    else {PyObject **slot=_PyObject_GetDictPtr(arg); if (slot) dict=*slot;}
    return dict;
}

static PyObject *namespace_version(PyObject *self, PyObject *arg) {
    PyObject *dict=namespace_dict(arg);
    unsigned long long version=dict ? ((PyDictObject *)dict)->ma_version_tag : 0;
    PyObject *structure=PyType_Check(arg) ? ((PyTypeObject *)arg)->tp_mro : (PyObject *)Py_TYPE(arg);
    return Py_BuildValue("KO",version,structure ? structure : Py_None);
}

static int namespace_dirty(PyObject *object, PyObject *version) {
    PyObject *dict=namespace_dict(object);
    if (!PyTuple_Check(version) || PyTuple_GET_SIZE(version)!=2) return 1;
    PyObject *structure=PyType_Check(object) ? ((PyTypeObject *)object)->tp_mro : (PyObject *)Py_TYPE(object);
    if ((structure ? structure : Py_None)!=PyTuple_GET_ITEM(version,1)) return 1;
    return (dict ? ((PyDictObject *)dict)->ma_version_tag : 0)!=PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(version,0));
}

static PyObject *namespaces_changed(PyObject *self, PyObject *args) {
    PyObject *objects, *versions;
    if (!PyArg_ParseTuple(args, "O!O!", &PyTuple_Type, &objects, &PyTuple_Type, &versions)) return NULL;
    if (PyTuple_GET_SIZE(objects) != PyTuple_GET_SIZE(versions)) {
        PyErr_SetString(PyExc_ValueError, "namespace/version length mismatch");return NULL;
    }
    for (Py_ssize_t i=0;i<PyTuple_GET_SIZE(objects);i++) {
        if (namespace_dirty(PyTuple_GET_ITEM(objects,i),PyTuple_GET_ITEM(versions,i))) Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

typedef struct {
    PyObject *object,*dict,*mro;
    unsigned long long version;
    int kind;
} BindingVersion;
typedef struct {
    PyObject *python,*cfunc,*descriptor;
    Py_ssize_t count;
    BindingVersion *items;
} BindingFilter;

static void destroy_filter(PyObject *capsule) {
    BindingFilter *f=PyCapsule_GetPointer(capsule,"relation_binding_filter");
    if (!f) {PyErr_Clear();return;}
    Py_XDECREF(f->python);Py_XDECREF(f->cfunc);Py_XDECREF(f->descriptor);
    for (Py_ssize_t i=0;i<f->count;i++) {
        Py_XDECREF(f->items[i].object);Py_XDECREF(f->items[i].dict);Py_XDECREF(f->items[i].mro);
    }
    PyMem_Free(f->items);PyMem_Free(f);
}

static PyObject *make_filter(PyObject *self, PyObject *args) {
    PyObject *python,*cfunc,*descriptor,*objects,*versions;
    if (!PyArg_ParseTuple(args,"O!O!O!O!O!",&PyDict_Type,&python,&PyDict_Type,&cfunc,
                         &PyDict_Type,&descriptor,&PyTuple_Type,&objects,&PyTuple_Type,&versions)) return NULL;
    if (PyTuple_GET_SIZE(objects)!=PyTuple_GET_SIZE(versions)) {
        PyErr_SetString(PyExc_ValueError,"Mismatched binding snapshot");return NULL;
    }
    BindingFilter *f=PyMem_Calloc(1,sizeof(BindingFilter));
    if (!f) return PyErr_NoMemory();
    f->count=PyTuple_GET_SIZE(objects);
    f->items=PyMem_Calloc(f->count+1,sizeof(BindingVersion));
    if (!f->items) {PyMem_Free(f);return PyErr_NoMemory();}
    PyObject *capsule=PyCapsule_New(f,"relation_binding_filter",destroy_filter);
    if (!capsule) {PyMem_Free(f->items);PyMem_Free(f);return NULL;}
    f->python=Py_NewRef(python);f->cfunc=Py_NewRef(cfunc);f->descriptor=Py_NewRef(descriptor);
    for (Py_ssize_t i=0;i<f->count;i++) {
        BindingVersion *v=&f->items[i];PyObject *version=PyTuple_GET_ITEM(versions,i);
        v->object=Py_NewRef(PyTuple_GET_ITEM(objects,i));
        v->dict=Py_XNewRef(namespace_dict(v->object));
        v->kind=PyType_Check(v->object) ? 1 : (PyModule_Check(v->object) ? 0 : 2);
        v->mro=Py_NewRef(PyTuple_GET_ITEM(version,1));version=PyTuple_GET_ITEM(version,0);
        v->version=PyLong_AsUnsignedLongLong(version);
        if (PyErr_Occurred()) {Py_DECREF(capsule);return NULL;}
    }
    return capsule;
}

static BindingFilter *get_filter(PyObject *state) {
    if (!PyList_Check(state) || PyList_GET_SIZE(state)!=1) return NULL;
    PyObject *capsule=PyList_GET_ITEM(state,0);
    if (!PyCapsule_IsValid(capsule,"relation_binding_filter")) return NULL;
    return PyCapsule_GetPointer(capsule,"relation_binding_filter");
}

static int binding_filter_dirty(BindingFilter *f) {
    if (!f) return 1;
    for (Py_ssize_t i=0;i<f->count;i++) {
        BindingVersion *v=&f->items[i];
        if (v->kind==2 && namespace_dict(v->object)!=v->dict) return 1;
        if (v->dict && ((PyDictObject *)v->dict)->ma_version_tag!=v->version) return 1;
        if (v->kind==1 && ((PyTypeObject *)v->object)->tp_mro!=v->mro) return 1;
        if (v->kind!=1 && (PyObject *)Py_TYPE(v->object)!=v->mro) return 1;
    }
    return 0;
}

static PyObject *filter_dirty(PyObject *self,PyObject *state) {
    return PyBool_FromLong(binding_filter_dirty(get_filter(state)));
}

/* Conservative exact-identity rejection only. A dirty binding always reaches
 * Python resolution BEFORE a call can be rejected. Client profilers still see
 * all events through guarded_profile, independently of this observer filter. */
static int filter_observation(PyObject *filter, PyFrameObject *frame, int event, PyObject *arg) {
    if (filter==Py_None) return 1;
    if (event!=PyTrace_CALL && event!=PyTrace_C_CALL) return 0;
    BindingFilter *f=get_filter(filter);
    if (binding_filter_dirty(f)) return 1;
    if (event==PyTrace_CALL) {
        if (!frame->f_frame || !frame->f_frame->f_func) return 1;
        return PyDict_Contains(f->python,(PyObject *)frame->f_frame->f_func);
    }
    if (!arg || !PyCFunction_Check(arg)) return 1;
    PyObject *key=PyLong_FromVoidPtr(arg);
    if (!key) return -1;
    int found=PyDict_Contains(f->cfunc,key);Py_DECREF(key);
    if (found) return found;
    key=PyLong_FromVoidPtr(boundary_original_definition(((PyCFunctionObject *)arg)->m_ml));
    if (!key) return -1;
    found=PyDict_Contains(f->descriptor,key);Py_DECREF(key);
    return found;
}

#include "relation_guard.h"

static PyMethodDef methods[] = {
    {"native_transition_active", native_transition_active, METH_NOARGS, "Whether a protected native callback transition is in progress."},
    {"protect_native_boundary", protect_native_boundary, METH_O, "Preserve observers across an exact native profiler callable."},
    {"original_native_function", original_native_function, METH_O, "Original native address before boundary protection."},
    {"make_filter", make_filter, METH_VARARGS, "Build immutable exact binding/version snapshot."},
    {"filter_dirty", filter_dirty, METH_O, "Check native binding snapshot without Python traversal."},
    {"set_profile", set_profile, METH_VARARGS, "Install compiler-isolated observer trampoline."},
    {"current_profile", guard_current_profile, METH_NOARGS, "Current logical profiling callback."},
    {"install_guard", install_guard, METH_VARARGS, "Multiplex profiler with opcode guard."},
    {"remove_guard", remove_guard, METH_NOARGS, "Restore native profiling and tracing clients."},
    {"profile_client", profile_client, METH_NOARGS, "Current profiling client."},
    {"trace_client", trace_client, METH_NOARGS, "Current tracing client."},
    {"guard_intact", guard_intact, METH_NOARGS, "Whether both guard channels remain installed."},
    {"guard_available", guard_available, METH_NOARGS, "Whether the opcode guard can repair profiler changes."},
    {"protected_observer_available", protected_observer_available, METH_NOARGS, "Whether callback transitions are atomically multiplexed."},
    {"protected_backend_available", protected_backend_available, METH_NOARGS, "Whether the native callback transition adapter is loaded."},
    {"initialize_python_clients", initialize_python_clients, METH_VARARGS, "Save CPython's own Python callback trampolines."},
    {"set_python_client", set_python_client, METH_VARARGS, "Change one Python profiling client without replacing observers."},
    {"attach_callchain_observer", attach_callchain_observer, METH_VARARGS, "Attach independent callchain profile/trace observations."},
    {"detach_callchain_observer", detach_callchain_observer, METH_O, "Detach a protected callchain subscription."},
    {"frame_function", frame_function, METH_O, "Return exact executing Python function."},
    {"method_definition", method_definition, METH_O, "Return process-local method definition identity."},
    {"namespace_version", namespace_version, METH_O, "Namespace mutation counter for incremental resolution."},
    {"namespaces_changed", namespaces_changed, METH_VARARGS, "Check namespace mutations without Python-level scanning."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_relation_identity", NULL, -1, methods};
PyMODINIT_FUNC PyInit__relation_identity(void) { return PyModule_Create(&module); }
