/* CPython 3.11 independent observers and exact native transition boundaries.
 * Native profiler objects and callbacks are forwarded unchanged. No CPU assembly.
 * The trace channel is multiplexed, not replaced with a Python line observer.
 */
#include <stdint.h>
#include <dlfcn.h>

typedef struct RelationGuard {
    PyObject *observer;
    Py_tracefunc profile;
    PyObject *profile_obj;
    Py_tracefunc trace;
    PyObject *trace_obj;
    PyObject *frames;
    PyObject *extra_profile;
    PyObject *extra_trace;
    PyObject *extra_frames;
    PyThreadState *owner;
    PyObject *capsule;
    unsigned long generation;
    struct RelationGuard *next;
    int active;
    unsigned int client_profile_depth;
    unsigned int client_trace_depth;
} RelationGuard;

static int guarded_profile(PyObject *, PyFrameObject *, int, PyObject *);
static int guarded_trace(PyObject *, PyFrameObject *, int, PyObject *);
static RelationGuard *registered_guards;
static int boundary_protection_ready;
static Py_tracefunc python_profile_client;
static Py_tracefunc python_trace_client;
static void (*native_observer_enter)(void);
static void (*native_observer_leave)(void);

static void unregister_guard(RelationGuard *g) {
    RelationGuard **entry=&registered_guards;
    while (*entry && *entry!=g) entry=&(*entry)->next;
    if (*entry) *entry=g->next;
}

static RelationGuard *guard_data(PyObject *capsule) {
    return PyCapsule_GetPointer(capsule, "api_relation_guard");
}

static void guard_destroy(PyObject *capsule) {
    RelationGuard *g=guard_data(capsule);
    if (!g) {PyErr_Clear();return;}
    Py_XDECREF(g->observer);Py_XDECREF(g->profile_obj);
    unregister_guard(g);
    Py_XDECREF(g->trace_obj);Py_XDECREF(g->frames);
    Py_XDECREF(g->extra_profile);Py_XDECREF(g->extra_trace);
    Py_XDECREF(g->extra_frames);PyMem_Free(g);
}

static PyObject *extra_observation(RelationGuard *g,PyObject *callback,
                                    PyFrameObject *frame,int event,PyObject *arg) {
    static const char *names[]={"call","exception","line","return","c_call","c_exception","c_return","opcode"};
    PyObject *guard=PyTuple_GET_ITEM(g->observer,1),*previous=NULL;
    if (guard!=Py_None) {
        previous=PyObject_CallOneArg(guard,Py_None);
        if (!previous) return NULL;
    }
    PyObject *result=PyObject_CallFunction(callback,"OsO",(PyObject *)frame,names[event],arg ? arg : Py_None);
    PyObject *type=NULL,*value=NULL,*tb=NULL;
    if (!result) PyErr_Fetch(&type,&value,&tb);
    if (previous) {
        PyObject *restored=PyObject_CallOneArg(guard,previous);Py_DECREF(previous);
        if (!restored) {Py_XDECREF(result);Py_XDECREF(type);Py_XDECREF(value);Py_XDECREF(tb);return NULL;}
        Py_DECREF(restored);
    }
    if (!result) PyErr_Restore(type,value,tb);
    return result;
}

static int protected_trace(RelationGuard *g,PyFrameObject *frame,int event,PyObject *arg) {
    if (!g->extra_trace) return 0;
    PyObject *state=PyDict_GetItem(g->extra_frames,(PyObject *)frame);
    PyObject *callback=g->extra_trace;
    int opcode=0;
    if (event!=PyTrace_CALL) {
        if (!state) return 0;
        callback=PyTuple_GET_ITEM(state,0);opcode=PyTuple_GET_ITEM(state,1)==Py_True;
        if (event==PyTrace_OPCODE && !opcode) return 0;
    }
    Py_INCREF(callback);
    int saved=frame->f_trace_opcodes;frame->f_trace_opcodes=opcode;
    PyObject *result=extra_observation(g,callback,frame,event,arg);
    int requested=frame->f_trace_opcodes;frame->f_trace_opcodes=saved;
    if (!result) {Py_DECREF(callback);return -1;}
    if (event==PyTrace_RETURN || (event==PyTrace_CALL && result==Py_None)) {
        if (state && PyDict_DelItem(g->extra_frames,(PyObject *)frame)<0) {Py_DECREF(callback);Py_DECREF(result);return -1;}
    } else {
        PyObject *local=result==Py_None ? callback : result;
        PyObject *updated=PyTuple_Pack(2,local,requested ? Py_True : Py_False);
        if (!updated) {Py_DECREF(callback);Py_DECREF(result);return -1;}
        int rc=PyDict_SetItem(g->extra_frames,(PyObject *)frame,updated);Py_DECREF(updated);
        if (rc<0) {Py_DECREF(callback);Py_DECREF(result);return -1;}
    }
    Py_DECREF(callback);Py_DECREF(result);return 0;
}

static PyObject *find_guard(void) {
    PyThreadState *ts=PyThreadState_Get();
    if (ts->c_tracefunc==guarded_trace) return ts->c_traceobj;
    if (ts->c_profilefunc==guarded_profile) return ts->c_profileobj;
    return NULL;
}

/* CPython 3.11 call_trace builds a line cache BEFORE invoking our callback.
 * A synthetic traceback can copy a longer code object's co_linetable onto a
 * short eval code object. Stock _PyCode_CreateLineArray then writes beyond its
 * Py_SIZE(code)-entry allocation. Prepare only this overlong case at the exact
 * code.replace return boundary, before the new object can execute.
 * Keep the code object, bytecode, original position table and in-range mapping.
 * The interpreter owns/frees this ordinary PyMem cache exactly as before. */
static int prepare_code_line_cache(PyCodeObject *code) {
    if (code->_co_linearray) return 0;
    PyObject *ranges=PyObject_CallMethod((PyObject *)code,"co_lines",NULL);
    if (!ranges) return -1;
    PyObject *range;int overrun=0;
    Py_ssize_t units=Py_SIZE(code);
    while ((range=PyIter_Next(ranges))) {
        Py_ssize_t end=PyLong_AsSsize_t(PyTuple_GET_ITEM(range,1));
        Py_DECREF(range);
        if (end<0 && PyErr_Occurred()) {Py_DECREF(ranges);return -1;}
        if (end/2>units) {overrun=1;break;}
    }
    Py_DECREF(ranges);
    if (PyErr_Occurred()) return -1;
    if (!overrun) return 0;
    if (units<=0 || units>INT_MAX/2 || units>PY_SSIZE_T_MAX/(Py_ssize_t)sizeof(int32_t)) {
        PyErr_SetString(PyExc_ValueError,"invalid code size for tracing line cache");return -1;
    }
    int32_t *lines=PyMem_Malloc(units*sizeof(int32_t));
    if (!lines) {PyErr_NoMemory();return -1;}
    for (Py_ssize_t i=0;i<units;i++) lines[i]=PyCode_Addr2Line(code,(int)(i*2));
    code->_co_linearray_entry_size=4;
    code->_co_linearray=(char *)lines;
    return 0;
}

static int guard_frame(RelationGuard *g, PyFrameObject *frame) {
    if (prepare_code_line_cache(frame->f_frame->f_code)<0) return -1;
    if (!PyDict_GetItem(g->frames,(PyObject *)frame)) {
        uint64_t before=((PyDictObject *)g->frames)->ma_version_tag;
        if (PyDict_SetItem(g->frames,(PyObject *)frame,
                          frame->f_trace_opcodes ? Py_True : Py_False)<0) return -1;
        binding_private_frame_write(g->observer,g->frames,before);
    }
    frame->f_trace_opcodes=1;
    return 0;
}

static int guard_rearm_on(PyObject *capsule,PyThreadState *ts) {
    RelationGuard *g=guard_data(capsule);
    if (!g || !g->active) return 0;
    if (ts->c_profilefunc!=guarded_profile || ts->c_profileobj!=capsule) {
        PyObject *obj=Py_XNewRef(ts->c_profileobj);
        g->profile=ts->c_profilefunc;
        if (obj==PyTuple_GET_ITEM(g->observer,0)) {Py_CLEAR(obj);g->profile=NULL;}
        Py_XSETREF(g->profile_obj,obj);
        if (_PyEval_SetProfile(ts,guarded_profile,capsule)<0) return -1;
    }
    if (ts->c_tracefunc!=guarded_trace || ts->c_traceobj!=capsule) {
        PyObject *obj=Py_XNewRef(ts->c_traceobj);
        g->trace=ts->c_tracefunc;Py_XSETREF(g->trace_obj,obj);
        if (_PyEval_SetTrace(ts,guarded_trace,capsule)<0) return -1;
    }
    return 0;
}

static int guard_rearm(PyObject *capsule) {
    return guard_rearm_on(capsule,PyThreadState_Get());
}

/* CPython's Python trace/profile trampolines clear their global channel when
 * a client raises. Preserve that client-disable behavior and the exact error,
 * but restore our independent observer before returning to the interpreter. */
static int finish_client_callback(PyObject *capsule,int result) {
    PyObject *type=NULL,*value=NULL,*tb=NULL;
    if (result) PyErr_Fetch(&type,&value,&tb);
    int restored=guard_rearm(capsule);
    if (restored<0 && result) PyErr_WriteUnraisable(capsule);
    if (result) {PyErr_Restore(type,value,tb);return result;}
    return restored;
}

static PyObject *client_transition_active(PyObject *self,PyObject *channel) {
    long which=PyLong_AsLong(channel);
    if (which==-1 && PyErr_Occurred()) return NULL;
    PyObject *capsule=find_guard();
    if (!capsule) Py_RETURN_FALSE;
    RelationGuard *g=guard_data(capsule);
    return PyBool_FromLong(which ? g->client_trace_depth!=0 : g->client_profile_depth!=0);
}

/* A native profiler changes CPython callbacks inside a C call. Rearming on the
 * next opcode is too late: the genuine C_RETURN was already lost. Protect the
 * exact native callable's entry in place, rearming before it returns to ceval.
 * Object identity, calling convention, arguments, result and exception survive.
 * No invented call/return events or guessed unwinds are emitted. */
typedef struct NativeBoundary {
    PyObject *target;
    PyMethodDef *original;
    PyCFunction original_function;
    PyMethodDef replacement;
} NativeBoundary;
static NativeBoundary native_boundaries[16];
static int boundary_count;
static Py_tss_t native_transition_key=Py_tss_NEEDS_INIT;

static PyObject *boundary_guards(void) {
    PyObject *held=PyList_New(0);
    if (!held) return NULL;
    for (RelationGuard *g=registered_guards;g;g=g->next)
        if (g->active && PyList_Append(held,g->capsule)<0) {Py_DECREF(held);return NULL;}
    if (!PyThread_tss_is_created(&native_transition_key) && PyThread_tss_create(&native_transition_key)!=0) {
        Py_DECREF(held);PyErr_SetString(PyExc_RuntimeError,"native transition state unavailable");return NULL;
    }
    uintptr_t depth=(uintptr_t)PyThread_tss_get(&native_transition_key);
    PyThread_tss_set(&native_transition_key,(void *)(depth+1));
    return held;
}
static PyObject *native_transition_active(PyObject *self,PyObject *unused) {
    return PyBool_FromLong(PyThread_tss_is_created(&native_transition_key) && PyThread_tss_get(&native_transition_key)!=NULL);
}
static PyObject *boundary_finish(PyObject *held,PyObject *result) {
    if (result && PyCode_Check(result) && PyList_GET_SIZE(held)>0 && prepare_code_line_cache((PyCodeObject *)result)<0) {
        Py_DECREF(result);result=NULL;
    }
    PyObject *type=NULL,*value=NULL,*tb=NULL;
    if (!result) PyErr_Fetch(&type,&value,&tb);
    for (Py_ssize_t i=0;i<PyList_GET_SIZE(held);i++) {
        PyObject *capsule=PyList_GET_ITEM(held,i);
        RelationGuard *g=guard_data(capsule);
        if (!g || !g->active) continue;
        /* A native profiler may replace callbacks on every live thread. Never
         * dereference a saved thread-state pointer after the thread has exited. */
        PyInterpreterState *interp=PyThreadState_Get()->interp;
        for (PyThreadState *ts=PyInterpreterState_ThreadHead(interp);ts;ts=PyThreadState_Next(ts))
            if (ts==g->owner) {
                if (guard_rearm_on(capsule,ts)<0) {
                    /* Restoration rejection is observable through guard_intact;
                     * preserve the workload's own return/exception. */
                    PyErr_WriteUnraisable(g->capsule);
                }
                break;
            }
    }
    Py_DECREF(held);
    uintptr_t depth=(uintptr_t)PyThread_tss_get(&native_transition_key);
    PyThread_tss_set(&native_transition_key,(void *)(depth-1));
    if (!result) PyErr_Restore(type,value,tb);
    return result;
}
#define BOUNDARY(N) \
static PyObject *boundary_kw_##N(PyObject *self,PyObject *args,PyObject *kw) { \
 PyObject *held=boundary_guards();if (!held) return NULL; \
 return boundary_finish(held,((PyCFunctionWithKeywords)native_boundaries[N].original_function)(self,args,kw));} \
static PyObject *boundary_fast_##N(PyObject *self,PyObject *const *args,Py_ssize_t nargs,PyObject *kw) { \
 PyObject *held=boundary_guards();if (!held) return NULL; \
 return boundary_finish(held,((_PyCFunctionFastWithKeywords)native_boundaries[N].original_function)(self,args,nargs,kw));} \
static PyObject *boundary_plain_##N(PyObject *self,PyObject *args) { \
 PyObject *held=boundary_guards();if (!held) return NULL; \
 return boundary_finish(held,native_boundaries[N].original_function(self,args));}
BOUNDARY(0) BOUNDARY(1) BOUNDARY(2) BOUNDARY(3)
BOUNDARY(4) BOUNDARY(5) BOUNDARY(6) BOUNDARY(7)
BOUNDARY(8) BOUNDARY(9) BOUNDARY(10) BOUNDARY(11)
BOUNDARY(12) BOUNDARY(13) BOUNDARY(14) BOUNDARY(15)
#undef BOUNDARY
#define ENTRY(N) {_PyCFunction_CAST(boundary_plain_##N),_PyCFunction_CAST(boundary_kw_##N),_PyCFunction_CAST(boundary_fast_##N)}
static PyCFunction boundary_functions[][3]={
 ENTRY(0),ENTRY(1),ENTRY(2),ENTRY(3),ENTRY(4),ENTRY(5),ENTRY(6),ENTRY(7),
 ENTRY(8),ENTRY(9),ENTRY(10),ENTRY(11),ENTRY(12),ENTRY(13),ENTRY(14),ENTRY(15)};
#undef ENTRY

static PyMethodDef *boundary_original_definition(PyMethodDef *definition) {
    for (int i=0;i<boundary_count;i++)
        if (definition==&native_boundaries[i].replacement) return native_boundaries[i].original;
    return definition;
}
static int writable_definition(PyMethodDef *definition) {
    FILE *maps=fopen("/proc/self/maps","r");
    if (!maps) return 0;
    char line[1024],permissions[8];unsigned long long lo,hi;int writable=0;
    uintptr_t address=(uintptr_t)definition;
    while (fgets(line,sizeof(line),maps))
        if (sscanf(line,"%llx-%llx %7s",&lo,&hi,permissions)==3 && address>=lo &&
            address+sizeof(PyMethodDef)<=hi && permissions[1]=='w') {writable=1;break;}
    fclose(maps);return writable;
}
static PyObject *protect_native_boundary(PyObject *self,PyObject *target) {
    PyMethodDef **definition;
    if (PyCFunction_Check(target)) definition=&((PyCFunctionObject *)target)->m_ml;
    else if (Py_IS_TYPE(target,&PyMethodDescr_Type)) definition=&((PyMethodDescrObject *)target)->d_method;
    else {PyErr_SetString(PyExc_TypeError,"native function or method descriptor required");return NULL;}
    for (int i=0;i<boundary_count;i++)
        if (native_boundaries[i].target==target) Py_RETURN_NONE;
    if (boundary_count>=16) {PyErr_SetString(PyExc_RuntimeError,"native boundary capacity exhausted");return NULL;}
    int flags=(*definition)->ml_flags & ~(METH_CLASS|METH_STATIC|METH_COEXIST);
    int kind;
    if (flags==(METH_VARARGS|METH_KEYWORDS)) kind=1;
    else if (flags==(METH_FASTCALL|METH_KEYWORDS)) kind=2;
    else if (flags==METH_VARARGS || flags==METH_NOARGS || flags==METH_O) kind=0;
    else {PyErr_SetString(PyExc_TypeError,"unsupported native calling convention");return NULL;}
    NativeBoundary *b=&native_boundaries[boundary_count];
    int descriptor=Py_IS_TYPE(target,&PyMethodDescr_Type);
    if (descriptor && !writable_definition(*definition)) {
        PyErr_SetString(PyExc_RuntimeError,"native method definition is not writable; saved-method protection unavailable");return NULL;
    }
    b->target=Py_NewRef(target);b->original=*definition;b->original_function=(*definition)->ml_meth;b->replacement=**definition;
    b->replacement.ml_meth=boundary_functions[boundary_count][kind];
    if (descriptor) (*definition)->ml_meth=b->replacement.ml_meth;
    else *definition=&b->replacement;
    boundary_count++;
    Py_RETURN_NONE;
}
static PyObject *original_native_function(PyObject *self,PyObject *target) {
    if (!PyCFunction_Check(target)) {PyErr_SetString(PyExc_TypeError,"native function required");return NULL;}
    PyMethodDef *definition=((PyCFunctionObject *)target)->m_ml;
    for (int i=0;i<boundary_count;i++)
        if (definition==native_boundaries[i].original || definition==&native_boundaries[i].replacement)
            return PyLong_FromVoidPtr((void *)native_boundaries[i].original_function);
    return PyLong_FromVoidPtr((void *)definition->ml_meth);
}

static int guarded_profile(PyObject *capsule, PyFrameObject *frame, int event, PyObject *arg) {
    if (native_observer_enter) native_observer_enter();
    Py_INCREF(capsule);
    RelationGuard *g=guard_data(capsule);
    int result=0;
    if (g && g->active) {
        result=relation_profile(g->observer,frame,event,arg);
        if (!result && g->active && g->extra_profile) {
            PyObject *extra=extra_observation(g,g->extra_profile,frame,event,arg);
            if (!extra) result=-1;
            else Py_DECREF(extra);
        }
        if (!result && g->active && g->profile) {
            PyObject *client=Py_XNewRef(g->profile_obj);
            Py_tracefunc fn=g->profile;
            g->client_profile_depth++;
            result=fn(client,frame,event,arg);
            result=finish_client_callback(capsule,result);
            g->client_profile_depth--;
            Py_XDECREF(client);
        }
    } else if (g && g->profile) {
        result=g->profile(g->profile_obj,frame,event,arg);
    }
    Py_DECREF(capsule);
    if (native_observer_leave) native_observer_leave();
    return result;
}

static int guarded_trace(PyObject *capsule, PyFrameObject *frame, int event, PyObject *arg) {
    if (native_observer_enter) native_observer_enter();
    Py_INCREF(capsule);
    RelationGuard *g=guard_data(capsule);
    int result=0;
    if (!g) goto done;
    if (!g->active) {
        if (g->trace) result=g->trace(g->trace_obj,frame,event,arg);
        goto done;
    }
    if (prepare_code_line_cache(frame->f_frame->f_code)<0) {result=-1;goto done;}
    if (protected_trace(g,frame,event,arg)<0) {result=-1;goto done;}
    if (event==PyTrace_OPCODE && !g->trace) {
        /* Keep every-opcode rearming; no client needs frame/flag lookups here. */
        result=guard_rearm(capsule);goto done;
    }
    if (guard_frame(g,frame)<0 || guard_rearm(capsule)<0) {result=-1;goto done;}
    PyObject *opcode=PyDict_GetItem(g->frames,(PyObject *)frame);
    if (g->trace && (event!=PyTrace_OPCODE || opcode==Py_True)) {
        /* Restore the client's own flag while invoking its callback. */
        frame->f_trace_opcodes=(opcode==Py_True);
        PyObject *client=Py_XNewRef(g->trace_obj);
        Py_tracefunc fn=g->trace;
        g->client_trace_depth++;
        result=fn(client,frame,event,arg);
        result=finish_client_callback(capsule,result);
        g->client_trace_depth--;
        Py_XDECREF(client);
        if (!result && g->active) {
            uint64_t before=((PyDictObject *)g->frames)->ma_version_tag;
            result=PyDict_SetItem(g->frames,(PyObject *)frame,
                                 frame->f_trace_opcodes ? Py_True : Py_False);
            if (!result) binding_private_frame_write(g->observer,g->frames,before);
        }
        if (g->active) frame->f_trace_opcodes=1;
    }
    if (g->active && event==PyTrace_RETURN) {
        opcode=PyDict_GetItem(g->frames,(PyObject *)frame);
        frame->f_trace_opcodes=(opcode==Py_True);
        if (opcode) {
            uint64_t before=((PyDictObject *)g->frames)->ma_version_tag;
            if (PyDict_DelItem(g->frames,(PyObject *)frame)<0) result=-1;
            else binding_private_frame_write(g->observer,g->frames,before);
        }
    }
done:
    Py_DECREF(capsule);
    if (native_observer_leave) native_observer_leave();
    return result;
}

static PyObject *install_guard(PyObject *self, PyObject *args) {
    PyObject *callback,*eval_guard,*filter=Py_None;
    if (!PyArg_ParseTuple(args,"OO|O",&callback,&eval_guard,&filter)) return NULL;
    PyObject *existing=find_guard();
    if (existing) {
        if (guard_rearm(existing)<0) return NULL;
        Py_RETURN_NONE;
    }
    if (!PyCallable_Check(callback) || (eval_guard!=Py_None && !PyCFunction_Check(eval_guard))) {
        PyErr_SetString(PyExc_TypeError,"Invalid relation observer");return NULL;
    }
    RelationGuard *g=PyMem_Calloc(1,sizeof(RelationGuard));
    if (!g) return PyErr_NoMemory();
    PyObject *capsule=PyCapsule_New(g,"api_relation_guard",guard_destroy);
    if (!capsule) {PyMem_Free(g);return NULL;}
    PyThreadState *ts=PyThreadState_Get();
    g->observer=PyTuple_Pack(3,callback,eval_guard,filter);g->frames=PyDict_New();
    g->extra_frames=PyDict_New();g->owner=ts;g->capsule=capsule;
    g->profile=ts->c_profilefunc;g->profile_obj=Py_XNewRef(ts->c_profileobj);
    g->trace=ts->c_tracefunc;g->trace_obj=Py_XNewRef(ts->c_traceobj);
    if (!g->observer || !g->frames || !g->extra_frames) {Py_DECREF(capsule);return NULL;}
    g->active=1;
    native_observer_enter=(void (*)(void))dlsym(RTLD_DEFAULT,"trace_observer_enter");
    native_observer_leave=(void (*)(void))dlsym(RTLD_DEFAULT,"trace_observer_leave");
    if (!native_observer_enter || !native_observer_leave) native_observer_enter=native_observer_leave=NULL;
    g->next=registered_guards;registered_guards=g;
    PyFrameObject *frame=PyThreadState_GetFrame(ts);
    while (frame) {
        if (guard_frame(g,frame)<0) {Py_DECREF(frame);Py_DECREF(capsule);return NULL;}
        PyFrameObject *back=PyFrame_GetBack(frame);Py_DECREF(frame);frame=back;
    }
    PyEval_SetTrace(guarded_trace,capsule);
    PyEval_SetProfile(guarded_profile,capsule);
    Py_DECREF(capsule);
    if (PyErr_Occurred()) return NULL;
    Py_RETURN_NONE;
}

static PyObject *remove_guard(PyObject *self, PyObject *unused) {
    PyObject *capsule=find_guard();
    if (!capsule) Py_RETURN_NONE;
    Py_INCREF(capsule);
    RelationGuard *g=guard_data(capsule);g->active=0;
    unregister_guard(g);
    Py_ssize_t pos=0;PyObject *frame,*opcode;
    while (PyDict_Next(g->frames,&pos,&frame,&opcode))
        ((PyFrameObject *)frame)->f_trace_opcodes=(opcode==Py_True);
    PyDict_Clear(g->frames);
    PyThreadState *ts=PyThreadState_Get();
    if (ts->c_tracefunc==guarded_trace) PyEval_SetTrace(g->trace,g->trace_obj);
    if (ts->c_profilefunc==guarded_profile) PyEval_SetProfile(g->profile,g->profile_obj);
    Py_DECREF(capsule);
    if (PyErr_Occurred()) return NULL;
    Py_RETURN_NONE;
}

static PyObject *profile_client(PyObject *self, PyObject *unused) {
    PyThreadState *ts=PyThreadState_Get();
    PyObject *obj=ts->c_profileobj;
    if (ts->c_profilefunc==guarded_profile) obj=guard_data(obj)->profile_obj;
    return Py_NewRef(obj ? obj : Py_None);
}

static PyObject *trace_client(PyObject *self, PyObject *unused) {
    PyThreadState *ts=PyThreadState_Get();
    PyObject *obj=ts->c_traceobj;
    if (ts->c_tracefunc==guarded_trace) obj=guard_data(obj)->trace_obj;
    return Py_NewRef(obj ? obj : Py_None);
}

static PyObject *guard_current_profile(PyObject *self, PyObject *unused) {
    PyThreadState *ts=PyThreadState_Get();
    if (ts->c_profilefunc==guarded_profile)
        return Py_NewRef(PyTuple_GET_ITEM(guard_data(ts->c_profileobj)->observer,0));
    return current_profile(self,unused);
}

static PyObject *guard_intact(PyObject *self, PyObject *unused) {
    PyThreadState *ts=PyThreadState_Get();
    return PyBool_FromLong(ts->c_tracefunc==guarded_trace && ts->c_profilefunc==guarded_profile);
}

static PyObject *guard_available(PyObject *self, PyObject *unused) {
    return PyBool_FromLong(PyThreadState_Get()->c_tracefunc==guarded_trace);
}

static PyObject *protected_observer_available(PyObject *self,PyObject *unused) {
    PyObject *capsule=find_guard();
    return PyBool_FromLong(boundary_protection_ready && capsule && guard_data(capsule)->active);
}

static PyObject *protected_backend_available(PyObject *self,PyObject *unused) {
    Py_RETURN_TRUE;
}

/* Capture CPython's own trampolines before installation. This preserves its
 * Python callback and frame-local tracing semantics, without guessing offsets. */
static PyObject *initialize_python_clients(PyObject *self,PyObject *args) {
    PyObject *profile_setter,*trace_setter,*callback;
    if (!PyArg_ParseTuple(args,"OOO",&profile_setter,&trace_setter,&callback)) return NULL;
    if (python_profile_client && python_trace_client) Py_RETURN_NONE;
    PyThreadState *ts=PyThreadState_Get();
    Py_tracefunc old_profile=ts->c_profilefunc,old_trace=ts->c_tracefunc;
    PyObject *profile_obj=Py_XNewRef(ts->c_profileobj),*trace_obj=Py_XNewRef(ts->c_traceobj);
    PyObject *result=PyObject_CallOneArg(profile_setter,callback);
    if (result) {
        Py_DECREF(result);python_profile_client=ts->c_profilefunc;
        result=PyObject_CallOneArg(trace_setter,callback);
        if (result) {Py_DECREF(result);python_trace_client=ts->c_tracefunc;}
    }
    PyObject *type=NULL,*value=NULL,*tb=NULL;
    if (!result) PyErr_Fetch(&type,&value,&tb);
    PyEval_SetTrace(old_trace,trace_obj);PyEval_SetProfile(old_profile,profile_obj);
    Py_XDECREF(profile_obj);Py_XDECREF(trace_obj);
    if (!result) {PyErr_Restore(type,value,tb);return NULL;}
    boundary_protection_ready=1;
    Py_RETURN_NONE;
}

static PyObject *set_python_client(PyObject *self,PyObject *args) {
    int channel;PyObject *callback,*capsule;
    if (!PyArg_ParseTuple(args,"iO",&channel,&callback)) return NULL;
    if (channel<0 || channel>1 || (callback!=Py_None && !PyCallable_Check(callback))) {
        PyErr_SetString(PyExc_TypeError,"profile/trace client must be callable or None");return NULL;
    }
    if (!python_profile_client || !python_trace_client) {
        PyErr_SetString(PyExc_RuntimeError,"Python callback trampolines were not initialized");return NULL;
    }
    if (PySys_Audit(channel ? "sys.settrace" : "sys.setprofile",NULL)<0) return NULL;
    capsule=find_guard();
    if (!capsule || !guard_data(capsule)->active) {
        PyErr_SetString(PyExc_RuntimeError,"observer guard disappeared during client update");return NULL;
    }
    RelationGuard *g=guard_data(capsule);PyObject *incoming=callback==Py_None ? NULL : Py_NewRef(callback);
    if (!channel) {g->profile=callback==Py_None ? NULL : python_profile_client;Py_XSETREF(g->profile_obj,incoming);}
    else {g->trace=callback==Py_None ? NULL : python_trace_client;Py_XSETREF(g->trace_obj,incoming);}
    Py_RETURN_NONE;
}

static PyObject *attach_callchain_observer(PyObject *self,PyObject *args) {
    PyObject *profile,*trace,*capsule=find_guard();
    if (!PyArg_ParseTuple(args,"OO",&profile,&trace)) return NULL;
    if (!boundary_protection_ready || !capsule) {PyErr_SetString(PyExc_RuntimeError,"protected callback guard is required");return NULL;}
    RelationGuard *g=guard_data(capsule);
    if (g->extra_profile || !PyCallable_Check(profile) || !PyCallable_Check(trace)) {
        PyErr_SetString(PyExc_RuntimeError,"callchain observer already attached or invalid");return NULL;
    }
    g->extra_profile=Py_NewRef(profile);g->extra_trace=Py_NewRef(trace);g->generation++;
    return PyLong_FromUnsignedLong(g->generation);
}

static PyObject *detach_callchain_observer(PyObject *self,PyObject *token) {
    PyObject *capsule=find_guard();
    unsigned long value=PyLong_AsUnsignedLong(token);
    if (PyErr_Occurred()) return NULL;
    if (!capsule || guard_data(capsule)->generation!=value || !guard_data(capsule)->extra_profile) {
        PyErr_SetString(PyExc_RuntimeError,"protected observer state was replaced");return NULL;
    }
    RelationGuard *g=guard_data(capsule);
    Py_CLEAR(g->extra_profile);Py_CLEAR(g->extra_trace);PyDict_Clear(g->extra_frames);
    Py_RETURN_NONE;
}
