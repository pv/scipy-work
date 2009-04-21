/*
 * Copyright (c) 2009  Pauli Virtanen
 */
#include <stdio.h>
#include <stdarg.h>

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

#include "scipyfunc.h"
#include "scipyfunc_docstrings.h"

/*
 * Tools for creating ufunc objects
 */

#define ADD_UFUNC(name, doc, functions, data, nin, nout)                \
    do {                                                                \
        int ntypes = sizeof(functions)/sizeof(PyUFuncGenericFunction);  \
        size_t _sz_types = sizeof(char)*ntypes*(nin+nout);              \
        char *_types = (char*)PyMem_Malloc(_sz_types);                  \
        size_t _sz_functions = sizeof(PyUFuncGenericFunction)*ntypes;   \
        PyUFuncGenericFunction *_functions =                            \
            (PyUFuncGenericFunction *)PyMem_Malloc(_sz_functions);      \
        void **_data = (void**)PyMem_Malloc(sizeof(void*)*ntypes);      \
        char *_name, *_doc;                                             \
        PyObject *_ufunc;                                               \
        memcpy(_functions, functions, _sz_functions);                   \
        memcpy(_data, data, sizeof(void*)*ntypes);                      \
        _name = strdup(name);                                           \
        _doc = strdup(doc);                                             \
        scipyfunc_get_types(name, _functions, _types, nin, nout, ntypes); \
        _ufunc = PyUFunc_FromFuncAndData(_functions, _data, _types,     \
                                         ntypes, nin, nout,             \
                                         PyUFunc_None, _name, _doc, 0); \
        PyDict_SetItemString(dictionary, name, _ufunc);                 \
        Py_DECREF(_ufunc);                                              \
    } while (0)

static void scipyfunc_get_types(char *name,
                                PyUFuncGenericFunction *functions,
                                char *types,
                                int nin, int nout, int ntypes)
{
    int k;
    int j = 0;
    for (k = 0; k < ntypes; ++k) {
        if (functions[k] == PyUFunc_ff_f) {
            types[j++] = PyArray_FLOAT;
            types[j++] = PyArray_FLOAT;
            types[j++] = PyArray_FLOAT;
        } else if (functions[k] == PyUFunc_dd_d) {
            types[j++] = PyArray_DOUBLE;
            types[j++] = PyArray_DOUBLE;
            types[j++] = PyArray_DOUBLE;
        } else if (functions[k] == PyUFunc_gg_g) {
            types[j++] = PyArray_LONGDOUBLE;
            types[j++] = PyArray_LONGDOUBLE;
            types[j++] = PyArray_LONGDOUBLE;
        } else if (functions[k] == PyUFunc_f_f) {
            types[j++] = PyArray_FLOAT;
            types[j++] = PyArray_FLOAT;
        } else if (functions[k] == PyUFunc_d_d) {
            types[j++] = PyArray_DOUBLE;
            types[j++] = PyArray_DOUBLE;
        } else if (functions[k] == PyUFunc_g_g) {
            types[j++] = PyArray_LONGDOUBLE;
            types[j++] = PyArray_LONGDOUBLE;
        } else {
            char msg[1024];
            PyOS_snprintf(msg, 1024,
                          "scipyfunc: invalid ufunc specifier %d for %s",
                          k, name);
            Py_FatalError(msg);
        }
        if (j > (nin+nout)*ntypes) {
            char msg[1024];
            PyOS_snprintf(msg, 1024,
                          "scipyfunc: invalid ufunc type position %d for %s",
                          j, name);
            Py_FatalError(msg);
        }
    }
    types[j] = 0;
}

/*
 * Create ufunc objects
 */

static void scipyfunc_init_ufuncs(PyObject *dictionary)
{
    {
        PyUFuncGenericFunction f[] = {PyUFunc_ff_f,PyUFunc_dd_d,PyUFunc_gg_g};
        void *d[] = {scf_ivf, scf_iv, scf_ivl};
        ADD_UFUNC("iv", iv_docstring, f, d, 2, 1);
    }
    {
        PyUFuncGenericFunction f[] = {PyUFunc_f_f,PyUFunc_d_d,PyUFunc_g_g};
        void *d[] = {scf_gammaf, scf_gamma, scf_gammal};
        ADD_UFUNC("gamma", gamma_docstring, f, d, 1, 1);
    }
}


/*
 * Warnings and errors
 */

static PyObject *scipyfunc_SpecialFunctionWarning = NULL;

static void scipyfuncmodule_raise_warning(char *fmt, ...)
{
    NPY_ALLOW_C_API_DEF
    char msg[1024];
    va_list ap;

    va_start(ap, fmt);
    PyOS_vsnprintf(msg, 1024, fmt, ap);
    va_end(ap);

    NPY_ALLOW_C_API
    PyErr_Warn(scipyfunc_SpecialFunctionWarning, msg);
    NPY_DISABLE_C_API
}

static void scipyfuncmodule_error_handler(char *name, int code, char *code_name,
                                          char *msg)
{
    scipyfuncmodule_raise_warning("%s: %s (%s)", name, code_name, msg);
}


/*
 * Initialize the module
 */

static struct PyMethodDef methods[] = {
    {NULL,		NULL, 0}		/* sentinel */
};

PyMODINIT_FUNC initscipyfunc(void)
{
    PyObject *m, *d, *s;

    /* Create the module and add the functions */
    m = Py_InitModule("scipyfunc", methods);

    /* Import the ufunc objects */
    import_array();
    import_ufunc();

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);

    /* Add warning object */
    scipyfunc_SpecialFunctionWarning = PyErr_NewException(
        "scipyfunc.SpecialFunctionWarning", PyExc_RuntimeWarning, NULL);
    PyModule_AddObject(m, "SpecialFunctionWarning",
                       scipyfunc_SpecialFunctionWarning);

    /* Load the scipyfunc ufuncs into the namespace */
    scipyfunc_init_ufuncs(d);

    /* Set error/warning handler */
    scf_error_set_handler(scipyfuncmodule_error_handler);

    /* Check for errors */
    if (PyErr_Occurred())
        Py_FatalError("can't initialize module _scipyfunc");
}
