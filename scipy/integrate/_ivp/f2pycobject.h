/*
 * Functions creating pointer objects that f2py understands.
 */
#ifndef F2PYCOBJECT_H_
#define F2PYCOBJECT_H_

#include <Python.h>

#if PY_VERSION_HEX >= 0x03000000

PyObject *
F2PyCapsule_FromVoidPtr(void *ptr, void *dtor)
{
    PyObject *ret = PyCapsule_New(ptr, NULL, (void(*)(PyObject *))dtor);
    if (ret == NULL) {
        PyErr_Clear();
    }
    return ret;
}

void *
F2PyCapsule_AsVoidPtr(PyObject *obj)
{
    void *ret = PyCapsule_GetPointer(obj, NULL);
    if (ret == NULL) {
        PyErr_Clear();
    }
    return ret;
}

int
F2PyCapsule_Check(PyObject *ptr)
{
    return PyCapsule_CheckExact(ptr);
}

#else

PyObject *
F2PyCapsule_FromVoidPtr(void *ptr, void *dtor)
{
    return PyCObject_FromVoidPtr(ptr, (void (*)(void *))dtor);
}

void *
F2PyCapsule_AsVoidPtr(PyObject *ptr)
{
    return PyCObject_AsVoidPtr(ptr);
}

int
F2PyCapsule_Check(PyObject *ptr)
{
    return PyCObject_Check(ptr);
}

#endif

#endif /* F2PYCOBJECT_H_ */
