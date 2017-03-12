import numpy as np

from scipy._lib.ccallback cimport (ccallback_t, ccallback_prepare,
                                   ccallback_prepare_obtain,
                                   ccallback_release_obtain,
                                   ccallback_release, CCALLBACK_DEFAULTS,
                                   ccallback_signature_t, ccallback_obtain)


cdef extern from "f2pycobject.h":
    object F2PyCapsule_FromVoidPtr(void *ptr, void *dtor)


#
# Supported callback signatures
#

cdef ccallback_signature_t fun_signatures[2]
cdef ccallback_signature_t jac_signatures[2]

fun_signatures[0].signature = b"void (int, double, double *, double *, void *)"
fun_signatures[0].value = 0
fun_signatures[1].signature = NULL

jac_signatures[0].signature = b"void (int, double, double *, int, int, double *, int, void *)"
jac_signatures[0].value = 0
jac_signatures[1].signature = NULL


#
# Callback thunks to be called from Fortran
#

cdef void fun_thunk(int *neq, double *t, double *y, double *ydot) nogil:
    cdef ccallback_t *callback

    callback = ccallback_obtain()

    # count low-level nfev
    callback.info += 1

    (<void(*)(int, double, double *, double *, void *) nogil>callback.c_function)(
        neq[0], t[0], y, ydot, callback.user_data)


cdef void jac_thunk(int *neq, double *t, double *y, int *ml, int *mu, double *jac, int *nrowpd) nogil:
    cdef ccallback_t *callback

    # ccallback_obtain was primed with fun_callback
    callback = ccallback_obtain()
    callback = <ccallback_t*>callback.info_p

    # count low-level njev
    callback.info += 1

    (<void(*)(int, double, double *, int, int, double *, int, void *) nogil>callback.c_function)(
        neq[0], t[0], y, ml[0], mu[0], jac, nrowpd[0], callback.user_data)


#
# Python-facing interface
#

cdef class LSODACallbacks:
    """
    Low-level callback functions for LSODA

    Parameters
    ----------
    fun : {callable, LowLevelCallable}
        Python callable or LowLevelCallable, for rhs function
    jac : {callable, LowLevelCallable}
        Python callable or LowLevelCallable, for jacobian

    Methods
    -------
    do_call

    Attributes
    ----------
    last_nfev : long
        Number of function evaluations in previous do_call
    last_njev : long
        Number of jacobian evaluations in previous do_call
    fun_obj : object
        f2py-compatible low-level callable for the rhs function.
        None if low-level callables are not used.
        This function pointer can *only* be used via the `do_call` method!
    jac_obj : object
        f2py-compatible low-level callable for the jacobian function
        None if low-level callables are not used.
        This function pointer can *only* be used via the `do_call` method!

    Notes
    -----
    The low-level function pointers are only usable during a call to ``do_call``.
    For LSODA, ``integrator.run(...)`` must be replaced by
    ``callbacks.do_call(integrator.run, ...)``.

    """

    cdef ccallback_t fun_callback
    cdef ccallback_t jac_callback
    cdef public long last_nfev, last_njev
    cdef public object fun_obj, jac_obj

    def __init__(self, fun, jac):
        ccallback_prepare(&self.fun_callback, fun_signatures, fun, CCALLBACK_DEFAULTS)
        if jac is None:
            jac = lambda: None
        ccallback_prepare(&self.jac_callback, jac_signatures, jac, CCALLBACK_DEFAULTS)

        if self.fun_callback.c_function != NULL:
            self.fun_obj = F2PyCapsule_FromVoidPtr(&fun_thunk, NULL)
        else:
            self.fun_obj = None

        if self.jac_callback.c_function != NULL:
            self.jac_obj = F2PyCapsule_FromVoidPtr(&jac_thunk, NULL)
        else:
            self.jac_obj = None

        self.last_nfev = 0
        self.last_njev = 0

    def __del__(self):
        ccallback_release(&self.fun_callback)
        ccallback_release(&self.jac_callback)

    def do_call(self, f, *args, **kwargs):
        """
        Call a function that calls the callbacks.

        .. warning::

           For LSODA, instead of ``integrator.run(...)``, you must always
           do ``callbacks.do_call(integrator.run, ...)``.

        """
        self.fun_callback.info_p = <void*>&self.jac_callback
        self.fun_callback.info = 0
        self.jac_callback.info = 0
        self.last_nfev = 0
        self.last_njev = 0

        ccallback_prepare_obtain(&self.fun_callback)
        try:
            return f(*args, **kwargs)
        finally:
            ccallback_release_obtain(&self.fun_callback)

            self.last_nfev += self.fun_callback.info
            self.last_njev += self.jac_callback.info


#
# Test functions
#

cdef api void fun_test(int neq, double t, double *y, double *ydot, void *user_data) nogil:
    cdef int k
    for k in range(neq):
        ydot[k] = 2 * y[k]


cdef api void jac_test(int neq, double t, double *y, int ml, int mu, double *jac, int nrowpd, void *user_data) nogil:
    cdef int k
    cdef double *pd
    pd = jac
    for k in range(nrowpd):
        pd[k] = 2
        pd = pd + nrowpd
