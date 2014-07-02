import threading
import ctypes
from _complexstuff cimport *
from libc.math cimport sqrt
import scipy.integrate
import numpy as np

cdef double _global_h2, _global_k2
cdef int _global_n, _global_p

from .ellip_harm cimport ellip_harmonic

cdef double _F_integrand(double t) nogil:
    cdef double h2, k2, t2, i, a
    cdef int n, p
    cdef double result
    t2 = t*t
    h2 = _global_h2
    k2 =_global_k2
    n = _global_n
    p = _global_p
    i = ellip_harmonic( h2, k2, n, p, 1/t, 1, 1)
    result = 1/(i*i*sqrt((1- k2*t2)*(1 - h2*t2)))
    
    return result

_F_integrand_t = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
_F_integrand_ctypes = ctypes.cast(<size_t>&_F_integrand, _F_integrand_t)
#del t

def _ellipsoid(double h2, double k2, int n, int p, double s):
    global _global_h2
    global _global_k2
    global _global_n
    global _global_p

    _global_h2 = h2
    _global_k2 = k2
    _global_n = n
    _global_p = p

    res, err = scipy.integrate.quad(_F_integrand_ctypes, 0, 1/s,
                                    epsabs=1e-08, epsrel=1e-15)
    if abs(err) > 1e-10 * abs(res):
        return nan
    res = (2*n + 1)*res*ellip_harmonic(h2, k2, n, p, s, 1, 1) 
    return res

