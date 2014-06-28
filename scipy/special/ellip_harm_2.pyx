import threading
import ctypes
import numpy as np


from _complexstuff cimport *
from libc.math cimport cos, sqrt
import scipy

cdef double _global_h2, _global_k2
cdef int _global_n, _global_p

from .ellip_harm cimport ellip_harmonic

_ellip_lock = threading.Lock()

cdef double _F_integrand(double t) nogil:
    cdef double h2, k2, t2, i, a
    cdef int n, p
    cdef double result
    t2 = t*t
   
    h2 = _global_h2
    k2 = _global_k2
    n = _global_n
    p = _global_p
    i = ellip_harmonic( h2, k2, n, p, t, 1, 1)
    a = sqrt(t2 - k2)*sqrt(t2 - h2)
    
    result = i*i*a
    result = 1/result
    return result

_F_integrand_t = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
_F_integrand_ctypes = ctypes.cast(<size_t>&_F_integrand, _F_integrand_t)
#del t

@np.vectorize
def _ellip_harm_2(double h2, double k2, int n, int p, double s):

    _global_h2 = h2
    _global_k2 = k2
    _global_n = n
    _global_p = p

    res, err = scipy.integrate.quad(_F_integrand_ctypes, s, np.inf,
                                    epsabs=1e-08, epsrel=1e-15)
    if abs(err) > 1e-10 * abs(res):
        return nan
    return res

def ellip_harm_2(h2, k2, n, p, s):
    with _ellip_lock:
        return _ellip_harm_2(h2, k2, n, p, s)
