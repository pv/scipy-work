import threading
import ctypes
from _complexstuff cimport *
from libc.math cimport sqrt
import scipy.integrate
import numpy as np

cdef double _global_h2, _global_k2
cdef int _global_n, _global_p

from .ellip_harm cimport ellip_harmonic

cdef double _F_integrand1(double t) nogil:
    cdef double h2, k2, t2, i, a, h
    cdef int n, p
    cdef double result
    t2 = t*t
    h2 = _global_h2
    k2 =_global_k2
    n = _global_n
    p = _global_p
    h = sqrt(h2)
    k = sqrt(k2)
    i = ellip_harmonic( h2, k2, n, p, t, 1, 1)
    result = i*i/sqrt((t + h)*(t + k))
    
    return result

_F_integrand1_t = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
_F_integrand1_ctypes = ctypes.cast(<size_t>&_F_integrand1, _F_integrand1_t)
#del t

cdef double _F_integrand2(double t) nogil:
    cdef double h2, k2, t2, i, a, h
    cdef int n, p
    cdef double result
    t2 = t*t
    h2 = _global_h2
    k2 =_global_k2
    n = _global_n
    p = _global_p
    h = sqrt(h2)
    k = sqrt(k2)
    i = ellip_harmonic( h2, k2, n, p, t, 1, 1)
    result = t2*i*i/sqrt((t + h)*(t + k))
    
    return result

_F_integrand2_t = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
_F_integrand2_ctypes = ctypes.cast(<size_t>&_F_integrand2, _F_integrand2_t)
#del t

cdef double _F_integrand3(double t) nogil:
    cdef double h2, k2, t2, i, a, h
    cdef int n, p
    cdef double result
    t2 = t*t
    h2 = _global_h2
    k2 =_global_k2
    n = _global_n
    p = _global_p
    h = sqrt(h2)
    k = sqrt(k2)
    i = ellip_harmonic( h2, k2, n, p, t, 1, 1)
    result = i*i/sqrt((t + h)*(k2 - t2))
    return result

_F_integrand3_t = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
_F_integrand3_ctypes = ctypes.cast(<size_t>&_F_integrand3, _F_integrand3_t)
#del t

cdef double _F_integrand4(double t) nogil:
    cdef double h2, k2, t2, i, a, h
    cdef int n, p
    cdef double result
    t2 = t*t
    h2 = _global_h2
    k2 =_global_k2
    n = _global_n
    p = _global_p
    h = sqrt(h2)
    k = sqrt(k2)
    i = ellip_harmonic( h2, k2, n, p, t, 1, 1)
    result = i*i*t2/sqrt((t + h)*(k2 - t2))
    return result

_F_integrand4_t = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
_F_integrand4_ctypes = ctypes.cast(<size_t>&_F_integrand4, _F_integrand4_t)
#del t

def _ellipsoid_norm(double h2, double k2, int n, int p):
    global _global_h2
    global _global_k2
    global _global_n
    global _global_p

    _global_h2 = h2
    _global_k2 = k2
    _global_n = n
    _global_p = p
    h = sqrt(h2)
    k = sqrt(k2)
    res, err = scipy.integrate.quad(_F_integrand1_ctypes, h, k,
                                    epsabs=1e-08, epsrel=1e-15, weight="alg", wvar=(-0.5, -0.5))
#    if abs(err) > 1e-10 * abs(res):
 #       return nan

    res1, err1 = scipy.integrate.quad(_F_integrand2_ctypes, h, k,
                                    epsabs=1e-08, epsrel=1e-15, weight="alg", wvar=(-0.5, -0.5))
  #  if abs(err1) > 1e-10 * abs(res1):
   #     return nan

    res2, err2 = scipy.integrate.quad(_F_integrand3_ctypes, 0, h,
                                    epsabs=1e-08, epsrel=1e-15, weight="alg", wvar=(0, -0.5))
  #  if abs(err2) > 1e-10 * abs(res2):
   #     return nan

    res3, err3 = scipy.integrate.quad(_F_integrand4_ctypes, 0, h,
                                    epsabs=1e-08, epsrel=1e-15, weight="alg", wvar=(0, -0.5))
#    if abs(err3) > 1e-10 * abs(res3):
 #       return nan

    return 8*(res1*res2 - res*res3)

