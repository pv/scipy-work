# -*-cython-*-
#
# Implementation of spherical Bessel functions and modified spherical Bessel
# functions of the first and second kinds.
#
# Author: Tadeusz Pudlik
#
# Distributed under the same license as SciPy.

import cython
from libc.math cimport cos, sin, sqrt

from numpy cimport npy_cdouble
from _complexstuff cimport *

cimport sf_error

cdef extern from "amos_wrappers.h":
    npy_cdouble cbesi_wrap( double v, npy_cdouble z) nogil
    npy_cdouble cbesj_wrap(double v, npy_cdouble z) nogil
    double cbesj_wrap_real(double v, double x) nogil
    npy_cdouble cbesk_wrap(double v, npy_cdouble z) nogil
    double cbesk_wrap_real(double v, double x) nogil
    npy_cdouble cbesy_wrap(double v, npy_cdouble z) nogil
    double cbesy_wrap_real(double v, double x) nogil

cdef extern from "cephes.h":
    double iv(double v, double x) nogil

# Fused type wrappers

cdef inline number_t cbesj(double v, number_t z) nogil:
    cdef npy_cdouble r
    if number_t is double:
        return cbesj_wrap_real(v, z)
    else:
        r = cbesj_wrap(v, (<npy_cdouble*>&z)[0])
        return (<number_t*>&r)[0]

cdef inline number_t cbesy(double v, number_t z) nogil:
    cdef npy_cdouble r
    if number_t is double:
        return cbesy_wrap_real(v, z)
    else:
        r = cbesy_wrap(v, (<npy_cdouble*>&z)[0])
        return (<number_t*>&r)[0]

cdef inline number_t cbesk(double v, number_t z) nogil:
    cdef npy_cdouble r
    if number_t is double:
        return cbesk_wrap_real(v, z)
    else:
        r = cbesk_wrap(v, (<npy_cdouble*>&z)[0])
        return (<number_t*>&r)[0]


@cython.cdivision(True)
cdef inline double spherical_jn_real(long n, double x) nogil:
    cdef double s0, s1, sn
    cdef int idx

    if zisnan(x):
        return x
    if n < 0:
        sf_error.error("spherical_jn", sf_error.DOMAIN, NULL)
        return nan
    if x == inf or x == -inf:
        return 0
    if x == 0:
        return 0

    if n > 1 and n <= x:
        return sqrt(0.5*pi/x)*cbesj(n + 0.5, x)

    s0 = sin(x)/x
    if n == 0:
        return s0
    s1 = (s0 - cos(x))/x
    if n == 1:
        return s1

    for idx in range(n - 1):
        sn = (2*idx + 3)/x*s1 - s0
        s0 = s1
        s1 = sn
        if zisinf(sn):
            # Overflow occurred already: terminate recurrence.
            return sn

    return sn


@cython.cdivision(True)
cdef inline double complex spherical_jn_complex(long n, double complex z) nogil:

    if npy_isnan(z.real) or npy_isnan(z.imag):
        return z
    if n < 0:
        sf_error.error("spherical_jn", sf_error.DOMAIN, NULL)
        return nan
    if z.real == inf or z.real == -inf:
        # http://dlmf.nist.gov/10.52.E3
        if z.imag == 0:
            return 0
        else:
            return (1+1j)*inf
    if z.real == 0 and z.imag == 0:
        return 0

    return sqrt(0.5*pi)/zsqrt(z)*cbesj(n + 0.5, z)


@cython.cdivision(True)
cdef inline double spherical_yn_real(long n, double x) nogil:
    cdef double s0, s1, sn
    cdef int idx

    if zisnan(x):
        return x
    if n < 0:
        sf_error.error("spherical_yn", sf_error.DOMAIN, NULL)
        return nan
    if x == inf or x == -inf:
        return 0

    s0 = -cos(x)/x
    if n == 0:
        return s0
    s1 = (s0 - sin(x))/x
    if n == 1:
        return s1

    for idx in range(n - 1):
        sn = (2*idx + 3)/x*s1 - s0
        s0 = s1
        s1 = sn
        if zisinf(sn):
            # Overflow occurred already: terminate recurrence.
            return sn

    return sn


@cython.cdivision(True)
cdef inline double complex spherical_yn_complex(long n, double complex z) nogil:

    if npy_isnan(z.real) or npy_isnan(z.imag):
        return z
    if n < 0:
        sf_error.error("spherical_yn", sf_error.DOMAIN, NULL)
        return nan
    if z.real == 0 and z.imag == 0:
        # http://dlmf.nist.gov/10.52.E2
        return nan
    if z.real == inf or z.real == -inf:
        # http://dlmf.nist.gov/10.52.E3
        if z.imag == 0:
            return 0
        else:
            return (1+1j)*inf

    return sqrt(0.5*pi)/zsqrt(z)*cbesy(n + 0.5, z)


@cython.cdivision(True)
cdef inline double spherical_in_real(long n, double z) nogil:

    if zisnan(z):
        return z
    if n < 0:
        sf_error.error("spherical_in", sf_error.DOMAIN, NULL)
        return nan
    if z == 0:
        # http://dlmf.nist.gov/10.52.E1
        return 0
    if zisinf(z):
        # http://dlmf.nist.gov/10.49.E8
        if z == -inf:
            return (-1)**n*inf
        else:
            return inf

    return sqrt(0.5*pi/z)*iv(n + 0.5, z)


@cython.cdivision(True)
cdef inline double complex spherical_in_complex(long n, double complex z) nogil:
    cdef npy_cdouble s

    if zisnan(z):
        return z
    if n < 0:
        sf_error.error("spherical_in", sf_error.DOMAIN, NULL)
        return nan
    if zabs(z) == 0:
        # http://dlmf.nist.gov/10.52.E1
        return 0
    if zisinf(z):
        # http://dlmf.nist.gov/10.52.E5
        if z.imag == 0:
            if z.real == -inf:
                return (-1)**n*inf
            else:
                return inf
        else:
            return z

    s = cbesi_wrap(n + 0.5, (<npy_cdouble*>&z)[0])
    return sqrt(0.5*pi)/zsqrt(z)*(<double complex*>&s)[0]


@cython.cdivision(True)
cdef inline double spherical_kn_real(long n, double z) nogil:

    if zisnan(z):
        return z
    if n < 0:
        sf_error.error("spherical_kn", sf_error.DOMAIN, NULL)
        return nan
    if zisinf(z):
        # http://dlmf.nist.gov/10.52.E6
        if z == inf:
            return 0
        else:
            return -inf

    return sqrt(0.5*pi/z)*cbesk(n + 0.5, z)


@cython.cdivision(True)
cdef inline double complex spherical_kn_complex(long n, double complex z) nogil:

    if zisnan(z):
        return z
    if n < 0:
        sf_error.error("spherical_kn", sf_error.DOMAIN, NULL)
        return nan
    if zisinf(z):
        # http://dlmf.nist.gov/10.52.E6
        if z.imag == 0:
            if z.real == inf:
                return 0
            else:
                return -inf
        else:
            return (1+1j)*inf

    return sqrt(0.5*pi)/zsqrt(z)*cbesk(n + 0.5, z)
