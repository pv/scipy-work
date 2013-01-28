"""
This is an internal module only, containing the inner loops
for lmfit.py.
"""
cdef extern from "alloca.h":
    void *alloca(int) nogil

from libc.math cimport sqrt, fabs
from libc.string cimport memset

ctypedef fused real_t:
    float
    double

cimport numpy
cimport cython

@cython.boundscheck(False)
@cython.cdivision(True)
def qrsolv(real_t[:,:] s, real_t[:] diag):
    """
    Inner-loop of lmfit.Fit._qrsolv. This is an internal function only.

    This function eliminates the diagonal matrix diag using a Givens rotation,
    in order to meet the goals of lmfit.Fit._qrsolv.
    """
    cdef unsigned int N
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l
    cdef real_t sin
    cdef real_t cos
    cdef real_t tan
    cdef real_t cotan
    cdef real_t tmp
    cdef real_t *ta

    with nogil:
        N = diag.shape[0]
        ta = <real_t *> alloca((N + 1) * sizeof(real_t))

        for 0 <= j < N:
            if diag[j] != 0:
                memset(ta, 0, (N + 1) * sizeof(real_t))
                ta[j] = diag[j]
                for j <= k < N:
                    if ta[k] != 0:
                        if fabs(s[k, k]) > fabs(ta[k]):
                            tan = ta[k] / s[k, k]
                            cos = 1 / sqrt(1 + tan * tan)
                            sin = cos * tan
                        else:
                            cotan = s[k, k] / ta[k]
                            sin = 1 / sqrt(1 + cotan * cotan)
                            cos = sin * cotan
                        for k <= l <= N:
                            tmp = s[k, l]
                            s[k, l] = cos * tmp + sin * ta[l]
                            ta[l] = -sin * tmp + cos * ta[l]
