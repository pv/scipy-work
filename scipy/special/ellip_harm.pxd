import cython

cimport sf_error

from _complexstuff cimport *

from libc.math cimport sqrt, fabs, pow, M_PI as pi
from libc.stdlib cimport abs, malloc

cdef extern from "lapack_defs.h":
    void c_dstevr(char *jobz, char *range, int *n, double *d, double *e, double *vl, double *vu, int *il, int *iu, double *abstol, int *m, double *w, double *z, int *ldz, int *isuppz, double *work, int *lwork, int *iwork, int *liwork, int *info) nogil

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)

cdef inline double ellip_harmonic(double coord_a, double coord_b, double coord_c, int n,int p, double l, double signm, double signn) nogil:
    """
    Evaluate E^p_n(l)

    Parameters
    ----------
    coord_a, coord_b, coord_c:
    The ellipsoidal coordinate system is defined by three axes a >= b >= c
    - a,b,c can be the semiaxes of the reference ellipsoid
    - a,b can be a list of points and radii so that an ellipse can be estimated

    n: degree
    p: order, can range between [1,2n+1]
    signm: Specifies the sign of \mu
    signn: Specifies the sign of \nu
    - The signs of \mu and \nu are necessary in order to determine the sign of \psi
     
    Returns
    -------
    E^p_n(l) : double

    Notes
    -----
    Uses LAPACK subroutine DSTEVR

    """
    cdef double h2, k2, l2, alpha, beta, gamma, lamba_romain, pp, psi, t1, tol, vl, vu
    cdef int r, tp, j, size, i, info, lwork, liwork, c, iu
    cdef char t

    if coord_a < coord_b:
        sf_error.error("ellip_harm",sf_error.ARG,"b_coord should be greater than c_coord")
        return nan

    if coord_b < coord_c:
        sf_error.error("ellip_harm",sf_error.ARG,"a_coord should be greater than b_coord")
        return nan

    if p < 1 or p > 2*n + 1:
        sf_error.error("ellip_harm",sf_error.ARG,"Invalid values of p for given n")
        return nan
   
    h2 = coord_a*coord_a - coord_b*coord_b
    k2 = coord_a*coord_a - coord_c*coord_c
    signm = signm/fabs(signm)
    signn = signn/fabs(signn)
    r = n/2
    l2 = l*l
    alpha = h2
    beta = k2 - h2
    gamma = alpha - beta

    if p - 1 < r + 1:
        t, tp = 'K', p
    elif p - 1 < (n - r) + (r + 1):
        t, tp = 'L', p - (r + 1)
    elif p - 1 < (n - r) + (n - r) + (r + 1):
        t, tp = 'M', p - (n - r) - (r + 1)
    elif p - 1 < 2*n + 1:
        t, tp = 'N', p - (n - r) - (n - r) - (r + 1)
    
    if t == 'K':
        size = r + 1
        psi = pow(l, n - 2*r)	
    elif t == 'L': 
        size = n - r
        psi = pow(l, 1 - n + 2*r)*signm*sqrt(fabs(l2 - h2))
    elif t == 'M':
        size = n - r	
        psi = pow(l, 1 - n + 2*r)*signn*sqrt(fabs(l2 - k2))
    if t == 'N':
        size = r	
        psi = pow(l,  n - 2*r)*signm*signn*sqrt(fabs((l2 - h2)*(l2 - k2)))

    lwork = 60*size
    liwork = 30*size
    tol = 0.0
    vl = 0
    vu = 0

    cdef int *iwork = <int *>malloc(sizeof(int)*liwork)
    cdef int *isuppz = <int *>malloc(sizeof(int)*2*size)

    cdef double *g =  <double *>malloc(sizeof(double)*size)
    cdef double *d =  <double *>malloc(sizeof(double)*size)
    cdef double *f =  <double *>malloc(sizeof(double)*size)
    cdef double *s =  <double *>malloc(sizeof(double)*size)
    cdef double *w = <double *>malloc(sizeof(double)*size)
    cdef double *m = <double *>malloc(sizeof(double)*(size*size))
    cdef double *dd = <double *>malloc(sizeof(double)*(size - 1))
    cdef double *eigv = <double *>malloc(sizeof(double)*size) 
    cdef double *work = <double *>malloc(sizeof(double)*lwork)

    if t == 'K':
        for j in range(0, r + 1):
           g[j] = (-(2*j + 2)*(2*j + 1)*beta)
           if n%2:
               f[j] = (-alpha*(2*(r- (j + 1)) + 2)*(2*((j + 1) + r) + 1))
               d[j] = ((2*r + 1)*(2*r + 2) - 4*j*j)*alpha + (2*j + 1)*(2*j + 1)*beta
           else:
               f[j] = (-alpha*(2*(r - (j + 1)) + 2)*(2*(r + (j + 1)) - 1))
               d[j] = 2*r*(2*r + 1)*alpha - 4*j*j*gamma
		
    elif t == 'L':
        for j in range(0, n - r):
           g[j] = (-(2*j + 2)*(2*j + 3)*beta)
           if n%2:
               f[j] = (-alpha*(2*(r- (j + 1)) + 2)*(2*((j + 1) + r) + 1))
               d[j] = (2*r + 1)*(2*r + 2)*alpha - (2*j + 1)*(2*j + 1)*gamma
           else:
               f[j] = (-alpha*(2*(r - (j + 1)))*(2*(r*(j + 1)) + 1))
               d[j] = (2*r*(2*r + 1) - (2*j + 1)*(2*j + 1))*alpha + (2*j + 2)*(2*j + 2)*beta
		
    elif t == 'M':
        for j in range(0, n - r):
           g[j] = (-(2*j + 2)*(2*j + 1)*beta)
           if n%2:
               f[j] = (-alpha*(2*(r - (j + 1)) + 2)*(2*((j + 1) + r) + 1))
               d[j] = ((2*r + 1)*(2*r + 2) - (2*j + 1)*(2*j + 1))*alpha + 4*j*j*beta
           else:
               f[j] = (-alpha*(2*(r - (j + 1)))*(2*(r*(j + 1)) + 1))
               d[j] = 2*r*(2*r + 1)*(2*r + 2) - (2*j + 1)*(2*j + 1)*gamma	

    elif t == 'N':
        for j in range(0, r):
           g[j] = (-(2*j + 2)*(2*j + 3)*beta) 
           if n%2:
               f[j] = (-alpha*(2*(r- (j + 1)) + 2)*(2*((j + 1) + r) + 3))
               d[j] = (2*r + 1)*(2*r + 2)*alpha - (2*j + 2)*(2*j + 2)*gamma	
           else:
               f[j] = (-alpha*(2*(r - (j + 1)))*(2*(r*(j + 1)) + 1))   
               d[j] = 2*r*(2*r + 1)*alpha - (2*j + 2)*(2*j +2)*alpha + (2*j + 1)*(2*j + 1)*beta
    
    for i in range(0, size):
        for j in range(0, size):
           if(i == j):
              m[i*size + j] = d[j]
           elif(i == j - 1):
              m[i*size + j] = g[i]
           elif(i == j + 1):
              m[i*size + j] = f[i]
           else:
              m[i*size + j] = 0

    for i in range(0, size):
        if(i == 0):
           s[i] = 1
        else:
           s[i] = sqrt(g[i - 1]/f[i - 1])*s[i - 1]
            
    for i in range(0, size):
        for j in range(0, size):
           if(i == j - 1):
              dd[i] = m[i*size + j]*s[i]/s[j]

    c_dstevr("V", "I", &size, <double *>d, <double *>dd, &vl, &vu, &tp, &tp, &tol, &c, <double *>w, <double *>eigv, &size, <int *>isuppz, <double *>work, &lwork, <int *>iwork, &liwork, &info)
              	 
    if info != 0: 
        sf_error.error("ellip_harm", sf_error.ARG, "illegal")
        return nan   

    for i in range(0, size):
        s[i] = 1/s[i]
 
    lambda_romain = 1.0 - <double>l2/<double>h2

    for i in range(0, size):
        eigv[i] = eigv[i]*s[i]

    for i in range(0, size):
        eigv[i] = eigv[i]/(eigv[size - 1]/pow(-h2, size - 1))

    pp = eigv[size - 1]

    for j in range(size - 2, -1, -1):
        pp = pp*lambda_romain + eigv[j]

    return psi*pp 
 
