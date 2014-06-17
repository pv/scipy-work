import cython

cimport sf_error

from _complexstuff cimport *

from libc.math cimport sqrt, fabs, pow, M_PI as pi
from libc.stdlib cimport abs, malloc

cdef extern from "lapack_defs.h":
    void c_dstev( char *jobz, int *n, double *d, double *e, double *z, int *ldz, double *work, int *info) nogil

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)

cdef inline double ellip_harmonic( double coord_a, double coord_b, double coord_c, int n,int p, double l, int signn, int signm) nogil:
    """
    Evaluate E^p_n(l)

    - The signs of \mu and \nu are necessary in order to determine the sign of \psi

    Parameters
    ----------
    coord_a, coord_b, coord_c:
    The ellipsoidal coordinate system is defined by three axes a >= b >= c
    - a,b,c can be the semiaxes of the reference ellipsoid
    - a,b can be a list of points and radii so that an ellipse can be estimated
    n: the order

    """
    cdef double h2, k2, l2, alpha, beta, gamma, lamba_romain, pp, psi, t1
    cdef int r, tp, j, size, i, info
    cdef char t

    if coord_a < coord_b:
        sf_error.error("ellip_harm",sf_error.ARG,"b_coord should be greater than c_coord")
        return nan

    if coord_b < coord_c:
        sf_error.error("ellip_harm",sf_error.ARG,"a_coord should be greater than b_coord")
        return nan
   
    r = n/2
    h2 = coord_a * coord_a - coord_b * coord_b
    k2 = coord_a * coord_a - coord_c * coord_c
    l2 = l * l

    if p < r + 1:
        t1, t, tp = 0,'K', p
    elif p < (n-r) + (r +1):
        t1, t, tp = 1,'L', p-(r+1)
    elif p < (n-r) + (n-r) + (r+1):
        t1, t, tp = 2, 'M', p - (n-r) - (r+1)
    elif p < 2*n+1:
        t1, t,tp = 3, 'N', p - (n-r) - (n-r) - (r+1)
    
    if t == 'K':
        size = r+1
        psi = pow(l, n - 2*r)	
    elif t == 'L': 
        size = n-r
        psi = pow(l, 1 - n + 2*r) * signm * sqrt(fabs(l2 - h2))
    elif t == 'M':
        size = n-r	
        psi = pow(l, 1 - n + 2*r) * signn * sqrt(fabs(l2 - k2))
    if t == 'N':
        size = r	
        psi = pow(l,  n - 2*r) * signm * signn * sqrt(fabs((l2 - h2) * (l2 - k2)))

    cdef double *g =  <double *> malloc(sizeof(double) * size)
    cdef double *d =  <double *> malloc(sizeof(double) * size)
    cdef double *f =  <double *> malloc(sizeof(double) * size)
    cdef double *s =  <double *> malloc(sizeof(double) * size)
    cdef double *work =  <double *> malloc(sizeof(double) * (2*size-2))
    cdef double *dd = <double *> malloc(sizeof(double) * size)
    cdef double *m = <double *> malloc(sizeof(double) * (size * size))
    cdef double *eigv = <double *> malloc(sizeof(double) * (size * size))
    
    alpha = h2
    beta = k2 - h2
    gamma= alpha - beta

    if t == 'K':
        for j in range(0, r+1):
           g[j] = (-(2*j + 2) * (2*j + 1) * beta)
           if n%2:
               f[j] = (-alpha * (2 * (r- (j+1)) +2)*(2*((j+1) + r) + 1))
               d[j] = ((2*r + 1) * (2*r + 2) - 4*j*j) * alpha + (2*j + 1) * (2*j + 1) * beta
           else:
               f[j] = (-alpha * (2 * (r - (j+1)) + 2) *(2*(r + (j+1)) - 1))
               d[j] = 2 * r * (2*r + 1) * alpha - 4*j*j*gamma
		
    elif t == 'L':
        for j in range(0, n-r):
           g[j] = (-(2*j + 2) * (2*j + 3) * beta)
           if n%2:
               f[j] = (-alpha * (2 * (r- (j+1)) +2)*(2*((j+1) + r) + 1))
               d[j] = (2*r +1) * (2*r+2) * alpha - (2 *j+1) * (2*j + 1) * gamma
           else:
               f[j] = (-alpha * (2 * (r - (j+1))) *(2*(r*(j+1)) + 1))
               d[j] = (2*r * (2*r +1) - (2*j +1)*(2*j +1)) * alpha + (2*j +2)*(2*j +2)*beta
		
    elif t == 'M':
        for j in range(0, n-r):
           g[j] = (-(2*j + 2) * (2*j + 1) * beta)
           if n%2:
               f[j] = (-alpha * (2 * (r- (j+1)) +2)*(2*((j+1) + r) + 1))
               d[j] = ((2*r +1) *(2*r+2) - (2*j + 1)*(2*j + 1))* alpha + 4*j*j*beta
           else:
               f[j] = (-alpha * (2 * (r - (j+1))) *(2*(r*(j+1)) + 1))
               d[j] = 2*r*(2*r + 1)*(2*r + 2) - (2*j +1) * (2*j + 1)*gamma	
    elif t == 'N':
        for j in range(0, r):
           g[j] = (-(2*j + 2) * (2*j + 3) * beta) 
           if n%2:
               f[j] = (-alpha * (2 * (r- (j+1)) +2)*(2*((j+1) + r) + 3))
               d[j] = (2*r + 1) * (2*r +2) * alpha - (2*j +2)*(2*j + 2)* gamma	
           else:
               f[j] = (-alpha * (2 * (r - (j+1))) *(2*(r*(j+1)) + 1))   
               d[j] = 2*r*(2*r + 1)*alpha - (2*j + 2) * (2*j +2)*alpha + (2*j +1)*(2*j +1) * beta
    

    for i in range(0,size):
        for j in range(0,size):
           if(i==j):
              m[i*size + j] = d[j]
           elif(i == j - 1):
              m[i*size + j] = g[i]
           elif(i == j + 1):
              m[i*size + j] = f[i]
           else:
              m[i*size + j] = 0

    for i in range(0,size):
        if(i==0):
           s[i] = 1
        else:
           s[i] = sqrt(g[i-1]/f[i-1])*s[i-1]
            
    for i in range(0,size):
        for j in range(0,size):
           if(i==j-1):
              dd[i] = m[i*size + j] * s[i] / s[j]

    

    c_dstev("V", &size, <double *> d, dd, eigv, &size, work, &info)              	 
    if info != 0: 
        sf_error.error("ellip_harm",sf_error.ARG,"illegal")
        return nan   

    for i in range(0,size):
        s[i] = 1 / s[i]


    lambda_romain = 1.0 - <double> l2 / <double> h2
    d = eigv + tp*size

    for i in range(0,size):
        d[i] = d[i]*s[i]


    for i in range(0,size):
        d[i] = d[i]/ (d[size - 1]/pow(-h2, size - 1))

    pp = d[size - 1]

    for j in range(size - 2, -1, -1):
        pp = pp * lambda_romain + d[j]

    return psi *pp 
 
