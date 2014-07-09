import cython

cimport sf_error

from _complexstuff cimport *

from libc.math cimport sqrt, fabs, pow, M_PI as pi
from libc.stdlib cimport abs, malloc, free

cdef extern from "lapack_defs.h":
    void c_dstevr(char *jobz, char *range, int *n, double *d, double *e, double *vl, double *vu, int *il, int *iu, double *abstol, int *m, double *w, double *z, int *ldz, int *isuppz, double *work, int *lwork, int *iwork, int *liwork, int *info) nogil

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline double ellip_norm(double h2, double k2, int n, int p) nogil:
   
    cdef double s2, alpha, beta, gamma, lamba_romain, pp, psi, t1, tol, vl, vu, res
    cdef int r, tp, j, size, i, info, lwork, liwork, c, iu, m, min, lu, ll, buf, buf1
    cdef char t

    r = n/2
    alpha = h2
    beta = k2 - h2
    gamma = alpha - beta

    if p < 1 or p > 2*n + 1:
        sf_error.error("ellip_harm", sf_error.ARG, "invalid value for p")
        return nan

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
	
    elif t == 'L': 
        size = n - r

    elif t == 'M':
        size = n - r

    if t == 'N':
        size = r

    m = size - 1
    lwork = 60*size
    liwork = 30*size
    tol = 0.0
    vl = 0
    vu = 0

    cdef void *buffer =  malloc((sizeof(double)*(7*size + lwork + 3*(n+1)+1+(2*m + 1)+ 2*n +7)) + (sizeof(int)*(2*size + liwork)))
    if not buffer:
       return nan 

    cdef double *g = <double *>buffer
    cdef double *d = g + size
    cdef double *f = d + size
    cdef double *ss =  f + size
    cdef double *w =  ss + size
    cdef double *dd = w + size
    cdef double *eigv = dd + size
    cdef double *dnorm = eigv + size
    cdef double *cnorm = dnorm + 2*m + 1
    cdef double *tou = cnorm + n + 1
    cdef double *tou1 = tou + n + 1
    cdef double *y = tou1 + n + 2
    cdef double *yy = y + n + 3 
    cdef double *work = yy + n + 4

    cdef int *iwork = <int *>(work + lwork)
    cdef int *isuppz = iwork + liwork
    

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
        if i == 0:
           ss[i] = 1
        else:
           ss[i] = sqrt(g[i - 1]/f[i - 1])*ss[i - 1]
            
    for i in range(0, size-1):
        dd[i] = g[i]*ss[i]/ss[i+1]

    c_dstevr("V", "I", &size, <double *>d, <double *>dd, &vl, &vu, &tp, &tp, &tol, &c, <double *>w, <double *>eigv, &size, <int *>isuppz, <double *>work, &lwork, <int *>iwork, &liwork, &info)
              	 
    if info != 0: 
        sf_error.error("ellip_harm", sf_error.ARG, "illegal")
        free(buffer)
        return nan   

    for i in range(0, size):
        eigv[i] /= ss[i]
#B_i is above
    for i in range(0, size):
        eigv[i] = eigv[i]/(eigv[size - 1]/pow(-h2, size - 1))

    for i in range(0, 2*m + 1):
        dnorm[i] = 0

        if i < m:
            lu = i
        else:
            lu = m

        if i - m > 0:
            ll = i - m
        else:
            ll = 0

        for j in range(ll, lu + 1):
            dnorm[i] += eigv[j]*eigv[i - j]

    if t == 'K':
        if n%2:
            cnorm[0] = h2*dnorm[0]
            for j in range(1, n):
                cnorm[j] = h2*(dnorm[j] - dnorm[j-1])
            cnorm[n] = -h2*dnorm[n-1]
        else:
            for j in range(0, n+1):
                cnorm[j] = dnorm[j]

    elif t == 'L':
        if n%2:
            cnorm[0] = 0
            for j in range(1, n + 1):
                cnorm[j] = h2*dnorm[j-1]
        else:
            cnorm[0] = 0
            cnorm[1] = h2*h2*dnorm[0]
            for j in range(2, n):
                cnorm[j] = h2*h2*(dnorm[j-1] - dnorm[j-2])
            cnorm[n] = -h2*h2*dnorm[n-2]

    elif t =='M':
        if n%2:
            cnorm[0] = (k2 - h2)*dnorm[0]
            for j in range(1, n):
                cnorm[j] = (k2 - h2)*dnorm[j] + h2*dnorm[j-1]
            cnorm[n] = h2*dnorm[n-1]
        else:
            cnorm[0] = h2*(k2 - h2)*dnorm[0]
            cnorm[1] = h2*(k2 - h2)*dnorm[1] + h2*(2*h2 - k2)*dnorm[0]
            for j in range(2, n-1):
                cnorm[j] = h2*(k2 - h2)*dnorm[j] + h2*(2*h2 - k2)*dnorm[j-1] - h2*h2*dnorm[j-2]
            cnorm[n-1] = h2*(2*h2 - k2)*dnorm[n-2] - h2*h2*dnorm[n-2]
            cnorm[n] = -h2*h2*dnorm[n-2]

    elif t == 'N':
        if n%2:
            cnorm[0] = 0
            cnorm[1] = h2*h2*(k2 - h2)*dnorm[0]
            cnorm[2] = h2*h2*((k2 - h2)*dnorm[1] + (2*h2 - k2)*dnorm[0])
            for j in range(3, n-1):
                cnorm[j] = h2*h2*((k2 - h2)*dnorm[j-1] + (2*h2 - k2)*dnorm[j-2] - h2*dnorm[j-3])
            cnorm[n-1] = h2*h2*((2*h2 - k2)*dnorm[n-3] - h2*dnorm[n-4])
            cnorm[n] = -h2*h2*h2*dnorm[n-3]
        else:
            cnorm[0] = 0
            cnorm[1] = h2*(k2 - h2)*dnorm[0]
            for j in range(2, n):
                cnorm[j] = h2*((k2 - h2)*dnorm[j-1] + h2*dnorm[j-1])
            cnorm[n] = h2*h2*dnorm[n-2]

    for j in range(0, n+1):
        tou[j] = -0.5*cnorm[j]
    tou1[0] = -0.5*h2*cnorm[0]
    for j in range(0, n+1):
        tou1[j] = -0.5*h2*(cnorm[j]-cnorm[j-1])
    tou1[n+1] = 0.5*h2*cnorm[n]

    y[n+2] = 0
    y[n+1] = 0
    for j in range(n, -1, -1):
        y[j] = (2*j/(2*j + 1))*(2 - k2/h2)*y[j+1] + ((2*j + 1)/(2*j + 3))*(k2/h2 - 1)*y[j+2] + tou[j]

    yy[n+3] = 0
    yy[n+2] = 0
    for j in range(n+1, -1, -1):
        yy[j] = (2*j/(2*j + 1))*(2 - k2/h2)*yy[j+1] + ((2*j + 1)/(2*j + 3))*(k2/h2 - 1)*yy[j+2] + tou1[j]
    res = yy[0]*y[1] - y[0]*yy[1]

    free(buffer)
    return res*16*pi/h2
 
