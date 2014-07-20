# Copyright (c) 2012, Jaydeep P. Bardhan
# Copyright (c) 2012, Matthew G. Knepley
# Copyright (c) 2014, Janani Padmanabhan
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

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
cdef inline double* lame_coefficients(double h2, double k2, int n, int p, void **bufferp) nogil:
   
    cdef double s2, alpha, beta, gamma, lamba_romain, pp, psi, t1, tol, vl, vu
    cdef int r, tp, j, size, i, info, lwork, liwork, c, iu
    cdef char t

    r = n/2
    alpha = h2
    beta = k2 - h2
    gamma = alpha - beta

    if p - 1 < r + 1:
        t, tp, size = 'K', p, r + 1
    elif p - 1 < (n - r) + (r + 1):
        t, tp, size = 'L', p - (r + 1), n - r
    elif p - 1 < (n - r) + (n - r) + (r + 1):
        t, tp, size = 'M', p - (n - r) - (r + 1), n - r
    elif p - 1 < 2*n + 1:
        t, tp, size = 'N', p - (n - r) - (n - r) - (r + 1), r
    

    lwork = 60*size
    liwork = 30*size
    tol = 0.0
    vl = 0
    vu = 0

    cdef void *buffer =  malloc((sizeof(double)*(7*size + lwork)) + (sizeof(int)*(2*size + liwork)))
    bufferp[0] = buffer
    if not buffer:
        sf_error.error("ellip_harm", sf_error.NO_RESULT, "failed to allocate memory")
        return NULL 

    cdef double *g = <double *>buffer
    cdef double *d = g + size
    cdef double *f = d + size
    cdef double *ss =  f + size
    cdef double *w =  ss + size
    cdef double *dd = w + size
    cdef double *eigv = dd + size 
    cdef double *work = eigv + size

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
        sf_error.error("ellip_harm", sf_error.NO_RESULT, "failed to allocate memory")
        return NULL   

    for i in range(0, size):
        eigv[i] /= ss[i]

    for i in range(0, size):
        eigv[i] = eigv[i]/(eigv[size - 1]/pow(-h2, size - 1))
    return eigv

cdef inline double ellip_harmonic(double h2, double k2, int n, int p, double s, double signm, double signn) nogil:
    cdef int size, tp, r, j
    cdef double s2, pp, lambda_romain, psi

    signm = signm/fabs(signm)
    signn = signn/fabs(signn)

    if p < 1 or p > 2*n + 1:
        sf_error.error("ellip_harm", sf_error.ARG, "invalid value for p")
        return nan

    if fabs(signm) != 1 or fabs(signn) != 1:
        sf_error.error("ellip_harm", sf_error.ARG, "invalid signm or signn")
        return nan

    s2 = s*s
    r = n/2
    if p - 1 < r + 1:
        size, psi = r + 1, pow(s, n - 2*r) 
    elif p - 1 < (n - r) + (r + 1):
        size, psi = n - r, pow(s, 1 - n + 2*r)*signm*sqrt(fabs(s2 - h2))
    elif p - 1 < (n - r) + (n - r) + (r + 1):
        size, psi = n - r, pow(s, 1 - n + 2*r)*signn*sqrt(fabs(s2 - k2))
    elif p - 1 < 2*n + 1:
        size, psi = r, pow(s,  n - 2*r)*signm*signn*sqrt(fabs((s2 - h2)*(s2 - k2)))
    

    cdef double *eigv
    cdef void *bufferp
    eigv = lame_coefficients(h2, k2, n, p, &bufferp)
    if not eigv:
        free(bufferp)
        return nan
    lambda_romain = 1.0 - <double>s2/<double>h2
    pp = eigv[size - 1]

    for j in range(size - 2, -1, -1):
        pp = pp*lambda_romain + eigv[j]
    pp = pp*psi
    free(bufferp)
    return pp 


cdef inline double ellip_norm(double h2, double k2, int n, int p) nogil:    
    cdef int r, tp, size, i , lu, ll, j
    cdef double res
    cdef char t

    if p < 1 or p > 2*n + 1:
        sf_error.error("ellip_harm", sf_error.ARG, "invalid value for p")
        return nan  
  
    r = n/2

    if p - 1 < r + 1:
        t, size = 'K', r + 1
    elif p - 1 < (n - r) + (r + 1):
        t, size = 'L', n - r
    elif p - 1 < (n - r) + (n - r) + (r + 1):
        t, size = 'M', n - r
    elif p - 1 < 2*n + 1:
        t, size = 'N', r

    cdef void *buffer =  malloc((sizeof(double)*(3*(n+1)+1+(2*size - 1)+ 2*n +7)))
    cdef double *eigv
    cdef double *dnorm = <double *> buffer
    cdef double *cnorm = dnorm + 2*size - 1
    cdef double *tou = cnorm + n + 1
    cdef double *tou1 = tou + n + 1
    cdef double *y = tou1 + n + 2
    cdef double *yy = y + n + 3 
    cdef double *work = yy + n + 4
    cdef void *bufferp
    eigv = lame_coefficients(h2, k2, n, p, &bufferp)
    if not eigv:
        
        return nan
    for i in range(0, 2*size - 1):
        dnorm[i] = 0

        if i < size - 1:
            lu = i
        else:
            lu = size - 1

        if i - (size -1) > 0:
            ll = i - (size - 1)
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
            if n > 2:
                cnorm[n-1] = h2*(2*h2 - k2)*dnorm[n-2] - h2*h2*dnorm[n-3]
            if n == 2:
                cnorm[n-1] = h2*(2*h2 - k2)*dnorm[n-2]
            cnorm[n] = -h2*h2*dnorm[n-2]

    elif t == 'N':
        if n%2:
            cnorm[0] = 0
            cnorm[1] = h2*h2*(k2 - h2)*dnorm[0]
            cnorm[2] = h2*h2*((k2 - h2)*dnorm[1] + (2*h2 - k2)*dnorm[0]) 
            for j in range(3, n-1):
                cnorm[j] = h2*h2*((k2 - h2)*dnorm[j-1] + (2*h2 - k2)*dnorm[j-2] - h2*dnorm[j-3])
            if n > 3:
                cnorm[n-1] = h2*h2*((2*h2 - k2)*dnorm[n-3] - h2*dnorm[n-4]) 
            cnorm[n] = -h2*h2*h2*dnorm[n-3]
        else:
            cnorm[0] = 0
            cnorm[1] = h2*(k2 - h2)*dnorm[0]
            for j in range(2, n):
                cnorm[j] = h2*((k2 - h2)*dnorm[j-1] + h2*dnorm[j-2]) 
            cnorm[n] = h2*h2*dnorm[n-2]

    for j in range(0, n+1):
        tou[j] = -0.5*cnorm[j]
    tou1[0] = -0.5*h2*cnorm[0]
    for j in range(1, n+1):
        tou1[j] = -0.5*h2*(cnorm[j]-cnorm[j-1])
    tou1[n+1] = 0.5*h2*cnorm[n]

    with gil:
        import sys
        sys.stdout.write("G = {")
        for j in range(n+1):
            sys.stdout.write("%.21g, " % (tou[j],))
        sys.stdout.write("0\n")
        sys.stdout.write("};\n")
        sys.stdout.write("Gt = {")
        for j in range(n+2):
            sys.stdout.write("%.21g, " % (tou1[j],))
        sys.stdout.write("};\n")

    y[n+2] = 0
    y[n+1] = 0
    for j in range(n, -1, -1):
        y[j] = (2*j/(2*j + 1.0))*(2 - k2/h2)*y[j+1] + ((2*j + 1)/(2*j + 3.0))*(k2/h2 - 1)*y[j+2] + tou[j]

    yy[n+3] = 0
    yy[n+2] = 0
    for j in range(n+1, -1, -1):
        yy[j] = (2*j/(2*j + 1.0))*(2 - k2/h2)*yy[j+1] + ((2*j + 1)/(2*j + 3.0))*(k2/h2 - 1)*yy[j+2] + tou1[j]
    res = yy[0]*y[1] - y[0]*yy[1]

    cdef double detX = 0.0, bprod = 0.0, maxterm = 0.0, term = 0.0
    cdef int k
    for k in range(n+1):
        bprod = 1.0
        for j in range(k):
            bprod *= -((2*j + 1)/(2*j + 3.0)) * (k2/h2 - 1)
        term = (yy[k+1]*tou[k] - y[k+1]*tou1[k]) * bprod
        with gil:
            print(term, bprod)
        detX += term
        if fabs(term) > maxterm:
            maxterm = fabs(term)
        #detX = -((2*j + 1)/(2*j + 3.0)) * (k2/h2 - 1) * detX + yy[j+1]*tou[j] - y[j+1]*tou1[j]
        #with gil:
        #    try:
        #        print(detX / (y[j+1]*tou1[j]), (yy[j+1]*tou[j] - y[j+1]*tou1[j])/ (y[j+1]*tou1[j]))
        #    except:
        #        pass

    with gil:
        print("->", -detX*16*pi/h2)
        print("precloss", maxterm/detX)
    res = -detX

    with gil:
        print yy[0]*y[1], y[0]*yy[1]
    if t == 'L' or t == 'N':
        res *= -1
    free(buffer)
    free(bufferp)
    res = res*16*pi/h2
    return res    


