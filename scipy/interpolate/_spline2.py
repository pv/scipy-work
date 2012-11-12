import math
import numpy as np
from numpy import dot, array, ones

from scipy import special
import _fitpack

__all__ = ['spline', 'spleval', 'splmake', 'spltopp', 'ppform']

class ppform(object):
    """The ppform of the piecewise polynomials is given in terms of coefficients
    and breaks.  The polynomial in the ith interval is
    x_{i} <= x < x_{i+1}

    S_i = sum(coefs[m,i]*(x-breaks[i])^(k-m), m=0..k)
    where k is the degree of the polynomial.
    """
    def __init__(self, coeffs, breaks, fill=0.0, sort=False):
        self.coeffs = np.asarray(coeffs)
        if sort:
            self.breaks = np.sort(breaks)
        else:
            self.breaks = np.asarray(breaks)
        self.K = self.coeffs.shape[0]
        self.fill = fill
        self.a = self.breaks[0]
        self.b = self.breaks[-1]

    def __call__(self, xnew):
        saveshape = np.shape(xnew)
        xnew = np.ravel(xnew)
        res = np.empty_like(xnew)
        mask = (xnew >= self.a) & (xnew <= self.b)
        res[~mask] = self.fill
        xx = xnew.compress(mask)
        indxs = np.searchsorted(self.breaks, xx)-1
        indxs = indxs.clip(0,len(self.breaks))
        pp = self.coeffs
        diff = xx - self.breaks.take(indxs)
        V = np.vander(diff,N=self.K)
        # values = np.diag(dot(V,pp[:,indxs]))
        values = array([dot(V[k,:],pp[:,indxs[k]]) for k in xrange(len(xx))])
        res[mask] = values
        res.shape = saveshape
        return res

    def fromspline(cls, xk, cvals, order, fill=0.0):
        N = len(xk)-1
        sivals = np.empty((order+1,N), dtype=float)
        for m in xrange(order,-1,-1):
            fact = special.gamma(m+1)
            res = _fitpack._bspleval(xk[:-1], xk, cvals, order, m)
            res /= fact
            sivals[order-m,:] = res
        return cls(sivals, xk, fill=fill)
    fromspline = classmethod(fromspline)


def _dot0(a, b):
    """Similar to numpy.dot, but sum over last axis of a and 1st axis of b"""
    if b.ndim <= 2:
        return dot(a, b)
    else:
        axes = range(b.ndim)
        axes.insert(-1, 0)
        axes.pop(0)
        return dot(a, b.transpose(axes))

def _find_smoothest(xk, yk, order, conds=None, B=None):
    # construct Bmatrix, and Jmatrix
    # e = J*c
    # minimize norm(e,2) given B*c=yk
    # if desired B can be given
    # conds is ignored
    N = len(xk)-1
    K = order
    if B is None:
        B = _fitpack._bsplmat(order, xk)
    J = _fitpack._bspldismat(order, xk)
    u,s,vh = np.dual.svd(B)
    ind = K-1
    V2 = vh[-ind:,:].T
    V1 = vh[:-ind,:].T
    A = dot(J.T,J)
    tmp = dot(V2.T,A)
    Q = dot(tmp,V2)
    p = np.dual.solve(Q,tmp)
    tmp = dot(V2,p)
    tmp = np.eye(N+K) - tmp
    tmp = dot(tmp,V1)
    tmp = dot(tmp,np.diag(1.0/s))
    tmp = dot(tmp,u.T)
    return _dot0(tmp, yk)

def _setdiag(a, k, v):
    if not a.ndim == 2:
        raise ValueError("Input array should be 2-D.")
    M,N = a.shape
    if k > 0:
        start = k
        num = N-k
    else:
        num = M+k
        start = abs(k)*N
    end = start + num*(N+1)-1
    a.flat[start:end:(N+1)] = v

# Return the spline that minimizes the dis-continuity of the
# "order-th" derivative; for order >= 2.

def _find_smoothest2(xk, yk):
    N = len(xk)-1
    Np1 = N+1
    # find pseudo-inverse of B directly.
    Bd = np.empty((Np1,N))
    for k in range(-N,N):
        if (k<0):
            l = np.arange(-k,Np1)
            v = (l+k+1)
            if ((k+1) % 2):
                v = -v
        else:
            l = np.arange(k,N)
            v = N-l
            if ((k % 2)):
                v = -v
        _setdiag(Bd,k,v)
    Bd /= (Np1)
    V2 = np.ones((Np1,))
    V2[1::2] = -1
    V2 /= math.sqrt(Np1)
    dk = np.diff(xk)
    b = 2*np.diff(yk, axis=0)/dk
    J = np.zeros((N-1,N+1))
    idk = 1.0/dk
    _setdiag(J,0,idk[:-1])
    _setdiag(J,1,-idk[1:]-idk[:-1])
    _setdiag(J,2,idk[1:])
    A = dot(J.T,J)
    val = dot(V2,dot(A,V2))
    res1 = dot(np.outer(V2,V2)/val,A)
    mk = dot(np.eye(Np1)-res1, _dot0(Bd,b))
    return mk

def _get_spline2_Bb(xk, yk, kind, conds):
    Np1 = len(xk)
    dk = xk[1:]-xk[:-1]
    if kind == 'not-a-knot':
        # use banded-solver
        nlu = (1,1)
        B = ones((3,Np1))
        alpha = 2*(yk[1:]-yk[:-1])/dk
        zrs = np.zeros((1,)+yk.shape[1:])
        row = (Np1-1)//2
        b = np.concatenate((alpha[:row],zrs,alpha[row:]),axis=0)
        B[0,row+2:] = 0
        B[2,:(row-1)] = 0
        B[0,row+1] = dk[row-1]
        B[1,row] = -dk[row]-dk[row-1]
        B[2,row-1] = dk[row]
        return B, b, None, nlu
    else:
        raise NotImplementedError("quadratic %s is not available" % kind)

def _get_spline3_Bb(xk, yk, kind, conds):
    # internal function to compute different tri-diagonal system
    # depending on the kind of spline requested.
    # conds is only used for 'second' and 'first'
    Np1 = len(xk)
    if kind in ['natural', 'second']:
        if kind == 'natural':
            m0, mN = 0.0, 0.0
        else:
            m0, mN = conds

        # the matrix to invert is (N-1,N-1)
        # use banded solver
        beta = 2*(xk[2:]-xk[:-2])
        alpha = xk[1:]-xk[:-1]
        nlu = (1,1)
        B = np.empty((3,Np1-2))
        B[0,1:] = alpha[2:]
        B[1,:] = beta
        B[2,:-1] = alpha[1:-1]
        dyk = yk[1:]-yk[:-1]
        b = (dyk[1:]/alpha[1:] - dyk[:-1]/alpha[:-1])
        b *= 6
        b[0] -= m0
        b[-1] -= mN

        def append_func(mk):
            # put m0 and mN into the correct shape for
            #  concatenation
            ma = array(m0,copy=0,ndmin=yk.ndim)
            mb = array(mN,copy=0,ndmin=yk.ndim)
            if ma.shape[1:] != yk.shape[1:]:
                ma = ma*(ones(yk.shape[1:])[np.newaxis,...])
            if mb.shape[1:] != yk.shape[1:]:
                mb = mb*(ones(yk.shape[1:])[np.newaxis,...])
            mk = np.concatenate((ma,mk),axis=0)
            mk = np.concatenate((mk,mb),axis=0)
            return mk

        return B, b, append_func, nlu


    elif kind in ['clamped', 'endslope', 'first', 'not-a-knot', 'runout',
                  'parabolic']:
        if kind == 'endslope':
            # match slope of lagrange interpolating polynomial of
            # order 3 at end-points.
            x0,x1,x2,x3 = xk[:4]
            sl_0 = (1./(x0-x1)+1./(x0-x2)+1./(x0-x3))*yk[0]
            sl_0 += (x0-x2)*(x0-x3)/((x1-x0)*(x1-x2)*(x1-x3))*yk[1]
            sl_0 += (x0-x1)*(x0-x3)/((x2-x0)*(x2-x1)*(x3-x2))*yk[2]
            sl_0 += (x0-x1)*(x0-x2)/((x3-x0)*(x3-x1)*(x3-x2))*yk[3]

            xN3,xN2,xN1,xN0 = xk[-4:]
            sl_N = (1./(xN0-xN1)+1./(xN0-xN2)+1./(xN0-xN3))*yk[-1]
            sl_N += (xN0-xN2)*(xN0-xN3)/((xN1-xN0)*(xN1-xN2)*(xN1-xN3))*yk[-2]
            sl_N += (xN0-xN1)*(xN0-xN3)/((xN2-xN0)*(xN2-xN1)*(xN3-xN2))*yk[-3]
            sl_N += (xN0-xN1)*(xN0-xN2)/((xN3-xN0)*(xN3-xN1)*(xN3-xN2))*yk[-4]
        elif kind == 'clamped':
            sl_0, sl_N = 0.0, 0.0
        elif kind == 'first':
            sl_0, sl_N = conds

        # Now set up the (N+1)x(N+1) system of equations
        beta = np.r_[0,2*(xk[2:]-xk[:-2]),0]
        alpha = xk[1:]-xk[:-1]
        gamma = np.r_[0,alpha[1:]]
        B = np.diag(alpha,k=-1) + np.diag(beta) + np.diag(gamma,k=1)
        d1 = alpha[0]
        dN = alpha[-1]
        if kind == 'not-a-knot':
            d2 = alpha[1]
            dN1 = alpha[-2]
            B[0,:3] = [d2,-d1-d2,d1]
            B[-1,-3:] = [dN,-dN1-dN,dN1]
        elif kind == 'runout':
            B[0,:3] = [1,-2,1]
            B[-1,-3:] = [1,-2,1]
        elif kind == 'parabolic':
            B[0,:2] = [1,-1]
            B[-1,-2:] = [-1,1]
        elif kind == 'periodic':
            raise NotImplementedError
        elif kind == 'symmetric':
            raise NotImplementedError
        else:
            B[0,:2] = [2*d1,d1]
            B[-1,-2:] = [dN,2*dN]

        # Set up RHS (b)
        b = np.empty((Np1,)+yk.shape[1:])
        dyk = (yk[1:]-yk[:-1])*1.0
        if kind in ['not-a-knot', 'runout', 'parabolic']:
            b[0] = b[-1] = 0.0
        elif kind == 'periodic':
            raise NotImplementedError
        elif kind == 'symmetric':
            raise NotImplementedError
        else:
            b[0] = (dyk[0]/d1 - sl_0)
            b[-1] = -(dyk[-1]/dN - sl_N)
        b[1:-1,...] = (dyk[1:]/alpha[1:]-dyk[:-1]/alpha[:-1])
        b *= 6.0
        return B, b, None, None
    else:
        raise ValueError("%s not supported" % kind)

# conds is a tuple of an array and a vector
#  giving the left-hand and the right-hand side
#  of the additional equations to add to B
def _find_user(xk, yk, order, conds, B):
    lh = conds[0]
    rh = conds[1]
    B = np.concatenate((B,lh),axis=0)
    w = np.concatenate((yk,rh),axis=0)
    M,N = B.shape
    if (M>N):
        raise ValueError("over-specification of conditions")
    elif (M<N):
        return _find_smoothest(xk, yk, order, None, B)
    else:
        return np.dual.solve(B, w)

# If conds is None, then use the not_a_knot condition
#  at K-1 farthest separated points in the interval
def _find_not_a_knot(xk, yk, order, conds, B):
    raise NotImplementedError
    return _find_user(xk, yk, order, conds, B)

# If conds is None, then ensure zero-valued second
#  derivative at K-1 farthest separated points
def _find_natural(xk, yk, order, conds, B):
    raise NotImplementedError
    return _find_user(xk, yk, order, conds, B)

# If conds is None, then ensure zero-valued first
#  derivative at K-1 farthest separated points
def _find_clamped(xk, yk, order, conds, B):
    raise NotImplementedError
    return _find_user(xk, yk, order, conds, B)

def _find_fixed(xk, yk, order, conds, B):
    raise NotImplementedError
    return _find_user(xk, yk, order, conds, B)

# If conds is None, then use coefficient periodicity
# If conds is 'function' then use function periodicity
def _find_periodic(xk, yk, order, conds, B):
    raise NotImplementedError
    return _find_user(xk, yk, order, conds, B)

# Doesn't use conds
def _find_symmetric(xk, yk, order, conds, B):
    raise NotImplementedError
    return _find_user(xk, yk, order, conds, B)

# conds is a dictionary with multiple values
def _find_mixed(xk, yk, order, conds, B):
    raise NotImplementedError
    return _find_user(xk, yk, order, conds, B)


def splmake(xk,yk,order=3,kind='smoothest',conds=None):
    """Return a (xk, cvals, k) representation of a spline given
    data-points where the (internal) knots are at the data-points.

    yk can be an N-d array to represent more than one curve, through
    the same xk points. The first dimension is assumed to be the
    interpolating dimension.

    kind can be 'smoothest', 'not_a_knot', 'fixed',
                'clamped', 'natural', 'periodic', 'symmetric',
                'user', 'mixed'

                it is ignored if order < 2
    """
    yk = np.asanyarray(yk)
    N = yk.shape[0]-1

    order = int(order)
    if order < 0:
        raise ValueError("order must not be negative")
    if order == 0:
        return xk, yk[:-1], order
    elif order == 1:
        return xk, yk, order

    try:
        func = eval('_find_%s' % kind)
    except:
        raise NotImplementedError

    # the constraint matrix
    B = _fitpack._bsplmat(order, xk)
    coefs = func(xk, yk, order, conds, B)
    return xk, coefs, order

def spleval((xj,cvals,k),xnew,deriv=0):
    """Evaluate a fixed spline represented by the given tuple at the new
    x-values. The xj values are the interior knot points.  The approximation
    region is xj[0] to xj[-1].  If N+1 is the length of xj, then cvals should
    have length N+k where k is the order of the spline.

    Internally, an additional k-1 knot points are added on either side of
    the spline.

    If cvals represents more than one curve (cvals.ndim > 1) and/or xnew is
    N-d, then the result is xnew.shape + cvals.shape[1:] providing the
    interpolation of multiple curves.
    """
    oldshape = np.shape(xnew)
    xx = np.ravel(xnew)
    sh = cvals.shape[1:]
    res = np.empty(xx.shape + sh, dtype=cvals.dtype)
    for index in np.ndindex(*sh):
        sl = (slice(None),)+index
        if issubclass(cvals.dtype.type, np.complexfloating):
            res[sl].real = _fitpack._bspleval(xx,xj,cvals.real[sl],k,deriv)
            res[sl].imag = _fitpack._bspleval(xx,xj,cvals.imag[sl],k,deriv)
        else:
            res[sl] = _fitpack._bspleval(xx,xj,cvals[sl],k,deriv)
    res.shape = oldshape + sh
    return res

def spltopp(xk,cvals,k):
    """Return a piece-wise polynomial object from a fixed-spline tuple.
    """
    return ppform.fromspline(xk, cvals, k)

def spline(xk,yk,xnew,order=3,kind='smoothest',conds=None):
    """Interpolate a curve (xk,yk) at points xnew using a spline fit.
    """
    return spleval(splmake(xk,yk,order=order,kind=kind,conds=conds),xnew)
