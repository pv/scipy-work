"""
Test Scipy functions versus mpmath, if available.

"""
from __future__ import division, print_function, absolute_import

import sys
import re
import os

import nose
import numpy as np
from numpy.testing import dec
import scipy.special as sc
from scipy.lib.six import reraise

from scipy.special._testutils import FuncData, assert_func_equal

try:
    import mpmath
except ImportError:
    try:
        import sympy.mpmath as mpmath
    except ImportError:
        mpmath = None

def mpmath_check(min_ver):
    if mpmath is None:
        return dec.skipif(True, "mpmath library is not present")

    def try_int(v):
        try: return int(v)
        except ValueError: return v

    def get_version(v):
        return list(map(try_int, re.split('[^0-9]', v)))

    return dec.skipif(get_version(min_ver) > get_version(mpmath.__version__),
                      "mpmath %s required" % min_ver)


#------------------------------------------------------------------------------
# expi
#------------------------------------------------------------------------------

@mpmath_check('0.10')
def test_expi_complex():
    dataset = []
    for r in np.logspace(-99, 2, 10):
        for p in np.linspace(0, 2*np.pi, 30):
            z = r*np.exp(1j*p)
            dataset.append((z, complex(mpmath.ei(z))))
    dataset = np.array(dataset, dtype=np.complex_)

    FuncData(sc.expi, dataset, 0, 1).check()


#------------------------------------------------------------------------------
# hyp2f1
#------------------------------------------------------------------------------

@mpmath_check('0.14')
def test_hyp2f1_strange_points():
    pts = [
        (2,-1,-1,0.7),
        (2,-2,-2,0.7),
    ]
    kw = dict(eliminate=True)
    dataset = [p + (float(mpmath.hyp2f1(*p, **kw)),) for p in pts]
    dataset = np.array(dataset, dtype=np.float_)

    FuncData(sc.hyp2f1, dataset, (0,1,2,3), 4, rtol=1e-10).check()

@mpmath_check('0.13')
def test_hyp2f1_real_some_points():
    pts = [
        (1,2,3,0),
        (1./3, 2./3, 5./6, 27./32),
        (1./4, 1./2, 3./4, 80./81),
        (2,-2,-3,3),
        (2,-3,-2,3),
        (2,-1.5,-1.5,3),
        (1,2,3,0),
        (0.7235, -1, -5, 0.3),
        (0.25, 1./3, 2, 0.999),
        (0.25, 1./3, 2, -1),
        (2,3,5,0.99),
        (3./2,-0.5,3,0.99),
        (2,2.5,-3.25,0.999),
        (-8, 18.016500331508873, 10.805295997850628, 0.90875647507000001),
        (-10,900,-10.5,0.99),
        (-10,900,10.5,0.99),
        (-1,2,1,1.0),
        (-1,2,1,-1.0),
        (-3,13,5,1.0),
        (-3,13,5,-1.0),
    ]
    dataset = [p + (float(mpmath.hyp2f1(*p)),) for p in pts]
    dataset = np.array(dataset, dtype=np.float_)

    olderr = np.seterr(invalid='ignore')
    try:
        FuncData(sc.hyp2f1, dataset, (0,1,2,3), 4, rtol=1e-10).check()
    finally:
        np.seterr(**olderr)


@mpmath_check('0.14')
def test_hyp2f1_some_points_2():
    # Taken from mpmath unit tests -- this point failed for mpmath 0.13 but
    # was fixed in their SVN since then
    pts = [
        (112, (51,10), (-9,10), -0.99999),
        (10,-900,10.5,0.99),
        (10,-900,-10.5,0.99),
    ]

    def fev(x):
        if isinstance(x, tuple):
            return float(x[0]) / x[1]
        else:
            return x

    dataset = [tuple(map(fev, p)) + (float(mpmath.hyp2f1(*p)),) for p in pts]
    dataset = np.array(dataset, dtype=np.float_)

    FuncData(sc.hyp2f1, dataset, (0,1,2,3), 4, rtol=1e-10).check()

@mpmath_check('0.13')
def test_hyp2f1_real_some():
    dataset = []
    for a in [-10, -5, -1.8, 1.8, 5, 10]:
        for b in [-2.5, -1, 1, 7.4]:
            for c in [-9, -1.8, 5, 20.4]:
                for z in [-10, -1.01, -0.99, 0, 0.6, 0.95, 1.5, 10]:
                    try:
                        v = float(mpmath.hyp2f1(a, b, c, z))
                    except:
                        continue
                    dataset.append((a, b, c, z, v))
    dataset = np.array(dataset, dtype=np.float_)

    olderr = np.seterr(invalid='ignore')
    try:
        FuncData(sc.hyp2f1, dataset, (0,1,2,3), 4, rtol=1e-9).check()
    finally:
        np.seterr(**olderr)

@mpmath_check('0.12')
@dec.slow
def test_hyp2f1_real_random():
    dataset = []

    npoints = 500
    dataset = np.zeros((npoints, 5), np.float_)

    np.random.seed(1234)
    dataset[:,0] = np.random.pareto(1.5, npoints)
    dataset[:,1] = np.random.pareto(1.5, npoints)
    dataset[:,2] = np.random.pareto(1.5, npoints)
    dataset[:,3] = 2*np.random.rand(npoints) - 1

    dataset[:,0] *= (-1)**np.random.randint(2, npoints)
    dataset[:,1] *= (-1)**np.random.randint(2, npoints)
    dataset[:,2] *= (-1)**np.random.randint(2, npoints)

    for ds in dataset:
        if mpmath.__version__ < '0.14':
            # mpmath < 0.14 fails for c too much smaller than a, b
            if abs(ds[:2]).max() > abs(ds[2]):
                ds[2] = abs(ds[:2]).max()
        ds[4] = float(mpmath.hyp2f1(*tuple(ds[:4])))

    FuncData(sc.hyp2f1, dataset, (0,1,2,3), 4, rtol=1e-9).check()

#------------------------------------------------------------------------------
# erf (complex)
#------------------------------------------------------------------------------

@mpmath_check('0.14')
def test_erf_complex():
    # need to increase mpmath precision for this test
    old_dps, old_prec = mpmath.mp.dps, mpmath.mp.prec
    try:
        mpmath.mp.dps = 70
        x1, y1 = np.meshgrid(np.linspace(-10, 1, 31), np.linspace(-10, 1, 11))
        x2, y2 = np.meshgrid(np.logspace(-80, .8, 31), np.logspace(-80, .8, 11))
        points = np.r_[x1.ravel(),x2.ravel()] + 1j*np.r_[y1.ravel(),y2.ravel()]

        assert_func_equal(sc.erf, lambda x: complex(mpmath.erf(x)), points,
                          vectorized=False, rtol=1e-13)
        assert_func_equal(sc.erfc, lambda x: complex(mpmath.erfc(x)), points,
                          vectorized=False, rtol=1e-13)
    finally:
        mpmath.mp.dps, mpmath.mp.prec = old_dps, old_prec



#------------------------------------------------------------------------------
# lpmv
#------------------------------------------------------------------------------

@mpmath_check('0.15')
def test_lpmv():
    pts = []
    for x in [-0.99, -0.557, 1e-6, 0.132, 1]:
        pts.extend([
            (1, 1, x),
            (1, -1, x),
            (-1, 1, x),
            (-1, -2, x),
            (1, 1.7, x),
            (1, -1.7, x),
            (-1, 1.7, x),
            (-1, -2.7, x),
            (1, 10, x),
            (1, 11, x),
            (3, 8, x),
            (5, 11, x),
            (-3, 8, x),
            (-5, 11, x),
            (3, -8, x),
            (5, -11, x),
            (-3, -8, x),
            (-5, -11, x),
            (3, 8.3, x),
            (5, 11.3, x),
            (-3, 8.3, x),
            (-5, 11.3, x),
            (3, -8.3, x),
            (5, -11.3, x),
            (-3, -8.3, x),
            (-5, -11.3, x),
        ])

    dataset = [p + (mpmath.legenp(p[1], p[0], p[2]),) for p in pts]
    dataset = np.array(dataset, dtype=np.float_)

    evf = lambda mu,nu,x: sc.lpmv(mu.astype(int), nu, x)

    olderr = np.seterr(invalid='ignore')
    try:
        FuncData(evf, dataset, (0,1,2), 3, rtol=1e-10, atol=1e-14).check()
    finally:
        np.seterr(**olderr)


#------------------------------------------------------------------------------
# beta
#------------------------------------------------------------------------------

@mpmath_check('0.15')
def test_beta():
    np.random.seed(1234)

    b = np.r_[np.logspace(-200, 200, 4),
              np.logspace(-10, 10, 4),
              np.logspace(-1, 1, 4),
              -1, -2.3, -3, -100.3, -10003.4]
    a = b

    ab = np.array(np.broadcast_arrays(a[:,None], b[None,:])).reshape(2, -1).T

    old_dps, old_prec = mpmath.mp.dps, mpmath.mp.prec
    try:
        mpmath.mp.dps = 400

        assert_func_equal(sc.beta,
                          lambda a, b: float(mpmath.beta(a, b)),
                          ab,
                          vectorized=False,
                          rtol=1e-10)

        assert_func_equal(
            sc.betaln,
            lambda a, b: float(mpmath.log(abs(mpmath.beta(a, b)))),
            ab,
            vectorized=False,
            rtol=1e-10)
    finally:
        mpmath.mp.dps, mpmath.mp.prec = old_dps, old_prec


#------------------------------------------------------------------------------
# Machinery for systematic tests
#------------------------------------------------------------------------------

class Arg(object):
    def __init__(self, a=-np.inf, b=np.inf, inclusive_a=True, inclusive_b=True):
        self.a = a
        self.b = b
        self.inclusive_a = inclusive_a
        self.inclusive_b = inclusive_b
        if self.a == -np.inf:
            self.a = -np.finfo(float).max
        if self.b == np.inf:
            self.b = np.finfo(float).max

    def values(self, n):
        v1 = np.linspace(-1, 1, n//4)
        v2 = np.linspace(-10, 10, n//4)
        v3 = np.linspace(-self.a, self.b, n//4)
        if self.a >= 0 and self.b >= 0:
            v4 = np.logspace(-30, np.log10(self.b), n//4)
        elif self.a < 0 and self.b >= 0:
            v4 = np.r_[
                np.logspace(-30, np.log10(self.b), n//8),
                -np.logspace(-30, np.log10(-self.a), n//8)
                ]
        else:
            v4 = np.r_[
                -np.logspace(-30, np.log10(-self.b), n//8)
                ]
        v = np.r_[v1, v2, v3, v4]
        if self.inclusive_a:
            v = v[v >= self.a]
        else:
            v = v[v > self.a]
        if self.inclusive_b:
            v = v[v <= self.b]
        else:
            v = v[v < self.b]
        return v


class ComplexArg(object):
    def __init__(self, a=-np.inf, b=np.inf):
        self.real = Arg(a.real, b.real)
        self.imag = Arg(a.imag, b.imag)
    def values(self, n):
        m = int(np.sqrt(n)) + 1
        x = self.real.values(m)
        y = self.imag.values(m)
        return (x[:,None] + 1j*y[None,:]).ravel()


class IntArg(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def values(self, n):
        m = (self.b - self.a)
        v = np.arange(self.a, self.b, 1 + int(m//n))
        return v


class MpmathData(object):
    def __init__(self, scipy_func, mpmath_func, arg_spec, name=None,
                 dps=None, prec=None, n=5000, rtol=1e-10, atol=0):
        self.scipy_func = scipy_func
        self.mpmath_func = mpmath_func
        self.arg_spec = arg_spec
        self.dps = dps
        self.prec = prec
        self.n = n
        self.rtol = rtol
        self.atol = atol
        self.is_complex = any([isinstance(arg, ComplexArg) for arg in self.arg_spec])
        if not name or name == '<lambda>':
            name = getattr(scipy_func, '__name__', None)
        if not name or name == '<lambda>':
            name =  getattr(mpmath_func, '__name__', None)
        self.name = name

    def check(self):
        mpmath_check('0.17')(lambda: None)()

        np.random.seed(1234)

        # Generate values for the arguments
        num_args = len(self.arg_spec)
        m = int(self.n**(1./num_args)) + 3

        argvals = []
        for arg in self.arg_spec:
            argvals.append(arg.values(m))

        argarr = np.array(np.broadcast_arrays(*np.ix_(*argvals))).reshape(num_args, -1).T

        # Check
        old_dps, old_prec = mpmath.mp.dps, mpmath.mp.prec
        try:
            if self.dps is not None:
                dps_list = [self.dps]
            else:
                dps_list = [mpmath.mp.dps]
            if self.prec is not None:
                mpmath.mp.prec = self.prec

            if np.issubdtype(argarr.dtype, np.complexfloating):
                mptype = complex
            else:
                def mptype(x):
                    if abs(x.imag) != 0:
                        return np.nan
                    else:
                        return float(x)

            # try out different dps until one (or none) works
            for j, dps in enumerate(dps_list):
                mpmath.mp.dps = dps

                try:
                    assert_func_equal(self.scipy_func,
                                      lambda *a: mptype(self.mpmath_func(*a)),
                                      argarr,
                                      vectorized=False,
                                      rtol=self.rtol, atol=self.atol,
                                      nan_ok=True)
                    break
                except AssertionError:
                    if j >= len(dps_list)-1:
                        reraise(*sys.exc_info())
        finally:
            mpmath.mp.dps, mpmath.mp.prec = old_dps, old_prec

    def __repr__(self):
        if self.is_complex:
            return "<MpmathData: %s (complex)>" % (self.name,)
        else:
            return "<MpmathData: %s>" % (self.name,)

def assert_mpmath_equal(*a, **kw):
    d = MpmathData(*a, **kw)
    d.check()

def xslow(func):
    if 'SCIPY_TEST_XSLOW' not in os.environ:
        def wrap(*a, **kw):
            raise nose.SkipTest("Extremely slow test --- set environment "
                                "variable SCIPY_TEST_XSLOW to enable")
        wrap.__name__ = func.__name__
        return wrap
    else:
        return dec.slow(func)

#------------------------------------------------------------------------------
# Systematic tests
#------------------------------------------------------------------------------

@xslow
def test_systematic_airyai():
    assert_mpmath_equal(lambda z: sc.airy(z)[0],
                        mpmath.airyai,
                        [Arg()])

@xslow
def test_systematic_airyai_complex():
    assert_mpmath_equal(lambda z: sc.airy(z)[0],
                        mpmath.airyai,
                        [ComplexArg()])

@xslow
def test_systematic_airyai_prime():
    assert_mpmath_equal(lambda z: sc.airy(z)[1], lambda z:
                        mpmath.airyai(z, derivative=1),
                        [Arg()])

@xslow
def test_systematic_airyai_prime_complex():
    assert_mpmath_equal(lambda z: sc.airy(z)[1], lambda z:
                        mpmath.airyai(z, derivative=1),
                        [ComplexArg()])

@xslow
def test_systematic_airybi():
    assert_mpmath_equal(lambda z: sc.airy(z)[2], lambda z:
                        mpmath.airybi(z),
                        [Arg()])

@xslow
def test_systematic_airybi_complex():
    assert_mpmath_equal(lambda z: sc.airy(z)[2], lambda z:
                        mpmath.airybi(z),
                        [ComplexArg()])

@xslow
def test_systematic_airybi_prime():
    assert_mpmath_equal(lambda z: sc.airy(z)[3], lambda z:
                        mpmath.airybi(z, derivative=1),
                        [Arg()])

@xslow
def test_systematic_airybi_prime_complex():
    assert_mpmath_equal(lambda z: sc.airy(z)[3], lambda z:
                        mpmath.airybi(z, derivative=1),
                        [ComplexArg()])

@xslow
def test_systematic_bei():
    assert_mpmath_equal(sc.bei,
                        lambda z: mpmath.bei(0, z),
                        [Arg(-2e3, 2e3)],
                        rtol=1e-9)

@xslow
def test_systematic_ber():
    assert_mpmath_equal(sc.ber,
                        lambda z: mpmath.ber(0, z),
                        [Arg(-2e3, 2e3)],
                        rtol=1e-9)

@xslow
def test_systematic_bernoulli():
    assert_mpmath_equal(lambda n: sc.bernoulli(int(n))[int(n)],
                        lambda n: float(mpmath.bernoulli(int(n))),
                        [IntArg(0, 300)],
                        rtol=1e-9)

def _exception_to_nan(func):
    def wrap(*a, **kw):
        try:
            return func(*a, **kw)
        except:
            return np.nan
    return wrap

@xslow
def test_systematic_besseli():
    assert_mpmath_equal(sc.iv,
                        _exception_to_nan(mpmath.besseli),
                        [Arg(a=0), Arg(-1e9, 1e9)],
                        rtol=1e-9)

@xslow
def test_systematic_besseli_complex():
    assert_mpmath_equal(sc.iv,
                        mpmath.besseli,
                        [Arg(), ComplexArg()],
                        rtol=1e-9)

@xslow
def test_systematic_besselj():
    assert_mpmath_equal(sc.jv,
                        mpmath.besselj,
                        [Arg(), Arg()],
                        rtol=1e-9)

@xslow
def test_systematic_besselj_complex():
    assert_mpmath_equal(sc.jv,
                        mpmath.besselj,
                        [Arg(), ComplexArg()],
                        rtol=1e-9)

@xslow
def test_systematic_beta():
    assert_mpmath_equal(sc.beta,
                        mpmath.beta,
                        [Arg(), Arg()])

@xslow
def test_systematic_betainc():
    assert_mpmath_equal(sc.betainc,
                        mpmath.betainc,
                        [Arg(), Arg()])

@xslow
def test_systematic_binom():
    assert_mpmath_equal(sc.binom,
                        mpmath.binomial,
                        [Arg(), Arg()],
                        dps=400)
