"""
Convergence regions of the expansions used in ``struve.c``

"""
from __future__ import absolute_import, division, print_function

import numpy as np
from scipy import special
import matplotlib.pyplot as plt

from joblib import Memory

try:
    import mpmath
except:
    from sympy import mpmath

MAXITER = 20000
SUM_EPS = 1e-100

mem = Memory('cache')

@mem.cache
def asymp_series(v, z, is_h=True):
    assert z > 0

    m = z/2
    if m <= 0:
        maxiter = 0
    elif m > MAXITER:
        maxiter = MAXITER
    else:
        maxiter = int(m)
    if maxiter == 0:
        return np.nan, np.inf

    if is_h:
        sgn = -1.0
    else:
        sgn = 1.0

    maxterm = 0
    term = -sgn / np.sqrt(np.pi) * np.exp(-special.gammaln(v + 0.5) + (v - 1) * np.log(z/2)) * special.gammasgn(v + 0.5)
    sum = term
    for n in range(maxiter):
        term *= sgn * (1 + 2*n) * (1 + 2*n - 2*v) / (z*z)
        sum += term;
        maxterm = max(maxterm, abs(term))
        if abs(term) < SUM_EPS * abs(sum) or term == 0 or not np.isfinite(sum):
            break

    if is_h:
        sum += special.yv(v, z)
    else:
        sum += special.iv(v, z)
    err = abs(term) + 1e-16*maxterm

    return sum, err

asymp_series = np.vectorize(asymp_series, otypes='dd')

@mem.cache
def power_series_double2(v, z, is_h=True):
    if is_h:
        sgn = -1.0
    else:
        sgn = 1.0

    term = 2 / np.sqrt(np.pi) * np.exp(-special.gammaln(v + 1.5) + (v + 1)*np.log(z/2)) * special.gammasgn(v + 1.5)
    maxterm = abs(term)

    term = Double2(term)
    sum = term

    for n in range(MAXITER):
        term *= Double2(sgn * z*z) / Double2(3 + 2*n) / (Double2(3 + 2*n) + Double2(2*v))
        sum += term

        maxterm = max(maxterm, abs(term))
        if abs(term) < SUM_EPS * abs(sum) or term == 0 or not np.isfinite(float(sum)):
            break

    return float(sum), float(abs(term) + 1e-22*maxterm)

power_series_double2 = np.vectorize(power_series_double2, otypes='dd')

@mem.cache
def bessel_series(v, z, is_h):
    maxterm = 0

    sum = 0

    for n in range(MAXITER):
        n = np.float64(n)
        if is_h:
            term = np.sqrt(z / (2*np.pi)) * (.5*z)**n / special.gamma(n + 1) / (n + 0.5) * special.jv(n + v + 0.5, z)
        else:
            term = np.sqrt(z / (2*np.pi)) * (-.5*z)**n / special.gamma(n + 1) / (n + 0.5) * special.iv(n + v + 0.5, z)
        sum += term

        maxterm = max(maxterm, abs(term))
        if abs(term) < SUM_EPS * abs(sum) or term == 0 or not np.isfinite(sum) and n > 5:
            break

    return sum, abs(term) + 1e-14*maxterm

bessel_series = np.vectorize(bessel_series, otypes='dd')

def err_metric(a, b, atol=1e-290):
    m = abs(a - b) / (atol + abs(b))
    m[np.isinf(b) & (a == b)] = 0
    return m


def do_plot(is_h=True):
    vs = np.linspace(-500, 500, 51)
    zs = np.sort(np.r_[1e-5, 1.0, np.linspace(0, 300, 41)[1:]])

    rp = power_series_double2(vs[:,None], zs[None,:], is_h)
    ra = asymp_series(vs[:,None], zs[None,:], is_h)
    rb = bessel_series(vs[:,None], zs[None,:], is_h)

    mpmath.mp.dps = 160
    if is_h:
        sh = lambda v, z: float(mpmath.struveh(mpmath.mpf(v), mpmath.mpf(z)))
    else:
        sh = lambda v, z: float(mpmath.struvel(mpmath.mpf(v), mpmath.mpf(z)))
    ex = np.vectorize(sh, otypes='d')(vs[:,None], zs[None,:])

    err_a = err_metric(ra[0], ex) + 1e-300
    err_p = err_metric(rp[0], ex) + 1e-300
    err_b = err_metric(rb[0], ex) + 1e-300

    z_cutoff = np.where(vs >= 0, 0.7*abs(vs) + 12, 0.7*abs(vs) + 12)

    levels = [-1000, -12]

    plt.cla()

    plt.hold(1)
    plt.contourf(vs, zs, np.log10(err_p).T, levels=levels, colors=['r', 'r'], alpha=0.1)
    plt.contourf(vs, zs, np.log10(err_a).T, levels=levels, colors=['b', 'b'], alpha=0.1)
    plt.contourf(vs, zs, np.log10(err_b).T, levels=levels, colors=['g', 'g'], alpha=0.1)

    lp = plt.contour(vs, zs, np.log10(err_p).T, levels=levels, colors=['r', 'r'], linestyles=[':', '-'])
    la = plt.contour(vs, zs, np.log10(err_a).T, levels=levels, colors=['b', 'b'], linestyles=[':', '-'])
    lb = plt.contour(vs, zs, np.log10(err_b).T, levels=levels, colors=['g', 'g'], linestyles=[':', '-'])

    plt.clabel(lp, fmt={-1000: 'P', -12: 'P'})
    plt.clabel(la, fmt={-1000: 'A', -12: 'A'})
    plt.clabel(lb, fmt={-1000: 'B', -12: 'B'})

    plt.plot(vs, z_cutoff, 'k--')

    plt.xlim(vs.min(), vs.max())
    plt.ylim(zs.min(), zs.max())


def main():
    plt.clf()
    plt.subplot(121)
    do_plot(True)
    plt.title('Struve H')

    plt.subplot(122)
    do_plot(False)
    plt.title('Struve L')

    plt.savefig('struve_convergence.png')
    plt.show()

class Double2(object):
    """
    2-double floating point number
    """

    def __init__(self, hi, lo=0.0):
        self.hi = float(hi)
        self.lo = float(lo)

    @staticmethod
    def _sum_err(a, b):
        if abs(a) < abs(b):
            a, b = b, a

        c = a + b
        e = c - a
        g = c - e
        h = g - a
        f = b - h
        d = f - e
        x = d + e
        if x != f:
            c = a
            d = b
        return c, d

    @staticmethod
    def _split(a):
        SPLIT_TRESH = 6.69692879491417e+299
        SPLITTER = 134217729.0

        if a > SPLIT_TRESH or a < -SPLIT_TRESH:
            a *= 3.7252902984619140625e-09
            b = SPLITTER * a
            c = b - a
            hi = b - c
            lo = a - hi
            hi *= 268435456.0
            lo *= 268435456.0
        else:
            b = SPLITTER * a
            c = b - a
            hi = b - c
            lo = a - hi
        return hi, lo

    @staticmethod
    def _mul_err(a, b):
        a_hi, a_lo = Double2._split(a)
        b_hi, b_lo = Double2._split(b)
        p = a * b
        c = a_hi * b_hi - p
        d = c + a_hi * b_lo + a_lo * b_hi
        err = d + a_lo * b_lo
        return p, err

    @classmethod
    def _construct_ni(cls, obj):
        if not isinstance(obj, cls):
            try:
                return cls(obj)
            except ValueError:
                return None
        return obj

    def __neg__(self):
        return Double2(-self.hi, -self.lo)

    def __pos__(self):
        return self

    def __abs__(self):
        if self.hi < 0:
            return -self
        elif self.hi == 0 and self.lo < 0:
            return -self
        else:
            return self

    def __add__(self, other):
        other = Double2._construct_ni(other)
        if other is None:
            return NotImplemented

        s1, s2 = Double2._sum_err(self.hi, other.hi)
        t1, t2 = Double2._sum_err(self.lo, other.lo)

        s2 += t1
        s1, s2 = Double2._sum_err(s1, s2)
        s2 += t2
        s1, s2 = Double2._sum_err(s1, s2)

        return Double2(s1, s2)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        other = Double2._construct_ni(other)
        if other is None:
            return NotImplemented

        p1, p2 = Double2._mul_err(self.hi, other.hi)
        p2 += self.hi * other.lo + self.lo * other.hi
        p1, p2 = Double2._sum_err(p1, p2)
        return Double2(p1, p2)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = Double2._construct_ni(other)
        if other is None:
            return NotImplemented

        q1 = self.hi / other.hi
        r = self - q1*other

        q2 = r.hi / other.hi
        r = r - q2*other

        q3 = r.hi / other.hi
        q1, q2 = Double2._sum_err(q1, q2)
        e = Double2(q1, q2)
        return e + q3

    def __rtruediv__(self, other):
        other = Double2._construct_ni(other)
        if other is None:
            return NotImplemented
        return other.__div__(self)

    __div__ = __truediv__
    __rdiv__ = __rtruediv__

    def __cmp__(self, other):
        other = Double2._construct_ni(other)
        if other is None:
            return NotImplemented
        if self.hi == other.hi:
            return cmp(self.lo, other.lo)
        return cmp(self.hi, other.hi)

    def __float__(self):
        return self.hi + self.lo

    def __repr__(self):
        return "Double2(%r, %r)" % (self.hi, self.lo)

if __name__ == "__main__":
    main()
