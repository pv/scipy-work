"""
Tests for Scipyfunc
"""

from numpy.testing import *
import numpy as np

from scipy.special import *
try:
    import _cephes as cephes
except ImportError:
    import scipy.special._cephes as cephes
try:
    import scipyfunc as scf
except ImportError:
    import scipy.special.scipyfunc as scf
from test_basic import assert_tol_equal


def point_check(func, points, values, rtol=1e-11, atol=0, types=None):
    if types is None:
        if hasattr(func, 'types'):
            # inspect what types the ufunc accepts
            types = []
            for sig in func.types:
                r = sig.split('->')
                if len(r) == 2:
                    types.append([np.sctypeDict[x] for x in r[0]])
        else:
            types = [np.asarray(x).dtype.type for x in points[0]]

    # different ways to pass in values
    if values is None:
        items = points
    elif callable(values):
        items = [(p, values(*p)) for p in points]
    else:
        items = zip(points, values)

    # check all points
    for args, value in items:
        for typ in types:
            a = [t(a) for t, a in zip(typ, args)]
            v = func(*tuple(a))
            assert allclose(v, value, rtol=rtol, atol=atol), (args, value)

class TestGamma(object):
    def test_gamma(self):
        point_check(scf.gamma, [[5]], [24.0])

    def test_gammaln(self):
        point_check(scf.gammaln, [[3]], lambda x: log(scf.gamma(x)))

class TestBessel(object):
    def _vs_amos_points(self):
        """Yield points at which to compare the implementation to AMOS"""
        # check several points, including large-amplitude ones
        for v in [-120, -100.3, -20., -10., -1., -.5,
                  0., 1., 12.49, 120., 301]:
            for z in [-1300, -11, -10, -1, 1., 10., 200.5, 401., 600.5,
                      700.6, 1300, 10003]:
                yield v, z
                
        # check half-integers; these were problematic points at least
        # for cephes/iv
        for v in 0.5 + arange(-60, 60):
            yield v, 3.5

    def check_vs_amos(self, f1, f2, rtol=1e-11, atol=0):
        for v, z in self._vs_amos_points():
            c1, c2 = f1(v, z), f2(v,z+0j)
            if np.isinf(c1):
                assert np.abs(c2) >= 1e300, (v, z)
            elif np.isnan(c1):
                assert c2.imag != 0, (v, z)
            else:
                assert_tol_equal(c1, c2, err_msg=(v, z), rtol=rtol, atol=atol)

    #def test_jv_vs_amos(self):
    #    self.check_vs_amos(jv, jn, rtol=1e-10, atol=1e-305)

    #def test_yv_vs_amos(self):
    #    self.check_vs_amos(yv, yn, rtol=1e-11, atol=1e-305)

    def test_iv_vs_amos(self):
        self.check_vs_amos(scf.iv, cephes.iv, rtol=1e-12, atol=1e-305)

    @dec.slow
    def test_iv_vs_amos_mass_test(self):
        N = 1000000
        np.random.seed(1)
        v = np.random.pareto(0.5, N) * (-1)**np.random.randint(2, size=N)
        x = np.random.pareto(0.2, N) * (-1)**np.random.randint(2, size=N)

        imsk = (np.random.randint(8, size=N) == 0)
        v[imsk] = v.astype(int)

        c1 = scf.iv(v, x)
        c2 = cephes.iv(v, x+0j)

        dc = abs(c1/c2 - 1)
        dc[np.isnan(dc)] = 0

        k = np.argmax(dc)

        # Most error apparently comes from AMOS and not our implementation;
        # there are some problems near integer orders there
        assert dc[k] < 1e-9, (iv(v[k], x[k]), iv(v[k], x[k]+0j))

    #def test_kv_cephes_vs_amos(self):
    #    #self.check_cephes_vs_amos(kv, kn, rtol=1e-9, atol=1e-305)
    #    self.check_cephes_vs_amos(kv, kv, rtol=1e-9, atol=1e-305)

    #def test_ticket_623(self):
    #    assert_tol_equal(jv(3, 4), 0.43017147387562193)
    #    assert_tol_equal(jv(301, 1300), 0.0183487151115275)
    #    assert_tol_equal(jv(301, 1296.0682), -0.0224174325312048)

    def test_ticket_853(self):
        """Negative-order Bessels"""
        # cephes
        #assert_tol_equal(jv(-1,   1   ), -0.4400505857449335)
        #assert_tol_equal(jv(-2,   1   ), 0.1149034849319005)
        #assert_tol_equal(yv(-1,   1   ), 0.7812128213002887)
        #assert_tol_equal(yv(-2,   1   ), -1.650682606816255)
        assert_tol_equal(scf.iv(-1,   1   ), 0.5651591039924851)
        assert_tol_equal(scf.iv(-2,   1   ), 0.1357476697670383)
        #assert_tol_equal(kv(-1,   1   ), 0.6019072301972347)
        #assert_tol_equal(kv(-2,   1   ), 1.624838898635178)
        #assert_tol_equal(jv(-0.5, 1   ), 0.43109886801837607952)
        #assert_tol_equal(yv(-0.5, 1   ), 0.6713967071418031)
        assert_tol_equal(scf.iv(-0.5, 1   ), 1.231200214592967)
        #assert_tol_equal(kv(-0.5, 1   ), 0.4610685044478945)
        # amos
        #assert_tol_equal(jv(-1,   1+0j), -0.4400505857449335)
        #assert_tol_equal(jv(-2,   1+0j), 0.1149034849319005)
        #assert_tol_equal(yv(-1,   1+0j), 0.7812128213002887)
        #assert_tol_equal(yv(-2,   1+0j), -1.650682606816255)

        #assert_tol_equal(scf.iv(-1,   1+0j), 0.5651591039924851)
        #assert_tol_equal(scf.iv(-2,   1+0j), 0.1357476697670383)
        #assert_tol_equal(scf.kv(-1,   1+0j), 0.6019072301972347)
        #assert_tol_equal(scf.kv(-2,   1+0j), 1.624838898635178)
        
        #assert_tol_equal(jv(-0.5, 1+0j), 0.43109886801837607952)
        #assert_tol_equal(jv(-0.5, 1+1j), 0.2628946385649065-0.827050182040562j)
        #assert_tol_equal(yv(-0.5, 1+0j), 0.6713967071418031)
        #assert_tol_equal(yv(-0.5, 1+1j), 0.967901282890131+0.0602046062142816j)
        
        #assert_tol_equal(scf.iv(-0.5, 1+0j), 1.231200214592967)
        #assert_tol_equal(scf.iv(-0.5, 1+1j), 0.77070737376928+0.39891821043561j)
        #assert_tol_equal(kv(-0.5, 1+0j), 0.4610685044478945)
        #assert_tol_equal(kv(-0.5, 1+1j), 0.06868578341999-0.38157825981268j)

        #assert_tol_equal(jve(-0.5,1+0.3j), jv(-0.5, 1+0.3j)*exp(-0.3))
        #assert_tol_equal(yve(-0.5,1+0.3j), yv(-0.5, 1+0.3j)*exp(-0.3))
        #assert_tol_equal(ive(-0.5,0.3+1j), iv(-0.5, 0.3+1j)*exp(-0.3))
        #assert_tol_equal(kve(-0.5,0.3+1j), kv(-0.5, 0.3+1j)*exp(0.3+1j))

        #assert_tol_equal(hankel1(-0.5, 1+1j), jv(-0.5, 1+1j) + 1j*yv(-0.5,1+1j))
        #assert_tol_equal(hankel2(-0.5, 1+1j), jv(-0.5, 1+1j) - 1j*yv(-0.5,1+1j))

    def test_ticket_854(self):
        """Real-valued Bessel domains"""
        #assert isnan(jv(0.5, -1))
        assert isnan(scf.iv(0.5, -1))
        #assert isnan(yv(0.5, -1))
        #assert isnan(yv(1, -1))
        #assert isnan(kv(0.5, -1))
        #assert isnan(kv(1, -1))
        #assert isnan(jve(0.5, -1))
        #assert isnan(ive(0.5, -1))
        #assert isnan(yve(0.5, -1))
        #assert isnan(yve(1, -1))
        #assert isnan(kve(0.5, -1))
        #assert isnan(kve(1, -1))
        #assert isnan(airye(-1)[0:2]).all(), airye(-1)
        #assert not isnan(airye(-1)[2:4]).any(), airye(-1)

    def test_ticket_503(self):
        """Real-valued Bessel I overflow"""
        assert_tol_equal(scf.iv(1, 700), 1.528500390233901e302)
        assert_tol_equal(scf.iv(1000, 1120), 1.301564549405821e301)

    def test_iv_hyperg_poles(self):
        assert_tol_equal(scf.iv(-0.5, 1), 1.231200214592967)

    def iv_series(self, v, z, n=200):
        k = arange(0, n).astype(float_)
        r = (v+2*k)*log(.5*z) - gammaln(k+1) - gammaln(v+k+1)
        r[isnan(r)] = inf
        r = exp(r)
        err = abs(r).max() * finfo(float_).eps * n + abs(r[-1])*10
        return r.sum(), err

    def test_iv_series(self):
        for v in [-20., -10., -1., 0., 1., 12.49, 120.]:
            for z in [1., 10., 200.5]:
                value, err = self.iv_series(v, z)
                assert_tol_equal(scf.iv(v, z), value, atol=err, err_msg=(v, z))

if __name__ == "__main__":
    run_module_suite()
