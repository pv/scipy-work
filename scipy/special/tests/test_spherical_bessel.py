#
# Tests of spherical Bessel functions.
#

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose, dec
from numpy import sin, cos, sinh, cosh, exp, inf, nan

from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
from scipy.integrate import quad

class TestSphericalJn:
    def test_spherical_jn_exact(self):
        # http://dlmf.nist.gov/10.49.E3
        # Note: exact expression is numerically stable only for small
        # n or z >> n.
        x = np.array([0.12, 1.23, 12.34, 123.45, 1234.5])
        assert_allclose(spherical_jn(2, x),
                        (-1/x + 3/x**3)*sin(x) - 3/x**2*cos(x))

    def test_spherical_jn_recurrence_complex(self):
        # http://dlmf.nist.gov/10.51.E1
        n = np.array([1, 2, 3, 7, 12])
        x = 1.1 + 1.5j
        assert_allclose(spherical_jn(n - 1, x) + spherical_jn(n + 1, x),
                        (2*n + 1)/x*spherical_jn(n, x))

    def test_spherical_jn_recurrence_real(self):
        # http://dlmf.nist.gov/10.51.E1
        n = np.array([1, 2, 3, 7, 12])
        x = 0.12
        assert_allclose(spherical_jn(n - 1, x) + spherical_jn(n + 1,x),
                        (2*n + 1)/x*spherical_jn(n, x))

    def test_spherical_jn_inf_real(self):
        # http://dlmf.nist.gov/10.52.E3
        n = 6
        x = np.array([-inf, inf])
        assert_allclose(spherical_jn(n, x), np.array([0, 0]))

    def test_spherical_jn_inf_complex(self):
        # http://dlmf.nist.gov/10.52.E3
        n = 7
        x = np.array([-inf + 0j, inf + 0j, inf*(1+1j)])
        assert_allclose(spherical_jn(n, x), np.array([0, 0, inf*(1+1j)]))

    def test_spherical_jn_large_arg_1(self):
        # https://github.com/scipy/scipy/issues/2165
        # Reference value computed using mpmath, via
        # besselj(n + mpf(1)/2, z)*sqrt(pi/(2*z))
        assert_allclose(spherical_jn(2, 3350.507), -0.00029846226538040747)

    def test_spherical_jn_large_arg_2(self):
        # https://github.com/scipy/scipy/issues/1641
        # Reference value computed using mpmath, via
        # besselj(n + mpf(1)/2, z)*sqrt(pi/(2*z))
        assert_allclose(spherical_jn(2, 10000), 3.0590002633029811e-05)

    def test_spherical_jn_at_zero(self):
        # http://dlmf.nist.gov/10.52.E1
        # But note that n = 0 is a special case: j0 = sin(x)/x -> 1
        n = np.array([0, 1, 2, 5, 10, 100])
        x = 0
        assert_allclose(spherical_jn(n, x), np.array([1, 0, 0, 0, 0, 0]))


class TestSphericalYn:
    def test_spherical_yn_exact(self):
        # http://dlmf.nist.gov/10.49.E5
        # Note: exact expression is numerically stable only for small
        # n or z >> n.
        x = np.array([0.12, 1.23, 12.34, 123.45, 1234.5])
        assert_allclose(spherical_yn(2, x),
                        (1/x - 3/x**3)*cos(x) - 3/x**2*sin(x))

    def test_spherical_yn_recurrence_real(self):
        # http://dlmf.nist.gov/10.51.E1
        n = np.array([1, 2, 3, 7, 12])
        x = 0.12
        assert_allclose(spherical_yn(n - 1, x) + spherical_yn(n + 1,x),
                        (2*n + 1)/x*spherical_yn(n, x))

    def test_spherical_yn_recurrence_complex(self):
        # http://dlmf.nist.gov/10.51.E1
        n = np.array([1, 2, 3, 7, 12])
        x = 1.1 + 1.5j
        assert_allclose(spherical_yn(n - 1, x) + spherical_yn(n + 1, x),
                        (2*n + 1)/x*spherical_yn(n, x))

    def test_spherical_yn_inf_real(self):
        # http://dlmf.nist.gov/10.52.E3
        n = 6
        x = np.array([-inf, inf])
        assert_allclose(spherical_yn(n, x), np.array([0, 0]))

    def test_spherical_yn_inf_complex(self):
        # http://dlmf.nist.gov/10.52.E3
        n = 7
        x = np.array([-inf + 0j, inf + 0j, inf*(1+1j)])
        assert_allclose(spherical_yn(n, x), np.array([0, 0, inf*(1+1j)]))

    def test_spherical_yn_at_zero(self):
        # http://dlmf.nist.gov/10.52.E2
        n = np.array([0, 1, 2, 5, 10, 100])
        x = 0
        assert_allclose(spherical_yn(n, x), -inf*np.ones(shape=n.shape))

    def test_spherical_yn_at_zero_complex(self):
        # Consistently with numpy:
        # >>> -np.cos(0)/0
        # -inf
        # >>> -np.cos(0+0j)/(0+0j)
        # (-inf + nan*j)
        n = np.array([0, 1, 2, 5, 10, 100])
        x = 0 + 0j
        assert_allclose(spherical_yn(n, x), nan*np.ones(shape=n.shape))


class TestSphericalJnYnCrossProduct:
    def test_spherical_jn_yn_cross_product_1(self):
        # http://dlmf.nist.gov/10.50.E3
        n = np.array([1, 5, 8])
        x = np.array([0.1, 1, 10])
        left = (spherical_jn(n + 1, x) * spherical_yn(n, x) -
                spherical_jn(n, x) * spherical_yn(n + 1, x))
        right = 1/x**2
        assert_allclose(left, right)

    def test_spherical_jn_yn_cross_product_2(self):
        # http://dlmf.nist.gov/10.50.E3
        n = np.array([1, 5, 8])
        x = np.array([0.1, 1, 10])
        left = (spherical_jn(n + 2, x) * spherical_yn(n, x) -
                spherical_jn(n, x) * spherical_yn(n + 2, x))
        right = (2*n + 3)/x**3
        assert_allclose(left, right)


class TestSphericalIn:
    def test_spherical_in_exact(self):
        # http://dlmf.nist.gov/10.49.E9
        x = np.array([0.12, 1.23, 12.34, 123.45])
        assert_allclose(spherical_in(2, x),
                        (1/x + 3/x**3)*sinh(x) - 3/x**2*cosh(x))

    def test_spherical_in_recurrence_real(self):
        # http://dlmf.nist.gov/10.51.E4
        n = np.array([1, 2, 3, 7, 12])
        x = 0.12
        assert_allclose(spherical_in(n - 1, x) - spherical_in(n + 1,x),
                        (2*n + 1)/x*spherical_in(n, x))

    def test_spherical_in_recurrence_complex(self):
        # http://dlmf.nist.gov/10.51.E1
        n = np.array([1, 2, 3, 7, 12])
        x = 1.1 + 1.5j
        assert_allclose(spherical_in(n - 1, x) - spherical_in(n + 1,x),
                        (2*n + 1)/x*spherical_in(n, x))

    def test_spherical_in_inf_real(self):
        # http://dlmf.nist.gov/10.52.E3
        n = 5
        x = np.array([-inf, inf])
        assert_allclose(spherical_in(n, x), np.array([-inf, inf]))

    def test_spherical_in_inf_complex(self):
        # http://dlmf.nist.gov/10.52.E5
        # Ideally, i1n(n, 1j*inf) = 0 and i1n(n, (1+1j)*inf) = (1+1j)*inf, but
        # this appears impossible to achieve because C99 regards any complex
        # value with at least one infinite  part as a complex infinity, so
        # 1j*inf cannot be distinguished from (1+1j)*inf.  Therefore, nan is
        # the correct return value.
        n = 7
        x = np.array([-inf + 0j, inf + 0j, inf*(1+1j)])
        assert_allclose(spherical_in(n, x), np.array([-inf, inf, nan]))

    def test_spherical_in_at_zero(self):
        # http://dlmf.nist.gov/10.52.E1
        # But note that n = 0 is a special case: i0 = sinh(x)/x -> 1
        n = np.array([0, 1, 2, 5, 10, 100])
        x = 0
        assert_allclose(spherical_in(n, x), np.array([1, 0, 0, 0, 0, 0]))


class TestSphericalKn:
    def test_spherical_kn_exact(self):
        # http://dlmf.nist.gov/10.49.E13
        x = np.array([0.12, 1.23, 12.34, 123.45])
        assert_allclose(spherical_kn(2, x),
                        np.pi/2*exp(-x)*(1/x + 3/x**2 + 3/x**3))

    def test_spherical_kn_recurrence_real(self):
        # http://dlmf.nist.gov/10.51.E4
        n = np.array([1, 2, 3, 7, 12])
        x = 0.12
        assert_allclose((-1)**(n - 1)*spherical_kn(n - 1, x) - (-1)**(n + 1)*spherical_kn(n + 1,x),
                        (-1)**n*(2*n + 1)/x*spherical_kn(n, x))

    def test_spherical_kn_recurrence_complex(self):
        # http://dlmf.nist.gov/10.51.E4
        n = np.array([1, 2, 3, 7, 12])
        x = 1.1 + 1.5j
        assert_allclose((-1)**(n - 1)*spherical_kn(n - 1, x) - (-1)**(n + 1)*spherical_kn(n + 1,x),
                        (-1)**n*(2*n + 1)/x*spherical_kn(n, x))

    def test_spherical_kn_inf_real(self):
        # http://dlmf.nist.gov/10.52.E6
        n = 5
        x = np.array([-inf, inf])
        assert_allclose(spherical_kn(n, x), np.array([-inf, 0]))

    def test_spherical_kn_inf_complex(self):
        # http://dlmf.nist.gov/10.52.E6
        # The behavior at complex infinity depends on the sign of the real
        # part: if Re(z) >= 0, then the limit is 0; if Re(z) < 0, then it's
        # z*inf.  This distinction cannot be captured, so we return nan.
        n = 7
        x = np.array([-inf + 0j, inf + 0j, inf*(1+1j)])
        assert_allclose(spherical_kn(n, x), np.array([-inf, 0, nan]))

    def test_spherical_kn_at_zero(self):
        # http://dlmf.nist.gov/10.52.E2
        n = np.array([0, 1, 2, 5, 10, 100])
        x = 0
        assert_allclose(spherical_kn(n, x), inf*np.ones(shape=n.shape))

    def test_spherical_kn_at_zero_complex(self):
        # http://dlmf.nist.gov/10.52.E2
        n = np.array([0, 1, 2, 5, 10, 100])
        x = 0 + 0j
        assert_allclose(spherical_kn(n, x), nan*np.ones(shape=n.shape))


class SphericalDerivativesTestCase:
    def fundamental_theorem(self, n, a, b):
        integral, tolerance = quad(lambda z: self.df(n, z), a, b)
        assert_allclose(integral,
                        self.f(n, b) - self.f(n, a),
                        atol=tolerance)

    @dec.slow
    def test_fundamental_theorem_0(self):
        self.fundamental_theorem(0, 3.0, 15.0)

    @dec.slow
    def test_fundamental_theorem_7(self):
        self.fundamental_theorem(7, 0.5, 1.2)


class TestSphericalJnDerivatives(SphericalDerivativesTestCase):
    def f(self, n, z):
        return spherical_jn(n, z)

    def df(self, n, z):
        return spherical_jn(n, z, derivative=True)

    def test_spherical_jn_d_zero(self):
        n = np.array([1, 2, 3, 7, 15])
        assert_allclose(spherical_jn(n, 0, derivative=True),
                        np.zeros(5))


class TestSphericalYnDerivatives(SphericalDerivativesTestCase):
    def f(self, n, z):
        return spherical_yn(n, z)

    def df(self, n, z):
        return spherical_yn(n, z, derivative=True)


class TestSphericalInDerivatives(SphericalDerivativesTestCase):
    def f(self, n, z):
        return spherical_in(n, z)

    def df(self, n, z):
        return spherical_in(n, z, derivative=True)

    def test_spherical_in_d_zero(self):
        n = np.array([1, 2, 3, 7, 15])
        assert_allclose(spherical_in(n, 0, derivative=True),
                        np.zeros(5))


class TestSphericalKnDerivatives(SphericalDerivativesTestCase):
    def f(self, n, z):
        return spherical_kn(n, z)

    def df(self, n, z):
        return spherical_kn(n, z, derivative=True)
