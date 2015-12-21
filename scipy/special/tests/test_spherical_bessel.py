#
# Tests of spherical Bessel functions.
#

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from numpy import sin, cos, sinh, cosh, exp, inf, nan

from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn

# spherical_jn

def test_spherical_jn_exact():
    # http://dlmf.nist.gov/10.49.E3
    # Note: exact expression is numerically stable only for small
    # n or z >> n.
    x = np.array([0.12, 1.23, 12.34, 123.45, 1234.5])
    assert_allclose(spherical_jn(2, x),
                    (-1/x + 3/x**3)*sin(x) - 3/x**2*cos(x))

def test_spherical_yn_recurrence_complex():
    # http://dlmf.nist.gov/10.51.E1
    n = np.array([1, 2, 3, 7, 12])
    x = 1.1 + 1.5j
    assert_allclose(spherical_jn(n - 1, x) + spherical_jn(n + 1, x),
                    (2*n + 1)/x*spherical_jn(n, x))

def test_spherical_jn_recurrence_real():
    # http://dlmf.nist.gov/10.51.E1
    n = np.array([1, 2, 3, 7, 12])
    x = 0.12
    assert_allclose(spherical_jn(n - 1, x) + spherical_jn(n + 1,x),
                    (2*n + 1)/x*spherical_jn(n, x))

def test_spherical_jn_inf_real():
    # http://dlmf.nist.gov/10.52.E3
    n = 6
    x = np.array([0, -inf, inf])
    assert_allclose(spherical_jn(n, x), np.array([0, 0, 0]))

def test_spherical_jn_inf_complex():
    # http://dlmf.nist.gov/10.52.E3
    n = 7
    x = np.array([0, -inf, inf, inf*(1+1j)])
    assert_allclose(spherical_jn(n, x), np.array([0, 0, 0, inf*(1+1j)]))

def test_spherical_jn_large_arg_1():
    # https://github.com/scipy/scipy/issues/2165
    # Reference value computed using mpmath, via
    # besselj(n + mpf(1)/2, z)*sqrt(pi/(2*z))
    assert_allclose(spherical_jn(2, 3350.507), -0.00029846226538040747)

def test_spherical_jn_large_arg_2():
    # https://github.com/scipy/scipy/issues/1641
    # Reference value computed using mpmath, via
    # besselj(n + mpf(1)/2, z)*sqrt(pi/(2*z))
    assert_allclose(spherical_jn(2, 10000), 3.0590002633029811e-05)

# spherical_yn

def test_spherical_yn_exact():
    # http://dlmf.nist.gov/10.49.E5
    # Note: exact expression is numerically stable only for small
    # n or z >> n.
    x = np.array([0.12, 1.23, 12.34, 123.45, 1234.5])
    assert_allclose(spherical_yn(2, x),
                    (1/x - 3/x**3)*cos(x) - 3/x**2*sin(x))

def test_spherical_yn_recurrence_real():
    # http://dlmf.nist.gov/10.51.E1
    n = np.array([1, 2, 3, 7, 12])
    x = 0.12
    assert_allclose(spherical_yn(n - 1, x) + spherical_yn(n + 1,x),
                    (2*n + 1)/x*spherical_yn(n, x))

def test_spherical_yn_recurrence_complex():
    # http://dlmf.nist.gov/10.51.E1
    n = np.array([1, 2, 3, 7, 12])
    x = 1.1 + 1.5j
    assert_allclose(spherical_yn(n - 1, x) + spherical_yn(n + 1, x),
                    (2*n + 1)/x*spherical_yn(n, x))

def test_spherical_yn_inf_real():
    # http://dlmf.nist.gov/10.52.E3
    n = 6
    x = np.array([0, -inf, inf])
    assert_allclose(spherical_yn(n, x), np.array([nan, 0, 0]))

def test_spherical_yn_inf_complex():
    # http://dlmf.nist.gov/10.52.E3
    n = 7
    x = np.array([-inf, inf, inf*(1+1j)])
    assert_allclose(spherical_yn(n, x), np.array([0, 0, inf*(1+1j)]))

# jn, yn cross product tests

def test_spherical_jn_yn_cross_product_1():
    # http://dlmf.nist.gov/10.50.E3
    n = np.array([1, 5, 8])
    x = np.array([0.1, 1, 10])
    left = (  spherical_jn(n + 1, x)*spherical_yn(n,     x)
            - spherical_jn(n,     x)*spherical_yn(n + 1, x))
    right = 1/x**2
    assert_allclose(left, right)

def test_spherical_jn_yn_cross_product_2():
    # http://dlmf.nist.gov/10.50.E3
    n = np.array([1, 5, 8])
    x = np.array([0.1, 1, 10])
    left = (  spherical_jn(n + 2, x)*spherical_yn(n,     x)
            - spherical_jn(n,     x)*spherical_yn(n + 2, x))
    right = (2*n + 3)/x**3
    assert_allclose(left, right)

# spherical_in

def test_spherical_in_exact():
    # http://dlmf.nist.gov/10.49.E9
    x = np.array([0.12, 1.23, 12.34, 123.45])
    assert_allclose(spherical_in(2, x),
                    (1/x + 3/x**3)*sinh(x) - 3/x**2*cosh(x))

def test_spherical_in_recurrence_real():
    # http://dlmf.nist.gov/10.51.E4
    n = np.array([1, 2, 3, 7, 12])
    x = 0.12
    assert_allclose(spherical_in(n - 1, x) - spherical_in(n + 1,x),
                    (2*n + 1)/x*spherical_in(n, x))

def test_spherical_in_recurrence_complex():
    # http://dlmf.nist.gov/10.51.E1
    n = np.array([1, 2, 3, 7, 12])
    x = 1.1 + 1.5j
    assert_allclose(spherical_in(n - 1, x) - spherical_in(n + 1,x),
                    (2*n + 1)/x*spherical_in(n, x))

def test_spherical_in_inf_real():
    # http://dlmf.nist.gov/10.52.E3
    n = 5
    x = np.array([0, -inf, inf])
    assert_allclose(spherical_in(n, x), np.array([0, -inf, inf]))

def test_spherical_in_inf_complex():
    # http://dlmf.nist.gov/10.52.E5
    # Ideally, i1n(n, 1j*inf) = 0, but this appears impossible to achieve
    # because C99 regards any complex value with at least one infinite part as
    # a complex infinity.
    n = 7
    x = np.array([-inf, inf, inf*(1+1j)])
    assert_allclose(spherical_in(n, x), np.array([-inf, inf, inf*(1+1j)]))

# spherical_kn

def test_spherical_kn_exact():
    # http://dlmf.nist.gov/10.49.E13
    x = np.array([0.12, 1.23, 12.34, 123.45])
    assert_allclose(spherical_kn(2, x),
                    np.pi/2*exp(-x)*(1/x + 3/x**2 + 3/x**3))

def test_spherical_kn_recurrence_real():
    # http://dlmf.nist.gov/10.51.E4
    n = np.array([1, 2, 3, 7, 12])
    x = 0.12
    assert_allclose((-1)**(n - 1)*spherical_kn(n - 1, x) - (-1)**(n + 1)*spherical_kn(n + 1,x),
                    (-1)**n*(2*n + 1)/x*spherical_kn(n, x))

def test_spherical_kn_recurrence_complex():
    # http://dlmf.nist.gov/10.51.E4
    n = np.array([1, 2, 3, 7, 12])
    x = 1.1 + 1.5j
    assert_allclose((-1)**(n - 1)*spherical_kn(n - 1, x) - (-1)**(n + 1)*spherical_kn(n + 1,x),
                    (-1)**n*(2*n + 1)/x*spherical_kn(n, x))

def test_spherical_kn_inf_real():
    # http://dlmf.nist.gov/10.52.E6
    n = 5
    x = np.array([0, -inf, inf])
    assert_allclose(spherical_kn(n, x), np.array([nan, -inf, 0]))

def test_spherical_kn_inf_complex():
    # http://dlmf.nist.gov/10.52.E6
    n = 7
    x = np.array([-inf, inf, inf*(1+1j)])
    assert_allclose(spherical_kn(n, x), np.array([-inf, 0, inf*(1+1j)]))
