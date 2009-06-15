from os.path import join, dirname

import numpy as np
from numpy.testing import *

from scipy.special.scipyfunc import gamma
from scipy.special.scipyfunc.testing import parse_txt_data, \
    assert_tol_equal, nulps_array

datadir = join(dirname(__file__), "data")

class _TestGamma(TestCase):
    dt = None
    atol = None
    rtol = None
    def _test(self, filename):
        data = parse_txt_data(join(datadir, filename)).astype(self.dt)
        x = data[:, 0]
        y_r = data[:, 1]

        y = gamma(x)
        self.failUnless(y.dtype == self.dt)

        # Check inf/nan values in ref and computed values coincide, and discard
        # them from assert_tol_equal
        id = np.isnan(y)
        id_r = np.isnan(y_r)
        np.testing.assert_array_equal(id, id_r)

        id = np.isinf(y)
        id_r = np.isinf(y_r)
        np.testing.assert_array_equal(id, id_r)
        np.testing.assert_array_equal(y[id], y_r[id])

        id = np.isfinite(y)
        id_r = np.isfinite(y_r)
        assert_tol_equal(y[id], y_r[id], atol=self.atol, rtol=self.rtol)

    def test_factorials(self):
        self._test("factorials.txt")

    def test_near_0(self):
        self._test("near_0.txt")

    def test_near_1(self):
        self._test("near_1.txt")

    def test_near_2(self):
        self._test("near_2.txt")

    def test_near_m10(self):
        self._test("near_2.txt")

    def test_near_m55(self):
        self._test("near_2.txt")

    def test_boundaries(self):
        self.failUnless(np.isinf(gamma(0)) and gamma(0) > 0)

        x = -1
        a = gamma(x)
        self.failUnless(np.isinf(a))

# Set those values correctly
class TestGamma(_TestGamma):
    dt = np.float
    atol = 1e-15
    rtol = 1e-15

class TestGammaSingle(_TestGamma):
    dt = np.float32
    atol = 1e-6
    rtol = 1e-6

class TestGammaLongExtended(_TestGamma):
    dt = np.longdouble
    atol = 1e-15
    rtol = 1e-15
