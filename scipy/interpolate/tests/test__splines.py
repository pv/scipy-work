from numpy.testing import assert_allclose
from scipy.interpolate._bspline import BSpline, _naive_eval

def test_simple_evaluation():
    spline = BSpline(t=[0, 0, 1, 2, 3, 4, 4], c=[1, 2, 1, 2, 1, 2, 1], k=2)
    a = spline(2.7)
    b = _naive_eval(2.7, spline.t, spline.c, spline.k)
    assert_allclose(a, b)
