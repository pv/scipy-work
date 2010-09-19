import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_almost_equal, \
        run_module_suite, dec

import scipy.interpolate.interpnd as interpnd
import scipy.spatial.qhull as qhull

class TestLinearNDInterpolation(object):
    def test_smoketest(self):
        # Test at single points
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.double)
        y = np.arange(x.shape[0], dtype=np.double)

        yi = interpnd.LinearNDInterpolator(x, y)(x)
        assert_almost_equal(y, yi)

    def test_complex_smoketest(self):
        # Test at single points
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.double)
        y = np.arange(x.shape[0], dtype=np.double)
        y = y - 3j*y

        yi = interpnd.LinearNDInterpolator(x, y)(x)
        assert_almost_equal(y, yi)

    def test_square(self):
        # Test barycentric interpolation on a square against a manual
        # implementation

        points = np.array([(0,0), (0,1), (1,1), (1,0)], dtype=np.double)
        values = np.array([1., 2., -3., 5.], dtype=np.double)

        # NB: assume triangles (0, 1, 3) and (1, 2, 3)
        #
        #  1----2
        #  | \  |
        #  |  \ |
        #  0----3

        def ip(x, y):
            t1 = (x + y <= 1)
            t2 = ~t1

            x1 = x[t1]
            y1 = y[t1]

            x2 = x[t2]
            y2 = y[t2]

            z = 0*x

            z[t1] = (values[0]*(1 - x1 - y1)
                     + values[1]*y1
                     + values[3]*x1)

            z[t2] = (values[2]*(x2 + y2 - 1)
                     + values[1]*(1 - x2)
                     + values[3]*(1 - y2))
            return z

        xx, yy = np.broadcast_arrays(np.linspace(0, 1, 14)[:,None],
                                     np.linspace(0, 1, 14)[None,:])
        xx = xx.ravel()
        yy = yy.ravel()

        xi = np.array([xx, yy]).T.copy()
        zi = interpnd.LinearNDInterpolator(points, values)(xi)

        assert_almost_equal(zi, ip(xx, yy))

class TestEstimateGradients2DGlobal(object):
    def test_smoketest(self):
        x = np.array([(0, 0), (0, 2),
                      (1, 0), (1, 2), (0.25, 0.75), (0.6, 0.8)], dtype=float)
        tri = qhull.Delaunay(x)

        # Should be exact for linear functions, independent of triangulation

        funcs = [
            (lambda x, y: 0*x + 1,            (0, 0)),
            (lambda x, y: 0 + x,              (1, 0)),
            (lambda x, y: -2 + y,             (0, 1)),
            (lambda x, y: 3 + 3*x + 14.15*y,  (3, 14.15))
        ]

        for j, (func, grad) in enumerate(funcs):
            z = func(x[:,0], x[:,1])
            dz = interpnd.estimate_gradients_2d_global(tri, z, tol=1e-6)

            assert_equal(dz.shape, (6, 2))
            assert_allclose(dz, np.array(grad)[None,:] + 0*dz,
                            rtol=1e-5, atol=1e-5, err_msg="item %d" % j)

class TestEstimateSmoothingNDGlobal(object):
    def test_simple_2d(self):
        points = np.array([(0.0,   0.0),
                           (0.0,   1.0),
                           (1.0,   1.0),
                           (1.0,   0.0),
                           ])
        values = np.array([0.0, 3.0, 1.0, 1.0])
        self._check_dataset_2d(points, values)

    def test_random_2d(self):
        np.random.seed(1234)
        points = np.random.randn(30, 2)
        values = np.random.randn(30)
        self._check_dataset_2d(points, values)

    def test_random_3d(self):
        np.random.seed(1234)
        points = np.random.randn(30, 3)
        values = np.random.randn(30)
        self._check_dataset_nd(points, values)

    def test_random_4d(self):
        np.random.seed(1234)
        points = np.random.randn(30, 4)
        values = np.random.randn(30)
        self._check_dataset_nd(points, values)

    def test_random_5d(self):
        np.random.seed(1234)
        points = np.random.randn(30, 5)
        values = np.random.randn(30)
        self._check_dataset_nd(points, values)
    
    def _check_dataset_2d(self, points, values, qtol=1e-4):
        tri = qhull.Delaunay(points)

        #
        # 1) Compute result in the limit where smoothness dominates
        #

        z = interpnd.estimate_smoothing_nd_global(tri, values,
                                                  scale=qtol**(-1./3))
        v = z[:,0]
        dx = z[:,1]
        dy = z[:,2]

        # Must be equivalent to a hyperplane least squares fit
        coef = np.c_[points[:,0], points[:,1], np.ones_like(points[:,1])]
        sol, res, rank, s = np.linalg.lstsq(coef, values)

        assert_allclose(dx, sol[0], rtol=qtol, atol=qtol)
        assert_allclose(dy, sol[1], rtol=qtol, atol=qtol)
        assert_allclose(v, np.dot(coef, sol), rtol=qtol, atol=qtol)

        #
        # 2) Compute result in the limit where point values dominate
        #

        z00 = interpnd.estimate_gradients_2d_global(tri, values)
        z0 = interpnd.estimate_smoothing_nd_global(tri, values, scale=1e-4)

        # Must coincide with the gradient estimation
        assert_allclose(z0[:,0], values, rtol=1e-3, atol=1e-3)
        assert_allclose(z0[:,1:], z00, rtol=1e-3, atol=1e-3)

    
    def _check_dataset_nd(self, points, values, qtol=1e-4):
        tri = qhull.Delaunay(points)

        #
        # 1) Compute result in the limit where smoothness dominates
        #

        z = interpnd.estimate_smoothing_nd_global(tri, values,
                                                  scale=qtol**(-1./3))

        # Must be equivalent to a hyperplane least squares fit
        coef = np.c_[points, np.ones_like(points[:,1])]
        sol, res, rank, s = np.linalg.lstsq(coef, values)

        for k in xrange(tri.ndim):
            assert_allclose(z[:,k+1], sol[k], rtol=qtol, atol=qtol,
                            err_msg=str(k))
        assert_allclose(z[:,0], np.dot(coef, sol), rtol=qtol, atol=qtol)

class TestCloughTocher2DInterpolator(object):

    def _check_accuracy(self, func, x=None, tol=1e-6, **kw):
        np.random.seed(1234)
        if x is None:
            x = np.array([(0, 0), (0, 1),
                          (1, 0), (1, 1), (0.25, 0.75), (0.6, 0.8),
                          (0.5, 0.2)],
                         dtype=float)

        ip = interpnd.CloughTocher2DInterpolator(x, func(x[:,0], x[:,1]),
                                                 tol=1e-6)
        p = np.random.rand(50, 2)

        a = ip(p)
        b = func(p[:,0], p[:,1])

        try:
            assert_allclose(a, b, **kw)
        except AssertionError:
            print abs(a - b)
            print ip.grad
            raise

    def test_linear_smoketest(self):
        # Should be exact for linear functions, independent of triangulation
        funcs = [
            lambda x, y: 0*x + 1,
            lambda x, y: 0 + x,
            lambda x, y: -2 + y,
            lambda x, y: 3 + 3*x + 14.15*y,
        ]

        for j, func in enumerate(funcs):
            self._check_accuracy(func, tol=1e-13, atol=1e-7, rtol=1e-7,
                                 err_msg="Function %d" % j)

    def test_quadratic_smoketest(self):
        # Should be reasonably accurate for quadratic functions
        funcs = [
            lambda x, y: x**2,
            lambda x, y: y**2,
            lambda x, y: x**2 - y**2,
            lambda x, y: x*y,
        ]

        for j, func in enumerate(funcs):
            self._check_accuracy(func, tol=1e-9, atol=0.22, rtol=0,
                                 err_msg="Function %d" % j)

    def test_dense(self):
        # Should be more accurate for dense meshes
        funcs = [
            lambda x, y: x**2,
            lambda x, y: y**2,
            lambda x, y: x**2 - y**2,
            lambda x, y: x*y,
            lambda x, y: np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
        ]

        np.random.seed(4321) # use a different seed than the check!
        grid = np.r_[np.array([(0,0), (0,1), (1,0), (1,1)], dtype=float),
                     np.random.rand(30*30, 2)]

        for j, func in enumerate(funcs):
            self._check_accuracy(func, x=grid, tol=1e-9, atol=5e-3, rtol=1e-2,
                                 err_msg="Function %d" % j)

if __name__ == "__main__":
    run_module_suite()
