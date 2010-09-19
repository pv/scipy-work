"""
Convenience interface to N-D interpolation

.. versionadded:: 0.9

"""

import numpy as np
from interpnd import LinearNDInterpolator, NDInterpolatorBase, \
     CloughTocher2DInterpolator, _ndim_coords_from_arrays, \
     estimate_smoothing_nd_global
from scipy.spatial import cKDTree, Delaunay

__all__ = ['griddata', 'NearestNDInterpolator', 'LinearNDInterpolator',
           'CloughTocher2DInterpolator', 'NDSmoother', 'smoothdata']

#------------------------------------------------------------------------------
# Nearest-neighbour interpolation
#------------------------------------------------------------------------------

class NearestNDInterpolator(NDInterpolatorBase):
    """
    NearestNDInterpolator(points, values)

    Nearest-neighbour interpolation in N dimensions.

    .. versionadded:: 0.9

    Parameters
    ----------
    points : ndarray of floats, shape (npoints, ndims)
        Data point coordinates.
    values : ndarray of float or complex, shape (npoints, ...)
        Data values.

    Notes
    -----
    Uses ``scipy.spatial.cKDTree``

    """

    def __init__(self, x, y):
        x = _ndim_coords_from_arrays(x)
        self._check_init_shape(x, y)
        self.tree = cKDTree(x)
        self.points = x
        self.values = y

    def __call__(self, xi):
        """
        Evaluate interpolator at given points.

        Parameters
        ----------
        xi : ndarray of float, shape (..., ndim)
            Points where to interpolate data at.

        """
        xi = self._check_call_shape(xi)
        dist, i = self.tree.query(xi)
        return self.values[i]


#------------------------------------------------------------------------------
# Convenience interface function for interpolation
#------------------------------------------------------------------------------

def griddata(points, values, xi, method='linear', fill_value=np.nan):
    """
    Interpolate unstructured N-dimensional data.

    .. versionadded:: 0.9

    Parameters
    ----------
    points : ndarray of floats, shape (npoints, ndims)
        Data point coordinates. Can either be a ndarray of
        size (npoints, ndim), or a tuple of `ndim` arrays.
    values : ndarray of float or complex, shape (npoints, ...)
        Data values.
    xi : ndarray of float, shape (..., ndim)
        Points where to interpolate data at.

    method : {'linear', 'nearest', 'cubic'}
        Method of interpolation. One of

        - ``nearest``: return the value at the data point closest to
          the point of interpolation.  See `NearestNDInterpolator` for
          more details.

        - ``linear``: tesselate the input point set to n-dimensional
          simplices, and interpolate linearly on each simplex.  See
          `LinearNDInterpolator` for more details.

        - ``cubic`` (1-D): return the value detemined from a cubic
          spline.

        - ``cubic`` (2-D): return the value determined from a
          piecewise cubic, continuously differentiable (C1), and
          approximately curvature-minimizing polynomial surface. See
          `CloughTocher2DInterpolator` for more details.

    fill_value : float, optional
        Value used to fill in for requested points outside of the
        convex hull of the input points.  If not provided, then the
        default is ``nan``. This option has no effect for the
        'nearest' method.


    Examples
    --------

    Suppose we want to interpolate the 2-D function

    >>> def func(x, y):
    >>>     return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

    on a grid in [0, 1]x[0, 1]

    >>> grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

    but we only know its values at 1000 data points:

    >>> points = np.random.rand(1000, 2)
    >>> values = func(points[:,0], points[:,1])

    This can be done with `griddata` -- below we try out all of the
    interpolation methods:

    >>> from scipy.interpolate import griddata
    >>> grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
    >>> grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
    >>> grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

    One can see that the exact result is reproduced by all of the
    methods to some degree, but for this smooth function the piecewise
    cubic interpolant gives the best results:

    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(221)
    >>> plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
    >>> plt.plot(points[:,0], points[:,1], 'k.', ms=1)
    >>> plt.title('Original')
    >>> plt.subplot(222)
    >>> plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
    >>> plt.title('Nearest')
    >>> plt.subplot(223)
    >>> plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
    >>> plt.title('Linear')
    >>> plt.subplot(224)
    >>> plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
    >>> plt.title('Cubic')
    >>> plt.gcf().set_size_inches(6, 6)
    >>> plt.show()

    """

    points = _ndim_coords_from_arrays(points)
    xi = _ndim_coords_from_arrays(xi)

    ndim = points.shape[-1]

    if ndim == 1 and method in ('nearest', 'linear', 'cubic'):
        ip = interp1d(points, values, kind=method, axis=0, bounds_error=False,
                      fill_value=fill_value)
        return ip(xi)
    elif method == 'nearest':
        ip = NearestNDInterpolator(points, values)
        return ip(xi)
    elif method == 'linear':
        ip = LinearNDInterpolator(points, values, fill_value=fill_value)
        return ip(xi)
    elif method == 'cubic' and ndim == 2:
        ip = CloughTocher2DInterpolator(points, values, fill_value=fill_value)
        return ip(xi)
    else:
        raise ValueError("Unknown interpolation method %r for "
                         "%d dimensional data" % (method, ndim))


#------------------------------------------------------------------------------
# N-d data smoothing
#------------------------------------------------------------------------------

class NDSmoother(NDInterpolatorBase):
    """
    NDSmoother(points, values, scale=None, weights=None, fill_value=np.nan, method='linear', triangulation=None)

    Smooth unstructured N-dimensional data.

    .. versionadded:: 0.9

    The smoothing is done by fitting a smooth surface to the data, and
    minimizing::

        scale*||surface curvature||^2 + weights*||smoothed_data - data||^2

    The smooth surface is approximated with a minimum norm network in
    the computation.

    Parameters
    ----------
    points : ndarray of floats, shape (npoints, ndims)
        Data point coordinates. Can either be a ndarray of
        size (npoints, ndim), or a tuple of `ndim` arrays.
    values : ndarray of float or complex, shape (npoints, ...)
        Data values.
    scale : float or ndim-array of floats, optional
        Scaling of dimensions. If an array, specifies the scaling separately
        in each dimension.

        If the data weights are 1.0 (the default), this parameter has
        roughly the meaning of a shortest length scale to preserve in
        smoothing.
    weights : float, optional
        Weighing of the value at each data point.  These should
        typically be in the range [0, 1], zero meaning that the data
        point is completely neglected, and 1.0 that it has a standard
        weight vs. the curvature term.
    method : {'linear', 'nearest', 'cubic'}, optional
        Method of interpolation. See `griddata`.
    triangulation : Delaunay, optional
        Pre-existing triangulation of the data point set to use.

    Attributes
    ----------
    values
        Smoothed data values
    grad
        Estimated smoothed data gradients at vertices

    Methods
    -------
    __call__
        Evaluate the smoothed interpolant at a given point.

    See Also
    --------
    NDSmoother

    Examples
    --------
    Suppose we have 500 samples in 3-d in a box [0,1]x[0,1]x[0,1]:

    >>> np.random.seed(1234)
    >>> points = np.random.rand(500, 3)

    The samples contain some signal plus gaussian noise:

    >>> def func(x, y, z):
    ...     return (np.cos(3*x) + 2*y**2)*np.sin(5*x + z)

    >>> signal = func(points[:,0], points[:,1], points[:,2])
    >>> noise = 0.3*np.random.randn(500)
    >>> data = signal + noise

    We would like to smooth the noise away to recover the signal.
    This can be done with `smoothdata`:

    >>> from scipy.interpolate import NDSmoother
    >>> smoothed = NDSmoother(points, data, scale=0.1)

    We set give ``scale=0.1`` as parameter, which means that the
    smoothed data will be smooth on the length scale of 0.1, which
    hopefully eliminates the noise. The ``scale`` parameter controls
    how much curvature is allowed in the result: if it is very large,
    the result is a hyperplane (which have zero curvature), and if
    very small, no smoothing is done. We can also check what happens at
    ``scale=0.5``:

    >>> smoothed_2 = NDSmoother(points, data, scale=0.5)

    We can plot the slice at ``z=0.5`` to check how smoothing went:

    >>> from scipy.interpolate import griddata
    >>> x, y = np.mgrid[0:1:30j,0:1:30j]
    >>> data_grid = griddata(points, data, (x, y, 0.5))

    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(141)
    >>> plt.imshow(func(x, y, 0.5).T, extent=(0,1,0,1), origin='lower')
    >>> plt.title('signal')
    >>> plt.subplot(142)
    >>> plt.imshow(data_grid.T, extent=(0,1,0,1), origin='lower')
    >>> plt.title('data')
    >>> plt.subplot(143)
    >>> plt.imshow(smoothed((x, y, 0.5)).T,
    ...            extent=(0,1,0,1), origin='lower')
    >>> plt.title('scale=0.1')
    >>> plt.subplot(144)
    >>> plt.imshow(smoothed_2((x, y, 0.5)).T, extent=(0,1,0,1),
    ...            origin='lower')
    >>> plt.title('scale=0.5')
    >>> plt.show()

    """

    def __init__(self, points, values, scale=None, weights=None,
                 fill_value=np.nan, method='linear', triangulation=None):
        NDInterpolatorBase.__init__(self, points, values, fill_value=fill_value)

        scale_c = 1.0
        if scale is not None:
            if np.isscalar(scale):
                scale_c = scale
                self.scale = None
            else:
                # scaling should also affect the triangulation
                scale_c = 1.0
                self.scale = scale
                self.points = self.points.copy()
                self.points /= self.scale

        if triangulation is not None:
            if scale is not None:
                raise ValueError("Cannot re-scale when a triangulation is "
                                 "given")
            self.tri = triangulation
        else:
            self.tri = Delaunay(self.points)

        z = estimate_smoothing_nd_global(self.tri, self.values,
                                         scale=scale_c, weights=weights)
        self.values = z[:,0]
        self.grad = z[:,1:]

        if method == 'nearest':
            self.ip = NearestNDInterpolator(self.points, self.values)
        elif method == 'linear':
            self.ip = LinearNDInterpolator(
                self.points, self.values, fill_value=fill_value,
                triangulation=self.tri)
        elif method == 'cubic':
            if self.tri.ndim != 2:
                raise ValueError("Cubic interpolation is available only in 2D")
            self.ip = CloughTocher2DInterpolator(
                self.points, self.values, fill_value=fill_value,
                gradients=self.grad, triangulation=self.tri)
        else:
            raise ValueError("Unknown interpolation mehthod %r" % method)

    def _evaluate_double(self, xi):
        if self.scale is not None:
            xi = xi.copy()
            xi /= self.scale
        return self.ip(xi)

    _evaluate_complex = _evaluate_double
