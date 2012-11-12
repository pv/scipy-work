
""" Classes for interpolating values.
"""
from numpy import array, transpose, searchsorted,  logical_or, atleast_1d, \
     atleast_2d, meshgrid, ravel, poly1d, asarray, intp
import numpy as np

from _spline2 import splmake, spleval
import fitpack

__all__ = ['interp1d', 'interp2d', 'lagrange']

def lagrange(x, w):
    """
    Return a Lagrange interpolating polynomial.

    Given two 1-D arrays `x` and `w,` returns the Lagrange interpolating
    polynomial through the points ``(x, w)``.

    Warning: This implementation is numerically unstable. Do not expect to
    be able to use more than about 20 points even if they are chosen optimally.

    Parameters
    ----------
    x : array_like
        `x` represents the x-coordinates of a set of datapoints.
    w : array_like
        `w` represents the y-coordinates of a set of datapoints, i.e. f(`x`).

    """
    M = len(x)
    p = poly1d(0.0)
    for j in xrange(M):
        pt = poly1d(w[j])
        for k in xrange(M):
            if k == j: continue
            fac = x[j]-x[k]
            pt *= poly1d([1.0,-x[k]])/fac
        p += pt
    return p

# !! Need to find argument for keeping initialize.  If it isn't
# !! found, get rid of it!

class interp2d(object):
    """
    interp2d(x, y, z, kind='linear', copy=True, bounds_error=False,
             fill_value=nan)

    Interpolate over a 2-D grid.

    `x`, `y` and `z` are arrays of values used to approximate some function
    f: ``z = f(x, y)``. This class returns a function whose call method uses
    spline interpolation to find the value of new points.

    Methods
    -------
    __call__

    Parameters
    ----------
    x, y : 1-D ndarrays
        Arrays defining the data point coordinates.

        If the points lie on a regular grid, `x` can specify the column
        coordinates and `y` the row coordinates, for example::

          >>> x = [0,1,2];  y = [0,3]; z = [[1,2,3], [4,5,6]]

        Otherwise, x and y must specify the full coordinates for each point,
        for example::

          >>> x = [0,1,2,0,1,2];  y = [0,0,0,3,3,3]; z = [1,2,3,4,5,6]

        If `x` and `y` are multi-dimensional, they are flattened before use.

    z : 1-D ndarray
        The values of the function to interpolate at the data points. If
        `z` is a multi-dimensional array, it is flattened before use.
    kind : {'linear', 'cubic', 'quintic'}, optional
        The kind of spline interpolation to use. Default is 'linear'.
    copy : bool, optional
        If True, then data is copied, otherwise only a reference is held.
    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, an error is raised.
        If False, then `fill_value` is used.
    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. Defaults to NaN.

    See Also
    --------
    bisplrep, bisplev
        Spline interpolation based on FITPACK
    BivariateSpline : a more recent wrapper of the FITPACK routines
    interp1d

    Notes
    -----
    The minimum number of data points required along the interpolation
    axis is ``(k+1)**2``, with k=1 for linear, k=3 for cubic and k=5 for
    quintic interpolation.

    The interpolator is constructed by `bisplrep`, with a smoothing factor
    of 0. If more control over smoothing is needed, `bisplrep` should be
    used directly.

    Examples
    --------
    Construct a 2-D grid and interpolate on it:

    >>> from scipy import interpolate
    >>> x = np.arange(-5.01, 5.01, 0.25)
    >>> y = np.arange(-5.01, 5.01, 0.25)
    >>> xx, yy = np.meshgrid(x, y)
    >>> z = np.sin(xx**2+yy**2)
    >>> f = interpolate.interp2d(x, y, z, kind='cubic')

    Now use the obtained interpolation function and plot the result:

    >>> xnew = np.arange(-5.01, 5.01, 1e-2)
    >>> ynew = np.arange(-5.01, 5.01, 1e-2)
    >>> znew = f(xnew, ynew)
    >>> plt.plot(x, z[:, 0], 'ro-', xnew, znew[:, 0], 'b-')
    >>> plt.show()

    """

    def __init__(self, x, y, z, kind='linear', copy=True, bounds_error=False,
                 fill_value=np.nan):
        self.x, self.y, self.z = map(ravel, map(asarray, [x, y, z]))

        if len(self.z) == len(self.x) * len(self.y):
            self.x, self.y = meshgrid(x,y)
            self.x, self.y = map(ravel, [self.x, self.y])
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have equal lengths")
        if len(self.z) != len(self.x):
            raise ValueError("Invalid length for input z")

        try:
            kx = ky = {'linear' : 1,
                       'cubic' : 3,
                       'quintic' : 5}[kind]
        except KeyError:
            raise ValueError("Unsupported interpolation type.")

        self.tck = fitpack.bisplrep(self.x, self.y, self.z, kx=kx, ky=ky, s=0.)

    def __call__(self,x,y,dx=0,dy=0):
        """Interpolate the function.

        Parameters
        ----------
        x : 1D array
            x-coordinates of the mesh on which to interpolate.
        y : 1D array
            y-coordinates of the mesh on which to interpolate.
        dx : int >= 0, < kx
            Order of partial derivatives in x.
        dy : int >= 0, < ky
            Order of partial derivatives in y.

        Returns
        -------
        z : 2D array with shape (len(y), len(x))
            The interpolated values.

        """

        x = atleast_1d(x)
        y = atleast_1d(y)
        z = fitpack.bisplev(x, y, self.tck, dx, dy)
        z = atleast_2d(z)
        z = transpose(z)
        if len(z)==1:
            z = z[0]
        return array(z)


class interp1d(object):
    """
    interp1d(x, y, kind='linear', axis=-1, copy=True, bounds_error=True,
             fill_value=np.nan)

    Interpolate a 1-D function.

    `x` and `y` are arrays of values used to approximate some function f:
    ``y = f(x)``.  This class returns a function whose call method uses
    interpolation to find the value of new points.

    Parameters
    ----------
    x : array_like
        A 1-D array of monotonically increasing real values.
    y : array_like
        A N-D array of real or complex values. The length of `y` along the
        interpolation axis must be equal to the length of `x`.
    kind : str or int, optional
        Specifies the kind of interpolation as a string
        ('linear','nearest', 'zero', 'slinear', 'quadratic, 'cubic')
        or as an integer specifying the order of the spline interpolator
        to use. Default is 'linear'.
    axis : int, optional
        Specifies the axis of `y` along which to interpolate.
        Interpolation defaults to the last axis of `y`.
    copy : bool, optional
        If True, the class makes internal copies of x and y.
        If False, references to `x` and `y` are used. The default is to copy.
    bounds_error : bool, optional
        If True, an error is thrown any time interpolation is attempted on
        a value outside of the range of x (where extrapolation is
        necessary). If False, out of bounds values are assigned `fill_value`.
        By default, an error is raised.
    fill_value : float, optional
        If provided, then this value will be used to fill in for requested
        points outside of the data range. If not provided, then the default
        is NaN.

    See Also
    --------
    UnivariateSpline : A more recent wrapper of the FITPACK routines.
    splrep, splev
        Spline interpolation based on FITPACK.
    interp2d

    Examples
    --------
    >>> from scipy import interpolate
    >>> x = np.arange(0, 10)
    >>> y = np.exp(-x/3.0)
    >>> f = interpolate.interp1d(x, y)

    >>> xnew = np.arange(0,9, 0.1)
    >>> ynew = f(xnew)   # use interpolation function returned by `interp1d`
    >>> plt.plot(x, y, 'o', xnew, ynew, '-')
    >>> plt.show()

    """

    def __init__(self, x, y, kind='linear', axis=-1,
                 copy=True, bounds_error=True, fill_value=np.nan):
        """ Initialize a 1D linear interpolation class."""

        self.copy = copy
        self.bounds_error = bounds_error
        self.fill_value = fill_value

        if kind in ['zero', 'slinear', 'quadratic', 'cubic']:
            order = {'nearest':0, 'zero':0,'slinear':1,
                     'quadratic':2, 'cubic':3}[kind]
            kind = 'spline'
        elif isinstance(kind, int):
            order = kind
            kind = 'spline'
        elif kind not in ('linear', 'nearest'):
            raise NotImplementedError("%s is unsupported: Use fitpack "\
                                      "routines for other types." % kind)
        x = array(x, copy=self.copy)
        y = array(y, copy=self.copy)

        if x.ndim != 1:
            raise ValueError("the x array must have exactly one dimension.")
        if y.ndim == 0:
            raise ValueError("the y array must have at least one dimension.")

        # Force-cast y to a floating-point type, if it's not yet one
        if not issubclass(y.dtype.type, np.inexact):
            y = y.astype(np.float_)

        # Normalize the axis to ensure that it is positive.
        self.axis = axis % len(y.shape)
        self._kind = kind

        if kind in ('linear', 'nearest'):
            # Make a "view" of the y array that is rotated to the interpolation
            # axis.
            axes = range(y.ndim)
            del axes[self.axis]
            axes.append(self.axis)
            oriented_y = y.transpose(axes)
            minval = 2
            len_y = oriented_y.shape[-1]
            if kind == 'linear':
                self._call = self._call_linear
            elif kind == 'nearest':
                self.x_bds = (x[1:] + x[:-1]) / 2.0
                self._call = self._call_nearest
        else:
            axes = range(y.ndim)
            del axes[self.axis]
            axes.insert(0, self.axis)
            oriented_y = y.transpose(axes)
            minval = order + 1
            len_y = oriented_y.shape[0]
            self._call = self._call_spline
            self._spline = splmake(x,oriented_y,order=order)

        len_x = len(x)
        if len_x != len_y:
            raise ValueError("x and y arrays must be equal in length along "
                             "interpolation axis.")
        if len_x < minval:
            raise ValueError("x and y arrays must have at "
                             "least %d entries" % minval)
        self.x = x
        self.y = oriented_y

    def _call_linear(self, x_new):

        # 2. Find where in the orignal data, the values to interpolate
        #    would be inserted.
        #    Note: If x_new[n] == x[m], then m is returned by searchsorted.
        x_new_indices = searchsorted(self.x, x_new)

        # 3. Clip x_new_indices so that they are within the range of
        #    self.x indices and at least 1.  Removes mis-interpolation
        #    of x_new[n] = x[0]
        x_new_indices = x_new_indices.clip(1, len(self.x)-1).astype(int)

        # 4. Calculate the slope of regions that each x_new value falls in.
        lo = x_new_indices - 1
        hi = x_new_indices

        x_lo = self.x[lo]
        x_hi = self.x[hi]
        y_lo = self.y[..., lo]
        y_hi = self.y[..., hi]

        # Note that the following two expressions rely on the specifics of the
        # broadcasting semantics.
        slope = (y_hi-y_lo) / (x_hi-x_lo)

        # 5. Calculate the actual value for each entry in x_new.
        y_new = slope*(x_new-x_lo) + y_lo

        return y_new

    def _call_nearest(self, x_new):
        """ Find nearest neighbour interpolated y_new = f(x_new)."""

        # 2. Find where in the averaged data the values to interpolate
        #    would be inserted.
        #    Note: use side='left' (right) to searchsorted() to define the
        #    halfway point to be nearest to the left (right) neighbour
        x_new_indices = searchsorted(self.x_bds, x_new, side='left')

        # 3. Clip x_new_indices so that they are within the range of x indices.
        x_new_indices = x_new_indices.clip(0,  len(self.x)-1).astype(intp)

        # 4. Calculate the actual value for each entry in x_new.
        y_new = self.y[..., x_new_indices]

        return y_new

    def _call_spline(self, x_new):
        x_new =np.asarray(x_new)
        result = spleval(self._spline,x_new.ravel())
        return result.reshape(x_new.shape+result.shape[1:])

    def __call__(self, x_new):
        """Find interpolated y_new = f(x_new).

        Parameters
        ----------
        x_new : number or array
            New independent variable(s).

        Returns
        -------
        y_new : ndarray
            Interpolated value(s) corresponding to x_new.

        """

        # 1. Handle values in x_new that are outside of x.  Throw error,
        #    or return a list of mask array indicating the outofbounds values.
        #    The behavior is set by the bounds_error variable.
        x_new = asarray(x_new)
        out_of_bounds = self._check_bounds(x_new)

        y_new = self._call(x_new)

        # Rotate the values of y_new back so that they correspond to the
        # correct x_new values. For N-D x_new, take the last (for linear)
        # or first (for other splines) N axes
        # from y_new and insert them where self.axis was in the list of axes.
        nx = x_new.ndim
        ny = y_new.ndim

        # 6. Fill any values that were out of bounds with fill_value.
        # and
        # 7. Rotate the values back to their proper place.

        if nx == 0:
            # special case: x is a scalar
            if out_of_bounds:
                if ny == 0:
                    return asarray(self.fill_value)
                else:
                    y_new[...] = self.fill_value
            return asarray(y_new)
        elif self._kind in ('linear', 'nearest'):
            y_new[..., out_of_bounds] = self.fill_value
            axes = range(ny - nx)
            axes[self.axis:self.axis] = range(ny - nx, ny)
            return y_new.transpose(axes)
        else:
            y_new[out_of_bounds] = self.fill_value
            axes = range(nx, ny)
            axes[self.axis:self.axis] = range(nx)
            return y_new.transpose(axes)

    def _check_bounds(self, x_new):
        """Check the inputs for being in the bounds of the interpolated data.

        Parameters
        ----------
        x_new : array

        Returns
        -------
        out_of_bounds : bool array
            The mask on x_new of values that are out of the bounds.
        """

        # If self.bounds_error is True, we raise an error if any x_new values
        # fall outside the range of x.  Otherwise, we return an array indicating
        # which values are outside the boundary region.
        below_bounds = x_new < self.x[0]
        above_bounds = x_new > self.x[-1]

        # !! Could provide more information about which values are out of bounds
        if self.bounds_error and below_bounds.any():
            raise ValueError("A value in x_new is below the interpolation "
                "range.")
        if self.bounds_error and above_bounds.any():
            raise ValueError("A value in x_new is above the interpolation "
                "range.")

        # !! Should we emit a warning if some values are out of bounds?
        # !! matlab does not.
        out_of_bounds = logical_or(below_bounds, above_bounds)
        return out_of_bounds
