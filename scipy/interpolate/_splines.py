from __future__ import division, print_function, absolute_import

import numpy as np

from .interpolate import prod
from . import dfitpack

__all__ = ['BSpline']

class BSpline(object):
    """
    One-dimensional B-spline.

    Parameters
    ----------
    t : ndarray, shape (n,)
        Spline knots. These must be sorted in increasing order.
    c : ndarray, shape (n, ...)
        B-spline coefficients.
    k : int
        Order of spline
    extrapolate : bool, optional
        Whether to extrapolate to ouf-of-bounds points based on first
        and last intervals, or to return NaNs. Default: True.

    Attributes
    ----------
    t : ndarray, shape (n,)
        Spline knots. These must be sorted in increasing order.
    c : ndarray, shape (n, ...)
        B-spline coefficients.
    k : int
        Order of spline

    Methods
    -------
    __call__
    derivative
    antiderivative
    integrate
    roots
    extend
    from_spline
    from_bernstein_basis
    construct_fast
    integral
    derivatives
    roots

    See also
    --------
    PPoly : piecewise polynomials

    Notes
    -----
    The spline is represented in terms of the standard B-spline basis
    functions.  In short, a spline of degree ``k`` is represented in
    terms of the knots ``t`` and coefficients ``c`` by:

    .. math::

        s(x) = \sum_{j=0}^N c_{j} B^k_{j}(x)
        \\
        B^0_i(x) = 1, \text{if $t_i \le x < t_{i+1}$, otherwise $0$,}
        \\
        B^k_i(x) = \frac{x - t_i}{t_{i+k} - t_i} B^{k-1}_i(x)
                 + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B^{k-1}_{i+1}(x)

    Or, in terms of Python code:

    >>> def bspline(x, t, c, k):
    ...     i = np.searchsorted(t, x) - 1
    ...     assert t[i] <= x <= t[i+1]
    ...     assert i >= k and i < len(t) - k
    ...     s = sum(c[i-j] * B(x, k, i-j, t) for j in range(0, k+1))
    ...     return s

    >>> def B(x, k, i, t):
    ...     if k == 0:
    ...         return 1.0 if t[i] <= x < t[i+1] else 0.0
    ...     if t[i+k] == t[i]:
    ...         c1 = 0.0
    ...     else:
    ...         c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    ...     if t[i+k+1] == t[i+1]:
    ...         c2 = 0.0
    ...     else:
    ...         c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    ...     return c1 + c2

    Note that this is an inefficient (if straightforward) way to
    evaluate B-splines --- this spline class does it in a more
    efficient way:

    >>> spline = BSpline(t=[0, 0, 1, 2, 3, 4, 4], c=[1, 2, 1, 2, 1, 2, 1], k=2)
    >>> spline(2.7)
    array([ 1.29])
    >>> bspline(2.7, spline.t, spline.c, spline.k)
    1.29

    """

    __slots__ = ('t', 'c', 'k', 'extrapolate')

    def __init__(self, t, c, k, extrapolate=None):
        self.t = np.ascontiguousarray(t, dtype=np.float64)
        self.c = np.asarray(c)
        self.k = int(k)

        if extrapolate is None:
            extrapolate = True
        self.extrapolate = bool(extrapolate)

        if not k >= 0:
            raise ValueError("k must be non-negative")
        if self.t.ndim != 1:
            raise ValueError("t must be 1-dimensional")
        if self.t.size < 2:
            raise ValueError("at least 2 knots are needed")
        if self.c.ndim < 1:
            raise ValueError("c must have at least 1 dimensions")
        if self.c.shape[0] != self.t.size:
            raise ValueError("number of coefficients != len(x)")
        if np.any(self.t[1:] - self.t[:-1] < 0):
            raise ValueError("knots are not in increasing order")

        dtype = self._get_dtype(self.c.dtype)
        self.c = np.ascontiguousarray(self.c, dtype=dtype)

    def _get_dtype(self, dtype):
        if np.issubdtype(dtype, np.complexfloating) \
               or np.issubdtype(self.c.dtype, np.complexfloating):
            return np.complex_
        else:
            return np.float_

    @classmethod
    def construct_fast(cls, t, c, k, extrapolate=None):
        """
        Construct the piecewise polynomial without making checks.

        Takes the same parameters as the constructor. Input arguments
        `c` and `x` must be arrays of the correct shape and type.  The
        `c` array can only be of dtypes float and complex, and `x`
        array must have dtype float.

        """
        self = object.__new__(cls)
        self.t = t
        self.c = c
        self.k = k
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = extrapolate
        return self

    def __len__(self):
        # so that tck unpacking works
        return 3

    def __getitem__(self, i):
        return (self.t, self.c, self.k)[i]

    def __call__(self, x, nu=0, extrapolate=None):
        """
        Evaluate the piecewise polynomial or its derivative

        Parameters
        ----------
        x : array-like
            Points to evaluate the interpolant at.
        nu : int, optional
            Order of derivative to evaluate. Must be non-negative.
        extrapolate : bool, optional
            Whether to extrapolate to ouf-of-bounds points based on first
            and last intervals, or to return NaNs.

        Returns
        -------
        y : array-like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.

        """
        if extrapolate is None:
            extrapolate = self.extrapolate

        x = np.asarray(x)

        if x.size == 0:
            return np.zeros(x.shape + self.c.shape[1:], dtype=self.c.dtype)

        if np.issubdtype(self.c.dtype, np.complexfloating):
            cx = self.c.real.reshape(self.c.shape[0], -1)

            r = np.empty((x.size, prod(self.c.shape[1:])), dtype=self.c.dtype)

            cx = self.c.real.reshape(self.c.shape[0], -1)
            r.real, ier = dfitpack.splder_many(self.t, cx, self.k, x.ravel(), nu)
            if ier != 0:
                raise RuntimeError("spline interpolation failed")

            cx = self.c.imag.reshape(self.c.shape[0], -1)
            r.imag, ier = dfitpack.splder_many(self.t, cx, self.k, x.ravel(), nu)
            if ier != 0:
                raise RuntimeError("spline interpolation failed")
        else:
            cx = self.c.reshape(self.c.shape[0], -1)
            r, ier = dfitpack.splder_many(self.t, cx, self.k, x.ravel(), nu)
            if ier != 0:
                raise RuntimeError("spline interpolation failed")
            
        r.shape = x.shape + self.c.shape[1:]
        return r

    def _ensure_c_contiguous(self):
        """
        c and x may be modified by the user. The Cython code expects
        that they are C contiguous.
        """
        if not self.t.flags.c_contiguous:
            self.t = self.t.copy()
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()

    def integrate(self, a, b, extrapolate=None):
        """
        Compute a definite integral over a piecewise polynomial.

        Parameters
        ----------
        a : float
            Lower integration bound
        b : float
            Upper integration bound
        extrapolate : bool, optional
            Whether to extrapolate to ouf-of-bounds points based on first
            and last intervals, or to return NaNs.

        Returns
        -------
        ig : array_like
            Definite integral of the piecewise polynomial over [a, b]

        """
        if extrapolate is None:
            extrapolate = self.extrapolate

        return dfitpack.splint(*(self.t, self.c, self.k, a, b))

    def roots(self, discontinuity=True, extrapolate=None):
        """
        Find real roots of the piecewise polynomial.

        Parameters
        ----------
        discontinuity : bool, optional
            Whether to report sign changes across discontinuities at
            breakpoints as roots.
        extrapolate : bool, optional
            Whether to return roots from the polynomial extrapolated
            based on first and last intervals.

        Returns
        -------
        roots : ndarray
            Roots of the polynomial(s).

            If the PPoly object describes multiple polynomials, the
            return value is an object array whose each element is an
            ndarray containing the roots.

        Notes
        -----
        This routine works only on real-valued polynomials.

        If the piecewise polynomial contains sections that are
        identically zero, the root list will contain the start point
        of the corresponding interval, followed by a ``nan`` value.

        If the polynomial is discontinuous across a breakpoint, and
        there is a sign change across the breakpoint, this is reported
        if the `discont` parameter is True.

        Examples
        --------

        Finding roots of ``[x**2 - 1, (x - 1)**2]`` defined on intervals
        ``[-2, 1], [1, 2]``:

        >>> from scipy.interpolate import PPoly
        >>> pp = PPoly(np.array([[1, 0, -1], [1, 0, 0]]).T, [-2, 1, 2])
        >>> pp.roots()
        array([-1.,  1.])

        """
        if extrapolate is None:
            extrapolate = self.extrapolate

        if self.k == 3:
            z, m, ier = dfitpack.sproot(*self._tck[:2])
            if not ier == 0:
                raise ValueError("Error code returned by spalde: %s" % ier)
            return z[:m]

        raise NotImplementedError('finding roots unsupported for '
                                  'non-cubic splines')


    @classmethod
    def fit_free(cls, x, y, k=3):
        """
        Construct an interpolating spline to the data with free b.c.

        The natural boundary conditions imply the k-1 derivative of
        the spline at the endpoints vanishes.

        Parameters
        ----------
        x : array-like, shape (n,)
            1D array of points. Must be sorted in increasing order.
        y : array-like, shape (n, ...)
            Data values
        k : int, optional
            Order of the spline

        Returns
        -------
        spl : BSpline
            B-Spline that interpolates through the data points

        """
        x = np.asarray(x)
        y = np.asarray(y)

        # Check inputs
        if x.ndim != 1:
            raise ValueError("x-array must be 1-dimensional")
        if y.ndim < 1 or y.shape[0] != x.shape[0]:
            raise ValueError("shape of y array does not match x")

        # Construct a suitable knot set
        if k > 0:
            xp = np.unique(x)
            xb = np.repeat(xp[0], k+1)
            xe = np.repeat(xp[-1], k+1)
            t = np.r_[xb, xp[(k-1):-(k-1)], xe]
        else:
            t = x

        # LSQ spline --- enough D.O.F. to interpolate exactly
        return cls.fit_leastsq(x, y, t, k=k)


    @classmethod
    def fit_leastsq(cls, x, y, t, k=3):
        """
        Fit a least-squares spline to the data, with given knots

        Parameters
        ----------
        x : array-like, shape (n,)
            1D array of points. Must be sorted in increasing order.
        y : array-like, shape (n, ...)
            Data values
        t : array-like, shape (m,)
            1D array of internal knot points. They need to satisfy the
            conditions:

            - sorted in increasing order
            - at most `len(x) - 2*k`

        k : int, optional
            Order of the spline

        Returns
        -------
        spl : BSpline
            B-Spline that interpolates through the data points

        """
        x = np.asarray(x)
        y = np.asarray(y)

        # Check inputs
        if x.ndim != 1:
            raise ValueError("x-array must be 1-dimensional")
        if y.ndim < 1 or y.shape[0] != x.shape[0]:
            raise ValueError("shape of y array does not match x")

        # Check knots
        ier = dfitpack.fpchec(x, t, k)
        if ier != 0:
            raise RuntimeError("Invalid set of knots t")

        # LSQ spline --- enough D.O.F. to interpolate exactly
        def ev(y):
            _, _, _, _, _, _, _, _, _, c, fp, _, _, ier = \
                dfitpack.fpcurfm1(x, y, k, t, w=None, xb=x[0], xe=x[-1])
            if ier != 0 or abs(fp) > 1e-10:
                raise RuntimeError("Spline fitting failed: %r" % ((ier, fp),))
            return c

        if np.issubdtype(y.dtype, np.complexfloating):
            c = np.empty((t.size,) + y.shape[1:], dtype=y.dtype)
            c.real = np.apply_along_axis(ev, 0, y.real)
            c.imag = np.apply_along_axis(ev, 0, y.imag)
        else:
            c = np.apply_along_axis(ev, 0, y)

        return cls.construct_fast(t, c, k)
