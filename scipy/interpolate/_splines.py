import numpy as np

import dfitpack
import _spline_low


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
        if self.c.shape[0] != self.x.size:
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

    @classmethod
    def _from_valid_tck(cls, t, c, k, c_shape):
        self = cls.__new__(cls)
        self.t = t
        self.c = c
        self.k = k
        return self

    def __len__(self):
        # so that tck unpacking works
        return 3

    def __getitem__(self, i):
        return (self.t, self.c, self.k)[i]

    def __call__(self, x, nu=0):
        """ Evaluate spline (or its nu-th derivative) at positions x.

        Note: x can be unordered but the evaluation is more efficient
        if x is (partially) ordered.
        """
        x = np.asarray(x)

        # empty input yields empty output
        if x.size == 0:
            return np.array([])
        r, ier = dfitpack.splder_grid(self.t,
                                      self.c.reshape(-1, self.c.shape[-1]).T,
                                      self.k,
                                      x.ravel(),
                                      nu)
        r.shape = self.c.shape[:-1] + x.shape
        return r

    def integral(self, a, b):
        """ Return definite integral of the spline between two given points.
        """
        return dfitpack.splint(*(self._tck+(a,b)))

    def derivatives(self, x):
        """ Return all derivatives of the spline at the point x."""
        d,ier = dfitpack.spalde(*(self._tck+(x,)))
        if not ier == 0:
            raise ValueError("Error code returned by spalde: %s" % ier)
        return d

    def roots(self):
        """ Return the zeros of the spline.

        Restriction: only cubic splines are supported by fitpack.
        """
        if self.k == 3:
            z, m, ier = dfitpack.sproot(*self._tck[:2])
            if not ier == 0:
                raise ValueError("Error code returned by spalde: %s" % ier)
            return z[:m]
        raise NotImplementedError('finding roots unsupported for '
                                  'non-cubic splines')



    def _ensure_c_contiguous(self):
        """
        c and x may be modified by the user. The Cython code expects
        that they are C contiguous.
        """
        if not self.x.flags.c_contiguous:
            self.x = self.x.copy()
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()

    def extend(self, c, x, right=True):
        """
        Add additional breakpoints and coefficients to the polynomial.

        Parameters
        ----------
        c : ndarray, size (k, m, ...)
            Additional coefficients for polynomials in intervals
            ``self.x[-1] <= x < x_right[0]``, ``x_right[0] <= x < x_right[1]``,
            ..., ``x_right[m-2] <= x < x_right[m-1]``
        x : ndarray, size (m,)
            Additional breakpoints. Must be sorted and either to
            the right or to the left of the current breakpoints.
        right : bool, optional
            Whether the new intervals are to the right or to the left
            of the current intervals.

        """
        c = np.asarray(c)
        x = np.asarray(x)
        
        if c.ndim < 2:
            raise ValueError("invalid dimensions for c")
        if x.ndim != 1:
            raise ValueError("invalid dimensions for x")
        if x.shape[0] != c.shape[1]:
            raise ValueError("x and c have incompatible sizes")
        if c.shape[2:] != self.c.shape[2:] or c.ndim != self.c.ndim:
            raise ValueError("c and self.c have incompatible shapes")
        if right:
            if x[0] < self.x[-1]:
                raise ValueError("new x are not to the right of current ones")
        else:
            if x[-1] > self.x[0]:
                raise ValueError("new x are not to the left of current ones")

        if c.size == 0:
            return

        dtype = self._get_dtype(c.dtype)

        k2 = max(c.shape[0], self.c.shape[0])
        c2 = np.zeros((k2, self.c.shape[1] + c.shape[1]) + self.c.shape[2:],
                      dtype=dtype)

        if right:
            c2[k2-self.c.shape[0]:, :self.c.shape[1]] = self.c
            c2[k2-c.shape[0]:, self.c.shape[1]:] = c
            self.x = np.r_[self.x, x]
        else:
            c2[k2-self.c.shape[0]:, :c.shape[1]] = c
            c2[k2-c.shape[0]:, c.shape[1]:] = self.c
            self.x = np.r_[x, self.x]

        self.c = c2

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
        x_shape = x.shape
        x = np.ascontiguousarray(x.ravel(), dtype=np.float_)
        out = np.empty((len(x), prod(self.c.shape[2:])), dtype=self.c.dtype)
        self._ensure_c_contiguous()
        self._evaluate(x, nu, extrapolate, out)
        return out.reshape(x_shape + self.c.shape[2:])

    def _evaluate(self, x, nu, extrapolate, out):
        _ppoly.evaluate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                        self.x, x, nu, bool(extrapolate), out)

    def derivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the derivative.

        Parameters
        ----------
        n : int, optional
            Order of derivative to evaluate. (Default: 1)
            If negative, the antiderivative is returned.

        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k - n representing the derivative
            of this polynomial.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.

        """
        if nu < 0:
            return self.antiderivative(-nu)

        # reduce order
        if nu == 0:
            c2 = self.c.copy()
        else:
            c2 = self.c[:-nu,:].copy()

        if c2.shape[0] == 0:
            # derivative of order 0 is zero
            c2 = np.zeros((1,) + c2.shape[1:], dtype=c2.dtype)

        # multiply by the correct rising factorials
        factor = spec.poch(np.arange(c2.shape[0], 0, -1), nu)
        c2 *= factor[(slice(None),) + (None,)*(c2.ndim-1)]

        # construct a compatible polynomial
        return self.construct_fast(c2, self.x, self.extrapolate)

    def antiderivative(self, nu=1):
        """
        Construct a new piecewise polynomial representing the antiderivative.

        Antiderivativative is also the indefinite integral of the function,
        and derivative is its inverse operation.

        Parameters
        ----------
        n : int, optional
            Order of antiderivative to evaluate. (Default: 1)
            If negative, the derivative is returned.

        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k + n representing
            the antiderivative of this polynomial.

        Notes
        -----
        The antiderivative returned by this function is continuous and
        continuously differentiable to order n-1, up to floating point
        rounding error.

        """
        if nu <= 0:
            return self.derivative(-nu)

        c = np.zeros((self.c.shape[0] + nu, self.c.shape[1]) + self.c.shape[2:],
                     dtype=self.c.dtype)
        c[:-nu] = self.c

        # divide by the correct rising factorials
        factor = spec.poch(np.arange(self.c.shape[0], 0, -1), nu)
        c[:-nu] /= factor[(slice(None),) + (None,)*(c.ndim-1)]

        # fix continuity of added degrees of freedom
        self._ensure_c_contiguous()
        _ppoly.fix_continuity(c.reshape(c.shape[0], c.shape[1], -1),
                              self.x, nu)

        # construct a compatible polynomial
        return self.construct_fast(c, self.x, self.extrapolate)

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

        # Swap integration bounds if needed
        sign = 1
        if b < a:
            a, b = b, a
            sign = -1

        # Compute the integral
        range_int = np.empty((prod(self.c.shape[2:]),), dtype=self.c.dtype)
        self._ensure_c_contiguous()
        _ppoly.integrate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                         self.x, a, b, bool(extrapolate),
                         out=range_int)

        # Return
        range_int *= sign
        return range_int.reshape(self.c.shape[2:])

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

        self._ensure_c_contiguous()

        if np.issubdtype(self.c.dtype, np.complexfloating):
            raise ValueError("Root finding is only for "
                             "real-valued polynomials")

        r = _ppoly.real_roots(self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                              self.x, bool(discontinuity),
                              bool(extrapolate))
        if self.c.ndim == 2:
            return r[0]
        else:
            r2 = np.empty(prod(self.c.shape[2:]), dtype=object)
            r2[...] = r
            return r2.reshape(self.c.shape[2:])

    @classmethod
    def from_spline(cls, tck, extrapolate=None):
        """
        Construct a piecewise polynomial from a spline

        Parameters
        ----------
        tck
            A spline, as returned by `splrep`
        extrapolate : bool, optional
            Whether to extrapolate to ouf-of-bounds points based on first
            and last intervals, or to return NaNs. Default: True.

        """
        t, c, k = tck

        cvals = np.empty((k + 1, len(t)-1), dtype=c.dtype)
        for m in xrange(k, -1, -1):
            y = fitpack.splev(t[:-1], tck, der=m)
            cvals[k - m, :] = y/spec.gamma(m+1)

        return cls.construct_fast(cvals, t, extrapolate)

    @classmethod
    def from_bernstein_basis(cls, bp, extrapolate=None):
        """
        Construct a piecewise polynomial in the power basis
        from a polynomial in Bernstein basis.

        Parameters
        ----------
        bp : BPoly
            A Bernstein basis polynomial, as created by BPoly
        extrapolate : bool, optional
            Whether to extrapolate to ouf-of-bounds points based on first
            and last intervals, or to return NaNs. Default: True.

        """
        dx = np.diff(bp.x)
        k = bp.c.shape[0] - 1  # polynomial order

        rest = (None,)*(bp.c.ndim-2)

        c = np.zeros_like(bp.c)
        for a in range(k+1):
            factor = (-1)**(a) * comb(k, a) * bp.c[a]
            for s in range(a, k+1):
                val = comb(k-a, s-a) * (-1)**s
                c[k-s] += factor * val / dx[(slice(None),)+rest]**s

        if extrapolate is None:
            extrapolate = bp.extrapolate

        return cls.construct_fast(c, bp.x, extrapolate)


def _naive_B(x, k, i, t):
    """
    Naive way to compute B-spline basis functions. Useful only for testing!
    """

    if k == 0:
        return 1.0 if t[i] <= x < t[i+1] else 0.0

    if t[i+k] == t[i]:
        c1 = 0.0
    else:
        c1 = (x - t[i])/(t[i+k] - t[i]) * _naive_B(x, k-1, i, t)

    if t[i+k+1] == t[i+1]:
        c2 = 0.0
    else:
        c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * _naive_B(x, k-1, i+1, t)

    return (c1 + c2)

def _naive_eval(x, t, c, k):
    """
    Naive B-spline evaluation. Useful only for testing!
    """
    i = np.searchsorted(t, x) - 1
    assert t[i] <= x <= t[i+1]
    assert i >= k and i < len(t) - k
    return sum(c[i-j] * _naive_B(x, k, i-j, t) for j in range(0, k+1))
