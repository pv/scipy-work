import numpy as np

import dfitpack

__all__ = ['BSpline']

class BSpline(object):
    r"""
    One-dimensional B-spline.

    Attributes
    ----------
    t : ndarray, shape (nt,)
        Knot points for the spline
    c : ndarray, shape (..., nt)
        B-spline coefficients
    k : int
        Degree of the spline

    Methods
    -------
    __call__
    integral
    derivatives
    roots

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

    __slots__ = ('t', 'c', 'k')

    def __init__(self, t, c, k):
        t = np.ascontiguousarray(t)
        c = np.ascontiguousarray(c)
        k = int(k)
        if t.ndim != 1:
            raise ValueError("Knot array has an invalid shape")
        if t.shape[0] != c.shape[-1]:
            raise ValueError("Coefficient array has an invalid shape")
        self.t = t
        self.c = c
        self.k = k

    @classmethod
    def _from_valid_tck(cls, t, c, k):
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

        if nu == 0:
            return dfitpack.splev(self.t, self.c, self.k, x)
        else:
            return dfitpack.splder(self.t, self.c, self.k, x, nu)

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
