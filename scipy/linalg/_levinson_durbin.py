"""
Implementation of the Levinson-Durbin algorithm for solving Toeplitz systems.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.linalg


__all__ = ['solve_toeplitz']


def solve_toeplitz(c, r=None, y=None):
    """
    Solve the matrix equation (T x = y) where T is a Toeplitz matrix.

    The square Toeplitz input matrix T is represented through its
    first column and optionally its first row.
    This representation, including the ordering of the first two args,
    is taken from the linalg.toeplitz function.

    Parameters
    ----------
    c : 1d array_like
        The first column of the Toeplitz matrix.
    r : 1d array_like, optional
        The first row of the Toeplitz matrix.
    y : array_like
        The rhs of the matrix equation.

    Returns
    -------
    x : ndarray
        The solution of the matrix equation (T x = y).

    Notes
    -----
    This is an implementation of the Levinson-Durbin algorithm which uses
    only O(N) memory and O(N^2) time, but which has less than stellar
    numerical stability.

    References
    ----------
    .. [1] Wikipedia, "Levinson recursion",
           http://en.wikipedia.org/wiki/Levinson_recursion

    """
    # This block has been copied from the linalg.toeplitz construction function.
    c = np.asarray(c).ravel()
    if r is None:
        r = c.conjugate()
    else:
        r = np.asarray(r).ravel()

    # Check that the rhs makes sense.
    if y is None:
        raise ValueError('missing rhs')
    y = np.asarray(y)
    y_shape = y.shape
    N = y.shape[0]
    y = y.reshape(N, -1)

    # Check that the Toeplitz representation makes sense
    # and is compatible with the rhs shape.
    if c.shape != r.shape:
        raise ValueError('expected the Toeplitz matrix to be square')
    if c.shape != (N,):
        raise ValueError('the rhs shape is incompatible with the matrix shape')

    # If the diagonal is zero, then the Levinson-Durbin implementation fails.
    if not c[0]:
        raise np.linalg.LinAlgError(
                'the scipy implementation of the Levinson-Durbin algorithm '
                'fails when the main diagonal is zero')

    # Key relating entries of the toeplitz matrix to entries of c, r,
    # assuming n is a positive integer less than N:
    # M[0, 0] == c[0]
    # M[n, :n] == c[n:0:-1]
    # M[0, 1:n+1] == r[1:n+1]

    # If any of the input arrays are complex then use complex dtype.
    # Otherwise use real dtype.
    if any(np.iscomplexobj(obj) for obj in (c, r, y)):
        mytype = np.complex128
    else:
        mytype = np.float64

    # Grab BLAS routines; these are slightly faster than the
    # corresponding Numpy operations, and using them avoids
    # temporaries
    axpy = scipy.linalg.get_blas_funcs('axpy', dtype=mytype)
    if mytype == np.complex128:
        ger = scipy.linalg.get_blas_funcs('geru', dtype=mytype)
        xdot = scipy.linalg.get_blas_funcs('dotu', dtype=mytype)
    else:
        ger = scipy.linalg.get_blas_funcs('ger', dtype=mytype)
        xdot = scipy.linalg.get_blas_funcs('dot', dtype=mytype)

    # Initialize the forward, backward, and solution vectors.
    y = y.astype(mytype)
    f_prev = np.zeros(N, dtype=mytype)
    b_prev = np.zeros(N, dtype=mytype)
    x_prev = np.zeros(y.shape, dtype=mytype)
    f = np.zeros(N, dtype=mytype)
    b = np.zeros(N, dtype=mytype)
    x = np.zeros(y.shape, dtype=mytype)
    f[0] = 1 / c[0]
    b[0] = 1 / c[0]
    x[0] = y[0] / c[0]

    # Compute forward, backward, and solution vectors recursively.
    for n in range(1, N):
        f, f_prev = f_prev, f
        b, b_prev = b_prev, b
        x, x_prev = x_prev, x
        eps_f = xdot(c[n:0:-1], f_prev[:n])
        eps_x = np.dot(c[n:0:-1], x_prev[:n])
        eps_b = xdot(r[1:n+1], b_prev[:n])
        f.fill(0)
        b.fill(0)
        denom = 1 - eps_f * eps_b

        # Complain if the denominator is exactly zero.
        # For better numerical stability, maybe use a different algorithm.
        if not denom:
            raise np.linalg.LinAlgError(
                    'the Levinson-Durbin algorithm '
                    'failed to solve the matrix equation')

        coeff = 1 / denom

        # f[:n] += coeff * f_prev[:n]
        f[:n] = axpy(f_prev[:n], f[:n], a=coeff)

        # f[1:n+1] -= coeff * eps_f * b_prev[:n]
        f[1:n+1] = axpy(b_prev[:n], f[1:n+1], a=-coeff*eps_f)

        # b[1:n+1] += coeff * b_prev[:n]
        b[1:n+1] = axpy(b_prev[:n], b[1:n+1], a=coeff)

        # b[:n] -= coeff * eps_b * f_prev[:n]
        b[:n] = axpy(f_prev[:n], b[:n], a=-coeff*eps_b)

        # x[:n+1] = x_prev[:n+1] + b[:n+1,None] * (y[n] - eps_x)
        np.subtract(y[n], eps_x, out=eps_x)
        x[:n+1] = ger(1.0, y=b[:n+1], x=eps_x, a=x_prev[:n+1].T,
                      overwrite_x=True, overwrite_y=False, overwrite_a=True).T

    return x.reshape((x.shape[0],) + y_shape[1:])

