from __future__ import division, print_function, absolute_import

__all__ = ['least_squares']

import numpy as np

from scipy.lib.six import callable

from warnings import warn

from .optimize import MemoizeJac, Result, _check_unknown_options
from .minpack import _leastsq_minpack
from ._lmfit import _leastsq_lmfit

def least_squares(fun, x0, args=(), method='lm', jac=None, tol=None,
                  callback=None, options=None):
    """
    Minimize the sum of squares of a set of equations.

    .. versionadded:: 0.13.0

    Parameters
    ----------
    fun : callable
        Vector function whose squared sum to minimize. Should take at least
        one (possibly length N vector) argument and return M floating point
        numbers.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function and its Jacobian.
    method : str, optional
        Type of solver.  Should be one of

            - 'lm'
            - 'lm2'

    jac : bool or callable, optional
        If `jac` is a Boolean and is True, `fun` is assumed to return the
        value of Jacobian along with the objective function. If False, the
        Jacobian will be estimated numerically.
        `jac` can also be a callable returning the Jacobian of `fun`. In
        this case, it must accept the same arguments as `fun`.
    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.
    callback : function, optional
        Optional callback function. It is called on every iteration as
        ``callback(x, f)`` where `x` is the current solution and `f`
        the corresponding residual. For all methods but 'lm'.
    options : dict, optional
        A dictionary of solver options. E.g. `xtol` or `maxfev`, see
        ``show_options('least_squares', method)`` for details.

    Returns
    -------
    sol : Result
        The solution represented as a ``Result`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the algorithm exited successfully and
        ``message`` which describes the cause of the termination. See
        `Result` for a description of other attributes.

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter. The default method is *lm*.

    Method *lm* solves the system of nonlinear equations in a least squares
    sense using a modification of the Levenberg-Marquardt algorithm as
    implemented in MINPACK [1]_.

    Method *lm2* solves the system of nonlinear equations in a least
    squares sense using a similar algorithm as *lm*, but supporting
    sparse jacobians.

    References
    ----------
    .. [1] More, Jorge J., Burton S. Garbow, and Kenneth E. Hillstrom.
       1980. User Guide for MINPACK-1.

    Examples
    --------
    FIXME: update this section

    The following functions define a system of nonlinear equations and its
    jacobian.

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    >>> def jac(x):
    ...     return np.array([[1 + 1.5 * (x[0] - x[1])**2,
    ...                       -1.5 * (x[0] - x[1])**2],
    ...                      [-1.5 * (x[1] - x[0])**2,
    ...                       1 + 1.5 * (x[1] - x[0])**2]])

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.root(fun, [0, 0], jac=jac, method='hybr')
    >>> sol.x
    array([ 0.8411639,  0.1588361])
    """
    meth = method.lower()
    if options is None:
        options = {}

    if callback is not None and meth in ('lm',):
        warn('Method %s does not accept callback.' % method,
             RuntimeWarning)

    # fun also returns the jacobian
    if not callable(jac) and meth in ('lm', 'lm2'):
        if bool(jac):
            fun = MemoizeJac(fun)
            jac = fun.derivative
        else:
            jac = None

    # set default tolerances
    if tol is not None:
        options = dict(options)
        if meth in ('lm', 'lm2'):
            options.setdefault('xtol', tol)

    # run
    if meth == 'lm':
        sol = _leastsq_minpack(fun, x0, args=args, jac=jac, **options)
    elif meth == 'lm2':
        sol = _leastsq_lmfit(fun, x0, args=args, jac=jac, callback=callback,
                             **options)
    else:
        raise ValueError('Unknown solver %s' % method)

    return sol
