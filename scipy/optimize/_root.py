"""
Unified interfaces to root finding algorithms.

Functions
---------
- root : find a root of a vector function.
"""
from __future__ import division, print_function, absolute_import

__all__ = ['root']

import numpy as np

from scipy.lib.six import callable

from warnings import warn

from .optimize import MemoizeJac, Result, _check_unknown_options
from .minpack import _root_hybr, leastsq
from . import nonlin


def root(fun, x0, args=(), method='hybr', jac=None, tol=None, callback=None,
         options=None):
    """
    Find a root of a vector function.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    fun : callable
        A vector function to find a root of.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function and its Jacobian.
    method : str, optional
        Type of solver.  Should be one of

            - 'hybr'
            - 'lm'
            - 'broyden1'
            - 'broyden2'
            - 'anderson'
            - 'linearmixing'
            - 'diagbroyden'
            - 'excitingmixing'
            - 'krylov'

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
        the corresponding residual. For all methods but 'hybr' and 'lm'.
    options : dict, optional
        A dictionary of solver options. E.g. `xtol` or `maxiter`, see
        below for details.

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
    'method' parameter. The default method is *hybr*.

    *Hybr*

        A modification of the Powell hybrid method as implemented in
        MINPACK [1]_.

        Options:

        col_deriv : bool
            Specify whether the Jacobian function computes derivatives down
            the columns (faster, because there is no transpose operation).
        xtol : float
            The calculation will terminate if the relative error between
            two consecutive iterates is at most `xtol`.
        maxfev : int
            The maximum number of calls to the function. If zero, then
            ``100*(N+1)`` is the maximum where N is the number of elements
            in `x0`.
        band : sequence
            If set to a two-sequence containing the number of sub- and
            super-diagonals within the band of the Jacobi matrix, the
            Jacobi matrix is considered banded (only for ``fprime=None``).
        epsfcn : float
            A suitable step length for the forward-difference approximation
            of the Jacobian (for ``fprime=None``). If `epsfcn` is less than
            the machine precision, it is assumed that the relative errors
            in the functions are of the order of the machine precision.
        factor : float
            A parameter determining the initial step bound (``factor * ||
            diag * x||``).  Should be in the interval ``(0.1, 100)``.
        diag : sequence
            N positive entries that serve as a scale factors for the
            variables.

    *LM*

        Solves the system of nonlinear equations in a least squares
        sense using a modification of the Levenberg-Marquardt
        algorithm as implemented in MINPACK [1]_.

        Options:

        col_deriv : bool
            non-zero to specify that the Jacobian function computes derivatives
            down the columns (faster, because there is no transpose operation).
        ftol : float
            Relative error desired in the sum of squares.
        xtol : float
            Relative error desired in the approximate solution.
        gtol : float
            Orthogonality desired between the function vector and the columns
            of the Jacobian.
        maxiter : int
            The maximum number of calls to the function. If zero, then
            100*(N+1) is the maximum where N is the number of elements in x0.
        epsfcn : float
            A suitable step length for the forward-difference approximation of
            the Jacobian (for Dfun=None). If epsfcn is less than the machine
            precision, it is assumed that the relative errors in the functions
            are of the order of the machine precision.
        factor : float
            A parameter determining the initial step bound
            (``factor * || diag * x||``). Should be in interval ``(0.1, 100)``.
        diag : sequence
            N positive entries that serve as a scale factors for the variables.

    *broyden1*, *broyden2*, *anderson*, *linearmixing*, *diagbroyden*,
    *excitingmixing*, *krylov*

        These algorithms are inexact Newton methods, with backtracking
        or full line searches [2]_. Each method corresponds to a
        particular Jacobian approximations. See `nonlin` for details.

        *broyden1* uses Broyden's first Jacobian approximation, it is
        known as Broyden's good method.

        *broyden2* uses Broyden's second Jacobian approximation, it
        is known as Broyden's bad method.

        *anderson* uses (extended) Anderson mixing.

        *Krylov* uses Krylov approximation for inverse Jacobian. It
        is suitable for large-scale problem.

        *diagbroyden* uses diagonal Broyden Jacobian approximation.

        *linearmixing* uses a scalar Jacobian approximation.

        *excitingmixing* uses a tuned diagonal Jacobian
        approximation.

        
        .. warning::

           The algorithms implemented for methods *diagbroyden*,
           *linearmixing* and *excitingmixing* may be useful for
           specific problems, but whether they will work may depend
           strongly on the problem.

        Common options:

        nit : int, optional
            Number of iterations to make. If omitted (default), make as many
            as required to meet tolerances.
        disp : bool, optional
            Print status to stdout on every iteration.
        maxiter : int, optional
            Maximum number of iterations to make. If more are needed to
            meet convergence, `NoConvergence` is raised.
        ftol : float, optional
            Relative tolerance for the residual. If omitted, not used.
        fatol : float, optional
            Absolute tolerance (in max-norm) for the residual.
            If omitted, default is 6e-6.
        xtol : float, optional
            Relative minimum step size. If omitted, not used.
        xatol : float, optional
            Absolute minimum step size, as determined from the Jacobian
            approximation. If the step size is smaller than this, optimization
            is terminated as successful. If omitted, not used.
        tol_norm : function(vector) -> scalar, optional
            Norm to use in convergence check. Default is the maximum norm.
        line_search : {None, 'armijo' (default), 'wolfe'}, optional
            Which type of a line search to use to determine the step size in
            the direction given by the Jacobian approximation. Defaults to
            'armijo'.
        jac_options : dict, optional
            Options for the respective Jacobian approximation, see the
            list below.

            For *Broyden1*:

                alpha : float, optional
                    Initial guess for the Jacobian is (-1/alpha).
                reduction_method : str or tuple, optional
                    Method used in ensuring that the rank of the Broyden
                    matrix stays low. Can either be a string giving the
                    name of the method, or a tuple of the form ``(method,
                    param1, param2, ...)`` that gives the name of the
                    method and values for additional parameters.

                    Methods available:
                        - ``restart``: drop all matrix columns. Has no
                            extra parameters.
                        - ``simple``: drop oldest matrix column. Has no
                            extra parameters.
                        - ``svd``: keep only the most significant SVD
                            components.
                          Extra parameters:
                              - ``to_retain`: number of SVD components to
                                  retain when rank reduction is done.
                                  Default is ``max_rank - 2``.
                max_rank : int, optional
                    Maximum rank for the Broyden matrix.
                    Default is infinity (ie., no rank reduction).

            For *Broyden2*:

                alpha : float, optional
                    Initial guess for the Jacobian is (-1/alpha).
                reduction_method : str or tuple, optional
                    Method used in ensuring that the rank of the Broyden
                    matrix stays low. Can either be a string giving the
                    name of the method, or a tuple of the form ``(method,
                    param1, param2, ...)`` that gives the name of the
                    method and values for additional parameters.

                    Methods available:
                        - ``restart``: drop all matrix columns. Has no
                            extra parameters.
                        - ``simple``: drop oldest matrix column. Has no
                            extra parameters.
                        - ``svd``: keep only the most significant SVD
                            components.
                          Extra parameters:
                              - ``to_retain`: number of SVD components to
                                  retain when rank reduction is done.
                                  Default is ``max_rank - 2``.
                max_rank : int, optional
                    Maximum rank for the Broyden matrix.
                    Default is infinity (ie., no rank reduction).
                    
            For *Anderson*:

                alpha : float, optional
                    Initial guess for the Jacobian is (-1/alpha).
                M : float, optional
                    Number of previous vectors to retain. Defaults to 5.
                w0 : float, optional
                    Regularization parameter for numerical stability.
                    Compared to unity, good values of the order of 0.01.

            For *LinearMixing*:

                alpha : float, optional
                    initial guess for the jacobian is (-1/alpha).

            For *DiagBroyden*:

                alpha : float, optional
                    initial guess for the jacobian is (-1/alpha).

            For *ExcitingMixing*:

                alpha : float, optional
                    Initial Jacobian approximation is (-1/alpha).
                alphamax : float, optional
                    The entries of the diagonal Jacobian are kept in the range
                    ``[alpha, alphamax]``.

            For *Krylov*:

                rdiff : float, optional
                    Relative step size to use in numerical differentiation.
                method : {'lgmres', 'gmres', 'bicgstab', 'cgs', 'minres'} or
                    function
                    Krylov method to use to approximate the Jacobian.
                    Can be a string, or a function implementing the same
                    interface as the iterative solvers in
                    `scipy.sparse.linalg`.

                    The default is `scipy.sparse.linalg.lgmres`.
                inner_M : LinearOperator or InverseJacobian
                    Preconditioner for the inner Krylov iteration.
                    Note that you can use also inverse Jacobians as (adaptive)
                    preconditioners. For example,

                    >>> jac = BroydenFirst()
                    >>> kjac = KrylovJacobian(inner_M=jac.inverse).

                    If the preconditioner has a method named 'update', it will
                    be called as ``update(x, f)`` after each nonlinear step,
                    with ``x`` giving the current point, and ``f`` the current
                    function value.
                inner_tol, inner_maxiter, ...
                    Parameters to pass on to the "inner" Krylov solver.
                    See `scipy.sparse.linalg.gmres` for details.
                outer_k : int, optional
                    Size of the subspace kept across LGMRES nonlinear
                    iterations.

                    See `scipy.sparse.linalg.lgmres` for details.

    References
    ----------
    .. [1] More, Jorge J., Burton S. Garbow, and Kenneth E. Hillstrom.
       1980. User Guide for MINPACK-1.
    .. [2] C. T. Kelley. 1995. Iterative Methods for Linear and Nonlinear
        Equations. Society for Industrial and Applied Mathematics.
        <http://www.siam.org/books/kelley/>

    Examples
    --------
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

    if callback is not None and meth in ('hybr', 'lm'):
        warn('Method %s does not accept callback.' % method,
             RuntimeWarning)

    # fun also returns the jacobian
    if not callable(jac) and meth in ('hybr', 'lm'):
        if bool(jac):
            fun = MemoizeJac(fun)
            jac = fun.derivative
        else:
            jac = None

    # set default tolerances
    if tol is not None:
        options = dict(options)
        if meth in ('hybr', 'lm'):
            options.setdefault('xtol', tol)
        elif meth in ('broyden1', 'broyden2', 'anderson', 'linearmixing',
                      'diagbroyden', 'excitingmixing', 'krylov'):
            options.setdefault('xtol', tol)
            options.setdefault('xatol', np.inf)
            options.setdefault('ftol', np.inf)
            options.setdefault('fatol', np.inf)

    if meth == 'hybr':
        sol = _root_hybr(fun, x0, args=args, jac=jac, **options)
    elif meth == 'lm':
        sol = _root_leastsq(fun, x0, args=args, jac=jac, **options)
    elif meth in ('broyden1', 'broyden2', 'anderson', 'linearmixing',
                  'diagbroyden', 'excitingmixing', 'krylov'):
        if jac is not None:
            warn('Method %s does not use the jacobian (jac).' % method,
                 RuntimeWarning)
        sol = _root_nonlin_solve(fun, x0, args=args, jac=jac,
                                 _method=meth, _callback=callback,
                                 **options)
    else:
        raise ValueError('Unknown solver %s' % method)

    return sol


def _root_leastsq(func, x0, args=(), jac=None,
                  col_deriv=0, xtol=1.49012e-08, ftol=1.49012e-08,
                  gtol=0.0, maxiter=0, eps=0.0, factor=100, diag=None,
                  **unknown_options):
    _check_unknown_options(unknown_options)
    x, cov_x, info, msg, ier = leastsq(func, x0, args=args, Dfun=jac,
                                       full_output=True,
                                       col_deriv=col_deriv, xtol=xtol,
                                       ftol=ftol, gtol=gtol,
                                       maxfev=maxiter, epsfcn=eps,
                                       factor=factor, diag=diag)
    sol = Result(x=x, message=msg, status=ier,
                 success=ier in (1, 2, 3, 4), cov_x=cov_x,
                 fun=info.pop('fvec'))
    sol.update(info)
    return sol


def _root_nonlin_solve(func, x0, args=(), jac=None,
                       _callback=None, _method=None,
                       nit=None, disp=False, maxiter=None,
                       ftol=None, fatol=None, xtol=None, xatol=None,
                       tol_norm=None, line_search='armijo', jac_options=None,
                       **unknown_options):
    _check_unknown_options(unknown_options)

    f_tol = fatol
    f_rtol = ftol
    x_tol = xatol
    x_rtol = xtol
    verbose = disp
    if jac_options is None:
        jac_options = dict()

    jacobian = {'broyden1': nonlin.BroydenFirst,
                'broyden2': nonlin.BroydenSecond,
                'anderson': nonlin.Anderson,
                'linearmixing': nonlin.LinearMixing,
                'diagbroyden': nonlin.DiagBroyden,
                'excitingmixing': nonlin.ExcitingMixing,
                'krylov': nonlin.KrylovJacobian
                }[_method]

    if args:
        if jac == True:
            def f(x):
                return func(x, *args)[0]
        else:
            def f(x):
                return func(x, *args)
    else:
        f = func

    x, info = nonlin.nonlin_solve(f, x0, jacobian=jacobian(**jac_options),
                                  iter=nit, verbose=verbose,
                                  maxiter=maxiter, f_tol=f_tol,
                                  f_rtol=f_rtol, x_tol=x_tol,
                                  x_rtol=x_rtol, tol_norm=tol_norm,
                                  line_search=line_search,
                                  callback=_callback, full_output=True,
                                  raise_exception=False)
    sol = Result(x=x)
    sol.update(info)
    return sol
