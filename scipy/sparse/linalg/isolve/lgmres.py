import numpy as np
import scipy.lib.blas as blas
from iterative import set_docstring
from utils import make_system

__all__ = ['lgmres']

def norm2(q):
    q = np.asarray(q).ravel()
    if np.iscomplexobj(q):
        try:
            nrm2 = blas.fblas.dznrm2
        except AttributeError:
            nrm2 = blas.cblas.dznrm2
    else:
        try:
            nrm2 = blas.fblas.dnrm2
        except AttributeError:
            nrm2 = blas.cblas.dnrm2
    return nrm2(q)

def lgmres(A, b, x0=None, tol=1e-5, maxiter=1000, M=None, callback=None,
           inner_m=50, outer_k=3, outer_v=None):
    """
    Solve a matrix equation using the LGMRES algorithm.

    Parameters
    ----------
    A : {sparse matrix, dense matrix, LinearOperator}
        The N-by-N matrix of the linear system.
    b : {array, matrix}
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0  : {array, matrix}
        Starting guess for the solution.
    tol : float
        Relative tolerance to achieve before terminating.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse matrix, dense matrix, LinearOperator}
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.

    Additional parameters
    ---------------------
    inner_m : int, optional
        Number of inner GMRES iterations per each outer iteration.
    outer_k : int, optional
        Number of vectors to carry between inner GMRES iterations.
    outer_v : list of vectors, optional
        List containing vectors carried between inner GMRES iterations.
        This parameter is modified in place by `lgmres`, and can be used
        to pass "guess" vectors when solving nearly similar problems.

    Returns
    -------
    x : array or matrix
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : illegal input or breakdown

    References
    ----------
    .. [LGM] A.H. Baker and E.R. Jessup and T. Manteuffel,
             SIAM J. Matrix Anal. Appl. 26, 962 (2005).
    """
    from scipy.linalg.basic import lstsq
    A,M,x,b,postprocess = make_system(A,M,x0,b)

    matvec = A.matvec
    psolve = M.matvec

    if outer_v is None:
        outer_v = []

    axpy, dotc, scal = None, None, None
    total_iter = 0
    exit_flag = False

    for k_outer in xrange(maxiter):
        total_iter += 1
        f_outer = matvec(x)
        r_outer = f_outer - b

        # -- callback
        if callback is not None:
            callback(x)
        if total_iter >= maxiter:
            break

        # -- determine input type routines
        if axpy is None:
            if np.iscomplexobj(r_outer) and not np.iscomplexobj(x):
                x = x.astype(r_outer.dtype)

            try:
                axpy, dotc, scal = blas.get_blas_funcs(
                    ['axpy','dotc','scal'], (x, r_outer))
            except AttributeError:
                axpy, dotc, scal = blas.get_blas_funcs(
                    ['axpy','dot','scal'], (x, r_outer))

        # -- check stopping condition
        if norm2(r_outer) < tol * norm2(f_outer):
            break

        # -- inner LGMRES iteration
        vs0 = -psolve(r_outer)
        inner_res_0 = norm2(vs0)
        vs0 = scal(1.0/inner_res_0, vs0)
        hs = []
        vs = [vs0]
        ws = []
        y = None

        for j in xrange(1, 1 + inner_m + len(outer_v)):
            # -- Arnoldi process:

            #     ++ evaluate
            if j < len(outer_v) + 1:
                z = outer_v[j-1]
            elif j == len(outer_v) + 1:
                z = vs0
            else:
                z = vs[-1]

            total_iter += 1
            v_new = psolve(matvec(z))

            # -- callback
            if callback is not None:
                callback(x)
            if total_iter >= maxiter:
                exit_flag = True
                break

            #     ++ orthogonalize
            hcur = []
            for v in vs:
                alpha = dotc(v, v_new)
                hcur.append(alpha)
                v_new = axpy(v, v_new, v.shape[0], -alpha) # v_new -= alpha*v
            hcur.append(norm2(v_new))

            if hcur[-1] == 0:
                # Exact solution found; bail out.
                # Zero basis vector (v_new) in the least-squares problem
                # does no harm, so we can just use the same code as usually;
                # it will give zero (inner) residual as a result.
                bailout = True
            else:
                bailout = False
                v_new = scal(1.0/hcur[-1], v_new)

            vs.append(v_new)
            hs.append(hcur)
            ws.append(z)

            # XXX: Ugly: should implement the GMRES iteration properly,
            #      with Givens rotations and not using lstsq
            if not bailout and j % 5 != 0 and j < inner_m + len(outer_v) - 1:
                continue

            # -- GMRES optimization problem
            hess  = np.zeros((j+1, j), complex)
            e1    = np.zeros((j+1,), complex)
            e1[0] = inner_res_0
            for q in xrange(j):
                hess[:(q+2),q] = hs[q]

            y, resids, rank, s = lstsq(hess, e1)
            inner_res = norm2(np.dot(hess, y) - e1)

            # -- check for termination
            if inner_res < tol*inner_res_0:
                break

        if exit_flag:
            break

        # -- GMRES terminated: eval solution
        dx = ws[0]*y[0]
        for w, yc in zip(ws[1:], y[1:]):
            dx = axpy(w, dx, dx.shape[0], yc) # dx += w*yc

        # -- Store LGMRES augmentation vectors

        # XXX: Could store the previous (A x) products here...

        nx = norm2(dx)
        outer_v.append(dx / nx)
        while len(outer_v) > outer_k:
            del outer_v[0]

        # -- apply step
        x += dx

    if total_iter == maxiter or exit_flag:
        # didn't converge ...
        return postprocess(x), maxiter

    return postprocess(x), 0
