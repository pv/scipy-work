from __future__ import division, absolute_import, print_function

import inspect

import numpy as np

from numpy.testing import assert_, assert_allclose
from scipy import linalg, interpolate, special
from scipy.integrate import solve_bvp


class TwoPointBVP(object):
    """
    A generic two-point boundary value problem::

        u_i^{m_i}(x) = f_i(x, u),   i = 1 .. N
        g_j(u_a, u_b) = 0,          M = m_1 + ... + mnp

    where ``a`` and ``b`` are some given positions.
    """

    m = []  # Degrees of derivatives of variables
    a = 0  # Left boundary
    b = 1  # Right boundary
    linear = False  # Is this problem linear?
    homogenous = None  # If the problem is linear, is it homogenous?
    vectorized = False  # Are the functions vectorizable over x?
    dtype = np.float64  # Problem's dtype

    # If boundary conditions are separated, ``n_a`` indicates
    # that ``g[0:n_a]`` corresponds to boundary conditions at ``a``,
    # and ``g[n_a:]`` to boundary conditions at ``b``.
    # If boundary conditions are not separated, ``n_a`` must be ``None``.
    n_a = None

    def f(self, x, u):
        """
        Function ``f`` at position ``x``.
        """
        NotImplemented

    def df(self, x, u):
        """
        Partial derivatives of function ``f`` at position ``x``.
        """
        NotImplemented

    def g(self, u_a, u_b):
        """
        The function ``g``.
        """
        NotImplemented

    def dg(self, u_a, u_b):
        """
        Partial derivatives of the function g at ``a`` and at ``b``.
        """
        NotImplemented

    def guess(self, x):
        """
        Evaluate an initial guess for the exact solution at ``x``.
        """
        NotImplemented

    def exact_solution(self, x):
        """
        Evaluate the exact solution at ``x``.
        """
        NotImplemented


class FirstOrderConverter(object):
    """
    Convert a higher-order differential equation system to
    a first-order system
    """
    
    def __init__(self, problem):
        """Convert the underlying ``problem`` to a first-order system"""
        self.problem = problem

        self.mstar = sum(problem.m)
        self.f_map = np.zeros([len(problem.m)], np.int_)
        self.y_map = np.zeros([self.mstar - len(problem.m)], np.int_)
        self.m = [1]*self.mstar

        # Ugh, looks like someone is writing Fortran in Python
        i = 0
        j = 0
        for k, mval in enumerate(problem.m):
            for p in range(mval-1):
                self.y_map[j] = i
                i += 1
                j += 1
            self.f_map[k] = i
            i += 1

    def f(self, x, y):
        dy = np.empty((self.mstar,) + x.shape, self.problem.dtype)
        dy[self.f_map] = self.problem.f(x, y)
        dy[self.y_map] = y[self.y_map + 1]
        return dy

    def df(self, x, y):
        ddy = np.zeros((self.mstar, self.mstar) + x.shape, self.problem.dtype)
        ddy[self.f_map, :] = self.problem.df(x, y)
        ddy[self.y_map, self.y_map + 1] = 1
        return ddy

    def __setattr__(self, name, value):
        if name in ('mstar', 'f_map', 'y_map', 'problem', 'm'):
            object.__setattr__(self, name, value)
        else:
            setattr(self.problem, name, value)

    def __getattr__(self, name):
        return getattr(self.problem, name)


def solve_with_bvp(problem, initial_guess=None, max_nodes=1000,
                   numerical_jacobians=False, use_guess=True, tol=1e-6):
    assert_(problem.n_a != None, "Can solve only separated BCs")

    if max(problem.m) > 1:
        problem = FirstOrderConverter(problem)

    def vectorized_guess(x):
        us = []
        dms = []
        for i, xx in enumerate(x):
            u, dm = problem.guess(np.float64(xx))
            us.append(u)
            dms.append(dm)
        return np.transpose(np.asarray(us)), np.transpose(np.asarray(dms))

    def vectorized_f(x, u):
        fs = []
        for i, xx in enumerate(x):
            fs.append(problem.f(np.float64(xx), u[:,i]))
        return np.transpose(np.asarray(fs))

    def vectorized_df(x, u):
        dfs = []
        for i, xx in enumerate(x):
            dfs.append(problem.df(np.float64(xx), u[:,i]))
        dfs = np.asarray(dfs)
        return np.swapaxes(np.swapaxes(dfs, 0, 2), 0, 1)

    fun = problem.f
    fun_jac = problem.df
    bc = problem.g
    bc_jac = problem.dg
    if use_guess:
        guess = problem.guess
    else:
        guess = None

    if numerical_jacobians:
        fun_jac = None
        bc_jac = None

    if not problem.vectorized:
        fun = vectorized_f
        if problem.guess is not None:
            guess = vectorized_guess
        if problem.df is not None:
            fun_jac = vectorized_df

    if initial_guess is not None:
        x, y = initial_guess
    else:
        x = np.linspace(problem.a, problem.b, 5)
        if guess is None:
            y = np.zeros((sum(problem.m), len(x)), dtype=problem.dtype)
        else:
            y = guess(x)[0].astype(problem.dtype)

    solution = solve_bvp(fun, bc, x, y, fun_jac=fun_jac, bc_jac=bc_jac, tol=tol, max_nodes=max_nodes,
                         verbose=0)
    assert_(solution.success)
    return solution


class TestBvp(object):
    def test_problem_1(self, num_jac=False):
        # Solve problem #1 and compare to exact solution
        problem = Problem1()
        solution = solve_with_bvp(problem, numerical_jacobians=num_jac)
        x = np.linspace(problem.a, problem.b, 100)
        assert_allclose(problem.exact_solution(x), solution.sol(x)[0],
                        rtol=1e-5)

    def test_problem_2(self, num_jac=False):
        # Solve problem #2 and compare to exact solution
        problem = Problem2()
        solution = solve_with_bvp(problem, numerical_jacobians=num_jac)
        x = np.linspace(problem.a, problem.b, 100)
        assert_allclose(problem.exact_solution(x), solution.sol(x).T,
                        rtol=1e-3, atol=1e-6)

    def test_complex_problem_2(self, num_jac=False):
        # Solve problem (complex-valued) #2 and compare to exact solution
        problem = ComplexProblem2()
        solution = solve_with_bvp(problem, numerical_jacobians=num_jac)
        x = np.linspace(problem.a, problem.b, 100)
        assert_allclose(problem.exact_solution(x), solution.sol(x).T,
                        rtol=1e-3, atol=1e-6)

    def test_problem_3(self, num_jac=False):
        # Solve problem #3 and compare to exact solution
        problem = Problem3()
        solution = solve_with_bvp(problem, numerical_jacobians=num_jac)
        x = np.linspace(problem.a, problem.b, 100)
        assert_allclose(problem.exact_solution(x), solution.sol(x)[0],
                        rtol=1e-5)

    def test_problem_5(self, num_jac=False):
        # Solve problem #5 and compare to exact solution
        problem = Problem5()
        solution = solve_with_bvp(problem, numerical_jacobians=num_jac)
        x = np.linspace(problem.a, problem.b, 100)
        assert_allclose(problem.exact_solution(x), solution.sol(x).T,
                        rtol=1e-2)
        # Needed to reduce tolerance, since "exact" solution data is
        # interpolated

    def test_problem_6(self, num_jac=False):
        # Solve problem #6 and compare to exact solution
        problem = Problem6()
        solution = solve_with_bvp(problem, numerical_jacobians=num_jac)
        x = np.linspace(problem.a, problem.b, 100)
        assert_allclose(problem.exact_solution(x), solution.sol(x).T,
                        rtol=1e-5, atol=1e-9)

    def test_problem_7(self, num_jac=False):
        # Solve problem #7 for two different initial guesses
        # and compare to known approximate solutions
        problem = Problem7()

        tol = 1e-3

        solution1 = solve_with_bvp(problem, use_guess=False, tol=tol,
                                   numerical_jacobians=num_jac)

        solution2 = solve_with_bvp(problem, use_guess=True, tol=tol,
                                   numerical_jacobians=num_jac)

        x = problem.solution_1[:,0]
        assert_(curves_close(x, problem.solution_1[:,1],
                             problem.solution_1_xtol, problem.solution_1_ytol,
                             solution1.sol(x)[0], 0, 0))

        x = problem.solution_2[:,0]
        assert_(curves_close(x, problem.solution_2[:,1],
                             problem.solution_2_xtol, problem.solution_2_ytol,
                             solution2.sol(x)[0], 0, 0))

    def test_problem_8(self, num_jac=False):
        # Solve problem #8 with simple continuation
        # and compare to known approximate solutions
        problem = Problem8()

        x0 = np.linspace(problem.a, problem.b, 31)
        y0 = np.asarray([problem.guess(xx)[0] for xx in x0]).T
        guess = (x0, y0)

        for c in problem.continuation:
            problem.__dict__.update(c)
            solution = solve_with_bvp(problem, max_nodes=5000,
                                      tol=1e-5,
                                      initial_guess=guess,
                                      numerical_jacobians=num_jac)
            guess = (solution.x[::2], solution.y[...,::2])

        xg = problem.solution_g[:,0]
        assert_(curves_close(xg, problem.solution_g[:,1],
                             problem.solution_g_xtol, problem.solution_g_ytol,
                             solution.sol(xg)[0], 0, 0))

        xh = problem.solution_h[:,0]
        assert_(curves_close(xh, problem.solution_h[:,1],
                             problem.solution_h_xtol, problem.solution_h_ytol,
                             solution.sol(xh)[2], 0, 0))

    def test_problem_9(self, num_jac=False):
        # Solve problem #9
        problem = Problem9()
        solution = solve_with_bvp(problem, numerical_jacobians=num_jac)
        x = np.linspace(problem.a, problem.b, 10)
        assert_allclose(problem.exact_solution(x), solution.sol(x).T,
                        rtol=1e-3, atol=1e-6)

    def test_problem_10(self, num_jac=False):
        # Solve problem #10
        problem = Problem10()
        solution = solve_with_bvp(problem, numerical_jacobians=num_jac)
        x = np.linspace(problem.a, problem.b, 10)
        assert_allclose(problem.exact_solution(x), solution.sol(x).T,
                          rtol=1e-3, atol=1e-6)

    def test_continuation(self, num_jac=False):
        # Solve problem #3 for multiple values of C using continuation
        problem = Problem3()
        x = np.linspace(problem.a, problem.b, 501)

        # Near C = 1.69 there is a "difficult point", be careful about it.
        Cvals = np.r_[np.linspace(1, 1.6, 5),
                      np.linspace(1.6, 1.72, 7)[1:],
                      np.linspace(1.8, 150, 90)]

        # Use continuation
        x0 = np.linspace(problem.a, problem.b, 31)
        y0 = np.asarray([problem.guess(xx)[0] for xx in x0]).T
        guess = (x0, y0)

        prev_C = None
        prev_sol = None
        pprev_C = None
        pprev_sol = None

        for C in Cvals:
            problem.C = C

            if pprev_sol is not None:
                # Linear predictor
                guess = (prev_sol.x[::2],
                         prev_sol.y[...,::2]
                         + (C - prev_C) / (prev_C - pprev_C) * (
                             prev_sol.y[...,::2] - pprev_sol.sol(prev_sol.x[::2])))
            elif prev_sol is not None:
                guess = (prev_sol.x[::2], prev_sol.y[...,::2])

            solution = solve_with_bvp(problem, initial_guess=guess,
                                      numerical_jacobians=num_jac,
                                      tol=1e-6, max_nodes=9000)
            pprev_sol = prev_sol
            pprev_C = prev_C
            prev_sol = solution
            prev_C = C

            assert_allclose(problem.exact_solution(x), solution.sol(x)[0],
                            rtol=1e-5)

        # Finding the given analytical solution without continuation
        # is not possible, probably due to a branch point
        solution = solve_with_bvp(problem, numerical_jacobians=num_jac)
        assert_(not np.allclose(problem.exact_solution(x), solution.sol(x)[0],
                                rtol=1e-1))

    def test_tolerances(self, num_jac=False):
        # Solve problem #3 with different tolerances
        problem = Problem3()
        x = np.linspace(0, 1, 20)

        # Large tolerances, less exact solution
        solution = solve_with_bvp(problem, tol=1e-1,
                                  numerical_jacobians=num_jac)
        assert_allclose(problem.exact_solution(x), solution.sol(x)[0],
                          rtol=1e-1)
        assert_(not np.allclose(problem.exact_solution(x), solution.sol(x)[0],
                                rtol=1e-5))

        # Large tolerances, more exact solution
        solution = solve_with_bvp(problem, tol=1e-5,
                                  numerical_jacobians=num_jac)
        assert_allclose(problem.exact_solution(x), solution.sol(x)[0],
                        rtol=1e-5)

    def test_reentrancy(self):
        count = [0]
        depth = [0]
        class RecursiveProblem(Problem1):
            def f(self, x, z):
                # work out a back-and-forth jumping recursion
                # 1 2 3 4 5 6 4 5 5 3 4 3 1 2 2 0
                if count[0] + 2*depth[0] + 3*((count[0]+2) % 3) < 20:
                    count[0] += 1
                    depth[0] += 1
                    problem = RecursiveProblem()
                    solution = solve_with_bvp(problem)
                    depth[0] -= 1
                    xx = np.linspace(problem.a, problem.b, 100)
                    assert_allclose(problem.exact_solution(xx),
                                       solution.sol(xx)[0],
                                       rtol=1e-5)
                return Problem1.f(self, x, z)

        problem1 = RecursiveProblem()
        
        solution = solve_with_bvp(problem1)
        x = np.linspace(problem1.a, problem1.b, 100)
        assert_allclose(problem1.exact_solution(x), solution.sol(x)[0],
                          rtol=1e-5)


class TestBvpNumericalJacobians(TestBvp):
    # Same as test_bvp, but with numerically approximated Jacobians
    pass


def add_jac_func(cls, v):
    # Needed for making the lambda a closure
    func = lambda self: v(self, True)
    func.__name__ = v.__name__
    setattr(TestBvpNumericalJacobians, func.__name__, func)


def ignored_check(self):
    return True


for name, v in inspect.getmembers(TestBvpNumericalJacobians):
    if name.startswith('test_'):
        argspec = inspect.getargspec(v)
        if 'num_jac' in argspec[0] or 'numerical_jacobians' in argspec[0]:
            add_jac_func(TestBvpNumericalJacobians, v)
        else:
            setattr(TestBvpNumericalJacobians, name, ignored_check)
    del v


    
def curves_close(x, y1, xtol1, ytol1,
                    y2, xtol2, ytol2):
    """Returns true if curves coincide up to given absolute tolerances.

    :Parameters:
      - `x`: x-coordinates
      - `y1`: points of curve 1 at `x`
      - `y2`: points of curve 2 at `x`
      - `xtol1`: absolute x-tolerance for curve 1
      - `xtol2`: absolute x-tolerance for curve 2
      - `ytol1`: absolute y-tolerance for curve 1
      - `ytol2`: absolute y-tolerance for curve 2
    """
    
    def ddiff(x):
        dx = np.diff(x)
        return np.r_[dx, dx[-1]]

    dx = ddiff(x)
    dy1 = np.absolute(ddiff(y1))
    dy2 = np.absolute(ddiff(y2))

    tol = ytol1 + ytol2 + xtol1*dy1/dx + xtol2*dy2/dx
    d = np.less(np.absolute(y1 - y2), tol)
    return d.ravel().all()


#
# A set of test problems follows
#

class Problem1(TwoPointBVP):
    """
    A trivial test problem::
    
       y'' = 0
       y(0) = 0, y(1) = 1
    """

    m = [2]

    a = 0
    b = 1
    n_a = 1
    linear = True
    homogenous = True

    def f(self, x, u):
        return np.array([0])

    def df(self, x, u):
        return np.array([[0, 0]])

    def g(self, u_a, u_b):
        return np.array([u_a[0] - 0, u_b[0] - 1])

    def dg(self, u_a, u_b):
        return (np.array([[1, 0],
                         [0, 0]]),
                np.array([[0, 0],
                         [1, 0]]))


    guess = None

    def exact_solution(self, x):
        return x


class Problem2(TwoPointBVP):
    """
    Linear test problem with separated boundary conditions::

       u' = A u
       B u_a = [1]
       C u_b = [2; 3]

    Where A is 3x3, B is 1x3, and C is 2x3. They are given below.
    """

    m = [1, 1, 1]

    A = np.mat('0 1 2; 1 0 3; 2 3 4').A
    B = np.mat('1 2 3').A
    C = np.mat('4 5 6; 7 8 9').A

    a_rhs = np.array([1])
    b_rhs = np.array([2,3])

    a = 0
    b = 1
    n_a = 1
    linear = True
    homogenous = True

    def f(self, x, u):
        return self.A.dot(u.T)

    def df(self, x, u):
        return np.asarray(self.A)

    def g(self, u_a, u_b):
        return np.r_[self.B.dot(u_a) - self.a_rhs,
                     self.C.dot(u_b) - self.b_rhs]

    def dg(self, u_a, u_b):
        return (np.r_[self.B, 0*self.C],
                np.r_[0*self.B, self.C])

    guess = None

    def exact_solution(self, x):
        X = np.r_[self.B, self.C.dot(linalg.expm(self.A))]
        b = np.r_[self.a_rhs, self.b_rhs]
        y = np.zeros((len(x), 3), self.dtype)
        for i, xx in enumerate(np.asarray(x)):
            sol = np.dot(linalg.expm(self.A.dot(xx)), linalg.solve(X, b))
            y[i,:] = np.asarray(sol).ravel()
        return y


class ComplexProblem2(Problem2):
    """
    Complex version of Problem2

    """

    A = np.mat('0 1j 2; 1 0 2+3j; 2 3+1j 4').A
    B = np.mat('1 2j 0.5+3j').A
    C = np.mat('4 1+5j 6; 7 8j 9').A

    a_rhs = np.array([1])
    b_rhs = np.array([2j, 3])

    is_complex = True
    dtype = np.complex_


class Problem3(TwoPointBVP):
    """
    Nonlinear test problem::

        u''  = cosh(u) / sinh(u)**3
        u(0) = u(1) = arccosh(sqrt((1 + C)/C) cosh(sqrt(C)/2))

    ``u`` is scalar, and ``C`` a constant. The problem becomes stiffer
    and stiffer around ``x = 0.5`` as ``C`` increases, and the solution
    approaches::

        2 arccosh(sqrt((1 + C)/C) cosh(sqrt(C)/2)) |x - .5|

    The second reason why this is a difficult problem for numerical
    solvers is that there appears to be a branch point close to
    ``C = 1.7`` -- at least COLNEW appears to find a solution distinct
    from the analytical one, with residuals in u'' below 1e-5. This
    "difficult point" can be passed by careful continuation.
    """

    m = [2]
    
    a = 0
    b = 1
    n_a = 1
    linear = False

    C = 1.5

    vectorized = True

    def f(self, x, u):
        return np.array([np.cosh(u[0])/np.sinh(u[0])**3])

    def df(self, x, u):
        return np.array([[-(1 + 2*np.cosh(u[0])**2)/np.sinh(u[0])**4, 0*x]])

    def g(self, u_a, u_b):
        v = self.exact_solution(0)
        return np.array([u_a[0] - v, u_b[0] - v])

    def dg(self, u_a, u_b):
        return (np.array([[1, 0],
                         [0, 0]]),
                np.array([[0, 0],
                         [1, 0]]))

    def guess(self, x):
        z = np.zeros((2,) + np.asarray(x).shape)
        dm = np.zeros((1,) + np.asarray(x).shape)
        z[0] = 1
        return z, dm

    def exact_solution(self, x):
        return np.arccosh(np.sqrt((1+self.C)/self.C)
                         * np.cosh(np.sqrt(self.C)*(x - .5)))


class Problem4(TwoPointBVP):
    """
    R.M.M. Mattheij's example equation for MUSL. It is a linear problem
    with non-separated boundary conditions::
 
        dx(t) / dt = L(t)x(t) + r(t) ,   0 <= t <= 6

    with a boundary condition MA x(0) + MB x(6) = BCV.
    The matrices and the exact solution are given below.
    """
    
    m = [1, 1, 1]
    
    a = 0
    b = 6
    n_a = None # non-separated boundary conditions
    linear = True
    homogenous = False

    C = 1.5

    BCV = np.mat('0; 0; 0') + 1+np.exp(6)
    MA = MB = np.mat('1 0 0; 0 1 0; 0 0 1')

    def L(self, x):
        return np.matrix([ [ 1 - 2*np.cos(2*x), 0, 1 + 2*np.sin(2*x)],
                          [      0,          2,       0         ],
                          [-1 + 2*np.sin(2*x), 0, 1 + 2*np.cos(2*x)]])
    def r(self, x):
        return np.matrix([[(-1 + 2*np.cos(2*x) - 2*np.sin(2*x)) * np.exp(x)],
                         [                                   - np.exp(x)],
                         [( 1 - 2*np.cos(2*x) - 2*np.sin(2*x)) * np.exp(x)]])

    def f_homog(self, x, u):
        return self.L(x) * np.asmatrix(u).T

    def f(self, x, u):
        return self.L(x) * np.asmatrix(u).T + self.r(x)

    def df(self, x, u):
        return self.L(x)

    def g(self, u_a, u_b):
        return self.MA*np.asmatrix(u_a).T + self.MB*np.asmatrix(u_b).T - self.BCV

    def dg(self, u_a, u_b):
        return (self.MA, self.MB)

    def guess(self, x):
        return np.array([1, 1, 1]), np.array([0, 0, 0])

    def exact_solution(self, x):
        return np.transpose(np.array([np.exp(x)]*3))


class Problem5(TwoPointBVP):
    """
    R.M.M. Mattheij's nonlinear example equation for MUSN, having separated
    boundary conditions.
    """
    
    m = [1, 1, 1, 1, 1]
    
    a = 0
    b = 1
    n_a = 4
    linear = False

    def f(self, t, u):
        u, v, w, x, y = u
        ret = np.array([ .5*u*(w-u) / v,
                        -.5*(w-u),
                         (0.9 - 1000*(w-y) - 0.5*w*(w-u)) / x,
                         0.5*(w-u),
                         100*(w-y)])
        return ret

    def df(self, t, u):
        u, v, w, x, y = u
        return np.array([[(.5*w-u)/v, -.5*u*(w-u) / v**2, .5*u/v, 0, 0],
                        [.5, 0, -.5, 0, 0],
                        [.5*w/x,
                         0,
                         (-1000 - w + .5*u)/x,
                         -(0.9 - 1000*(w-y) - 0.5*w*(w-u)) / x**2,
                         1000/x],
                        [-0.5, 0, 0.5, 0, 0],
                        [0, 0, 100, 0, -100]])

    def g(self, u_a, u_b):
        u_a, v_a, w_a, x_a, y_a = u_a
        u_b, v_b, w_b, x_b, y_b = u_b
        return np.array([ u_a - 1, v_a - 1, w_a - 1, x_a - (-10), w_b - y_b])

    def dg(self, u_a, u_b):
        u_a, v_a, w_a, x_a, y_a = u_a
        u_b, v_b, w_b, x_b, y_b = u_b
        return (np.array([[ 1, 0, 0, 0, 0],
                         [ 0, 1, 0, 0, 0],
                         [ 0, 0, 1, 0, 0],
                         [ 0, 0, 0, 1, 0],
                         [ 0, 0, 0, 0, 0]]),
                np.array([[ 0, 0, 0, 0, 0],
                         [ 0, 0, 0, 0, 0],
                         [ 0, 0, 0, 0, 0],
                         [ 0, 0, 0, 0, 0],
                         [ 0, 0, 1, 0, -1]]))

    def guess(self, t):
        return np.array([
            1, 1, -4.5*t*t + 8.91*t + 1, -10, -4.5*t*t + 9*t + 0.91]), \
            np.array([0, 0, 0, 0, 0])

    solution = np.array([
    [ .0000, 1.00000e+00, 1.00000e+00, 1.00000e+00, -1.00000e+01, 9.67963e-01],
    [ .1000, 1.00701e+00, 9.93036e-01, 1.27014e+00, -9.99304e+00, 1.24622e+00],
    [ .2000, 1.02560e+00, 9.75042e-01, 1.47051e+00, -9.97504e+00, 1.45280e+00],
    [ .3000, 1.05313e+00, 9.49550e-01, 1.61931e+00, -9.94955e+00, 1.60610e+00],
    [ .4000, 1.08796e+00, 9.19155e-01, 1.73140e+00, -9.91915e+00, 1.72137e+00],
    [ .5000, 1.12900e+00, 8.85737e-01, 1.81775e+00, -9.88574e+00, 1.80994e+00],
    [ .6000, 1.17554e+00, 8.50676e-01, 1.88576e+00, -9.85068e+00, 1.87957e+00],
    [ .7000, 1.22696e+00, 8.15025e-01, 1.93990e+00, -9.81503e+00, 1.93498e+00],
    [ .8000, 1.28262e+00, 7.79653e-01, 1.98190e+00, -9.77965e+00, 1.97819e+00],
    [ .9000, 1.34161e+00, 7.45374e-01, 2.01050e+00, -9.74537e+00, 2.00827e+00],
    [1.0000, 1.40232e+00, 7.13102e-01, 2.02032e+00, -9.71310e+00, 2.02032e+00],
    ])

    def exact_solution(self, t):
        """
        Interpolate from Mattheij's data, since no real exact solution
        is available
        """
        interp = interpolate.interp1d(self.solution[:,0],
                                      self.solution[:,1:], axis=0)
        return interp(t)


class Problem6(TwoPointBVP):
    """
    COLSYS example problem 1 [1]

    .. [1] U. Ascher, J.Christiansen, R. D. Russell
           ACM Trans. Math. Software 7, 209 (1981).
    """

    m = [4]
    a = 1
    b = 2
    n_a = 2
    linear = False

    def f(self, x, u):
        return np.array([(1 - 6*x**2 * u[3] - 6*x*u[2]) / x**3])

    def df(self, x, u):
        return np.array([[0, 0, -6/x**2, -6/x]])

    def g(self, u_a, u_b):
        return np.array([u_a[0] - 0, u_a[2] - 0, u_b[0] - 0, u_b[2] - 0])

    def dg(self, u_a, u_b):
        return (np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]]),
                np.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 1, 0]]))

    def guess(self, x):
        return np.array([0, 0, 0, 0]), np.array([0])

    def exact_solution(self, x):
        sol = np.array([.25*(10*np.log(2)-3)*(1-x) + .5*(1/x+(3+x)*np.log(x)-x),
                        -.25* (10*np.log(2)-3) + .5*(-1/x/x+np.log(x)+(3+x)/x-1),
                        .5 * (2/x**3 + 1/x - 3/x/x),
                        .5 * (-6/x**4 - 1/x/x + 6/x**3)])
        return np.transpose(sol)


class Problem7(TwoPointBVP):
    """
    COLSYS example problem 2 [1]

    .. [1] U. Ascher, J.Christiansen, R. D. Russell
           ACM Trans. Math. Software 7, 209 (1981).
    """

    m = [2, 2]
    a = 0
    b = 1
    n_a = 2
    linear = False

    gamma = 1.1
    eps = 0.001
    dmu = eps
    eps4mu = eps**4/dmu
    xt = np.sqrt(2 * (gamma - 1)/gamma )

    vectorized = True

    def f(self, x, z):
        phi, dphi, psi, dpsi = z
        return np.array([ phi/x/x - dphi/x
                         + (phi - psi*(1-phi/x) - self.gamma*x*(1 - x*x/2))
                         / self.eps4mu,

                         psi/x/x - dpsi/x + phi*(1-phi/2/x) / self.dmu])

    def df(self, x, z):
        phi, dphi, psi, dpsi = z
        zero = np.zeros(x.shape)
        return np.array([
            [1/x/x +(1 + psi/x) / self.eps4mu, -1/x,
             -(1-phi/x) / self.eps4mu, zero],
            [(1 - phi/x) / self.dmu, zero, 1/x/x, -1/x]])

    def g(self, z_a, z_b):
        phi_a, dphi_a, psi_a, dpsi_a = z_a
        phi_b, dphi_b, psi_b, dpsi_b = z_b
        return np.array([phi_a - 0,
                        psi_a - 0,
                        phi_b - 0,
                        dpsi_b - .3*psi_b + .7 - 0])

    def dg(self, z_a, z_b):
        return (np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]]),
                np.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, -.3, 1]]))
        
    def guess(self, x):
        x_shape = x.shape
        x = np.atleast_1d(x)

        zero = np.zeros(x.shape)
        cons = self.gamma * x * (1 - .5*x*x)
        dcons = self.gamma * (1 - 1.5*x*x)
        d2cons = -3 * self.gamma * x

        i1 = (x < self.xt)
        i2 = (x >= self.xt)

        z = np.empty([4, len(x)])
        
        z[0, i1] = 2 * x[i1]
        z[1, i1] = 2
        z[2, i1] = -2 * x[i1] + cons[i1]
        z[3, i1] = -2 + dcons[i1]

        z[0, i2] = 0
        z[1, i2] = 0
        z[2, i2] = -cons[i2]
        z[3, i2] = -dcons[i2]

        z = z.reshape((4,) + x_shape)

        return z, np.array([-d2cons, zero])

    # This solution data has bad accuracy, since it is extracted from
    # a figure. I'm not sure whether it's useful to include this...

    # Solution I in the paper, extracted from Fig. 1
    # Corresponds to initial guess z = 0
    solution_1 = np.array([
        [0.000000000000,  0.000000000000000],
        [0.244547643166, -0.000371778511570],
        [0.33333245629,  -0.000506755423414],
        [0.397197847737,  0.000916422190941],
        [0.439253811848,  0.000852485759015], # no dimple here
        [0.568536960783,  0.000655940431242],
        [0.777264415297,  0.003379158828100],
        [0.90187731046,   0.004709983818560],
        [0.929911988739,  0.003147093260360],
        [0.956375883073, -0.006014760633050],
        [0.976601370818, -0.019727941273200],
        [0.990591083104, -0.036472182389900], # boundary layer
        [0.995240287846, -0.050161682870000],
        [0.999892123716, -0.062330917080000],
    ])
    solution_1_xtol = 0.002
    solution_1_ytol = 0.002

    # Solution II in the paper, extracted from Fig. 1
    # Corresponds to the initial guess given above
    solution_2 = np.array([
        [0.000000000000,  0.000000000000000],
        [0.051583281807,  0.104820228119000],
        [0.115672319209,  0.235466038704000],
        [0.160996145396,  0.323572809914000],
        [0.204759712154,  0.410161682870000],
        [0.284477655138,  0.571189138700000],
        [0.359506926447,  0.723102100956000],
        [0.406391012064,  0.812726770421000],
        [0.418899398787,  0.840072619157000],
        [0.423166747793,  0.861346874877000],
        [0.424002793996,  0.871986370752000],
        [0.425392061358,  0.691077052609000],
        [0.426332662834,  0.437192585479000],
        [0.429503583598, -0.017369397339900], # a dimple forms here
        [0.430949883573, -0.008257271782460],
        [0.439253811848,  0.000852485759015],
        [0.482870035389,  0.002306447581330],
        [0.560751450410,  0.002188046781470],
        [0.685361714444,  0.001998605501690],
        [0.803746727533,  0.003859168826380],
        [0.917453593464,  0.004686303658580],
        [0.947038006657, -0.001439753726340],
        [0.961043505716, -0.009062397221530],
        [0.976601370818, -0.019727941273200],
        [0.985928722718, -0.030384013260900],
        [0.993693184061, -0.044078249773100], # boundary layer
        [0.996795285017, -0.051684317156300],
        [0.999892123716, -0.062330917080000],
    ])
    solution_2_xtol = 0.001
    solution_2_ytol = 0.002


class Problem8(TwoPointBVP):
    """
    COLSYS example problem 3

        G''  = L^2 s (G - 1) - L [ c H G' - (n - 1) H' G ]
        H''' = L^3 (1 - G^2) + L^2 s H' - L [ c H H'' + n (H')^2 ]

        c := (3 - n) / 2

        G(0) = H(0) = H'(0) = 0, G(1) = 1, H'(1) = 0

    In Ref. [1] the problem is solved with a simple continuation scheme

         n    s     L
        ===  ====  ===
        0.2  0.2    60
        0.2  0.1   120
        0.2  0.05  200

    
    .. [1] U. Ascher, J.Christiansen, R. D. Russell
           ACM Trans. Math. Software 7, 209 (1981).
    """

    m = [2, 3]
    a = 0
    b = 1
    n_a = 3
    linear = False
    
    vectorized = True

    n = 0.2
    s = 0.2
    L = 60

    continuation = [{'n': 0.2, 's': 0.2,  'L': 60},
                    {'n': 0.2, 's': 0.1,  'L': 120},
                    {'n': 0.2, 's': 0.05, 'L': 200}]

    def f(self, x, z):
        G, dG, H, dH, d2H = z
        c = .5*(3 - self.n)
        L, s, n = self.L, self.s, self.n
        return np.array([
            L**2 * s * (G - 1) - L * (c * H * dG + (n-1) * dH * G),
            L**3 * (1-G**2) + L**2 * s * dH - L * (c * H * d2H + n * dH**2)
            ])

    def df(self, x, z):
        G, dG, H, dH, d2H = z
        c = .5*(3 - self.n)
        zero = np.zeros(x.shape)
        L, s, n = self.L, self.s, self.n
        return np.array([
            [-L * (n-1) * dH + L**2 * s,
             -L * c * H,
             -L * c * dG,
             -L * (n-1) * G,
             zero],
            [-L**3 * 2 * G,
             zero,
             -L * c * d2H,
             -L * n * 2 * dH + L**2 * s,
             -L * c * H]
            ])

    def g(self, z_a, z_b):
        G_a, dG_a, H_a, dH_a, d2H_a = z_a
        G_b, dG_b, H_b, dH_b, d2H_b = z_b
        return np.array([G_a - 0,
                        H_a - 0,
                        dH_a - 0,
                        G_b - 1,
                        dH_b - 0])
    def dg(self, z_a, z_b):
        return (np.array([[1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]]),
                np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0]]))

    def guess(self, x):
        L = self.L
        ex = np.exp(-L*x)
        z = np.array([
            1 - ex,
            L * ex,
            -L**2 * x**2 * ex,
            (L**3 *x**2 - 2 * L**2 * x) * ex,
            (-L**4 * x**2  + 4 * L**3 * x - 2 * L**2)*ex
            ])
        dm = np.array([
            -L * z[1],
            (L**5*x*x - 6*L**4*x + 6*L**3) * ex
            ])
        return z, dm

    # This solution data has bad accuracy, since it is extracted from
    # a figure. I'm not sure whether it's useful to include this...

    solution_h = np.array([
        [0.00130714830711, 8.67489072589e-05],
        [0.00730070917232, -1.60886692126],
        [0.0128762980298, -2.33851198022],
        [0.0172866913375, -3.36687690133],
        [0.0287848646816, -5.55615907384],
        [0.0387550743182, -7.28098759688],
        [0.052883316805, -9.5032777286],
        [0.0570255771267, -9.96755788025],
        [0.0624828720197, -10.4483420115],
        [0.0665068383769, -10.6637612355],
        [0.0705071459412, -10.8294082739],
        [0.0731687601412, -10.9287791471],
        [0.078405239634, -10.94502288],
        [0.0836180603339, -10.9114944274],
        [0.0861929256266, -10.828367287],
        [0.093893862712, -10.5292136803],
        [0.103996167276, -9.78193690594],
        [0.115200577275, -8.60321444133],
        [0.127373026216, -6.71100390173],
        [0.139135389414, -3.95607547944],
        [0.150000690048, -2.06395168874],
        [0.153685547041, -1.56596958662],
        [0.156197322219, -1.35011661813],
        [0.158780073776, -1.28358020627],
        [0.162709404962, -1.29991068806],
        [0.169363440462, -1.54833787122],
        [0.17348204199, -1.96284583733],
        [0.185877277898, -3.28932337824],
        [0.194146026013, -4.18470222452],
        [0.200855265363, -4.54926450728],
        [0.206178493763, -4.74800625381],
        [0.210123597477, -4.79751819263],
        [0.214068701191, -4.84703013144],
        [0.220580783934, -4.79682420137],
        [0.225746287048, -4.66375137763],
        [0.232163734619, -4.41445670539],
        [0.238510205812, -4.01584547654],
        [0.243565301226, -3.65050245361],
        [0.249738274604, -2.88689519746],
        [0.254746052432, -2.42200780346],
        [0.258564975917, -2.20606808606],
        [0.262423330724, -2.07308201123],
        [0.268927527203, -2.00628535264],
        [0.279447803774, -2.13831718949],
        [0.284794690967, -2.38683112156],
        [0.302150387117, -3.14887689739],
        [0.311466036817, -3.49667495382],
        [0.31803332341, -3.56260412333],
        [0.323269802903, -3.57884785622],
        [0.328482623602, -3.54531940356],
        [0.333663899245, -3.44542803685],
        [0.347829601487, -2.99652412901],
        [0.361979531201, -2.51443876414],
        [0.369767217194, -2.39778317111],
        [0.377610107036, -2.39726267766],
        [0.394776532843, -2.76113096916],
        [0.407997854936, -3.07548732184],
        [0.414573027793, -3.15800721987],
        [0.421116655593, -3.17416420385],
        [0.43024303495, -3.12378477596],
        [0.438046493471, -3.04031063995],
        [0.458795254835, -2.69051735865],
        [0.466598713356, -2.60704322264],
        [0.47836304812, -2.60626248248],
        [0.484922448448, -2.65560092348],
        [0.512514515655, -2.95241230967],
        [0.522987474641, -2.98489977544],
        [0.534736036876, -2.95093757825],
        [0.563382891933, -2.7167589031],
        [0.573832192125, -2.69947418333],
        [0.585612299418, -2.73187490019],
        [0.60266043126, -2.84688226399],
        [0.611842014467, -2.91263793569],
        [0.620999938881, -2.92862142185],
        [0.62883494246, -2.9115101999],
        [0.641874880474, -2.84427979677],
        [0.654914818488, -2.77704939364],
        [0.669285563602, -2.75950442715],
        [0.68760141243, -2.79147139948],
        [0.711169513279, -2.87286356171],
        [0.724248882615, -2.88858680115],
        [0.736005331115, -2.87121533248],
        [0.763423900507, -2.80303069137],
        [0.786944683771, -2.78487848253],
        [0.847081392162, -2.7974787613],
        [0.996119957966, -2.83736157142],
    ])
    solution_h_xtol = 0.02 # figure has thick lines
    solution_h_ytol = 0.02

    solution_g = np.array([
        [-7.88626429632e-06, 0.0165907285134],
        [0.00503932288535, 0.39852447995],
        [0.0127008286492, 0.780631729201],
        [0.0281973379915, 1.17985020041],
        [0.0411584133625, 1.41298788867],
        [0.0554581820978, 1.57984941178],
        [0.0750417479116, 1.63092283093],
        [0.0920425621684, 1.61545983821],
        [0.106476397397, 1.5002789766],
        [0.124902653925, 1.23604180509],
        [0.139446896854, 0.888590744286],
        [0.148699456439, 0.673518515963],
        [0.153967480989, 0.590911869025],
        [0.160503222525, 0.591345613561],
        [0.16831456731, 0.658229021058],
        [0.183850507974, 0.9744938497],
        [0.192898024688, 1.19078056272],
        [0.205922190173, 1.29119242288],
        [0.217678638673, 1.30856389156],
        [0.226852335616, 1.25939894837],
        [0.247916547551, 0.945563089129],
        [0.25841316533, 0.86330343782],
        [0.266271827701, 0.830642474237],
        [0.272799682972, 0.847666947286],
        [0.283217438108, 0.931314581111],
        [0.301438651764, 1.09843635095],
        [0.314486476043, 1.14907602556],
        [0.323644400457, 1.1330925394],
        [0.347236160099, 1.00192819162],
        [0.364276405677, 0.903511556335],
        [0.370804260949, 0.920536029384],
        [0.389072792191, 0.988113428139],
        [0.412562030398, 1.07262855104],
        [0.430869992962, 1.05725230723],
        [0.455745242118, 0.975946893896],
        [0.472746056375, 0.960483901177],
        [0.498873249989, 0.995400336349],
        [0.517149667495, 1.04638700659],
        [0.536764778366, 1.03109751169],
        [0.582538627908, 0.9843615379],
        [0.628296704922, 0.970807021141],
        [0.702812044691, 0.959160980342],
        [0.735506524898, 0.928148245996],
        [0.783847353468, 0.981130141105],
        [0.863599172731, 0.953240367421],
        #[0.969493958136, 0.927085571882],  # These points are probably invalid
        #[1.00086551751, 0.929167545657],   # since G(1) should be 1
    ])
    solution_g_xtol = 0.026 # figure has thick lines
    solution_g_ytol = 0.05


class Problem9(TwoPointBVP):
    """
    Mathieu's equation
    """

    m = [2, 1]
    a = 0
    b = np.pi
    n_a = 2
    linear = False
    vectorized = True

    q = 5

    def f(self, x, z):
        u, du, a = z    
        return np.array([-(a - 2*self.q*np.cos(2*x))*u, 0*x])

    def df(self, x, z):
        u, du, a = z    
        return np.array([[-(a - 2*self.q*np.cos(2*x)), 0*x, -u],
                        [0*x, 0*x, 0*x]])

    def g(self, z_a, z_b):
        u_a, du_a, a_a = z_a
        u_b, du_b, a_b = z_b
        return np.array([u_a - 1, du_a - 0, du_b - 0])

    def dg(self, z_a, z_b):
        return (np.array([[1,0,0],
                         [0,1,0],
                         [0,0,0]]),
                np.array([[0,0,0],
                         [0,0,0],
                         [0,1,0]]))
    def guess(self, x):
        z = np.zeros((3,) + np.asarray(x).shape)
        dm = np.zeros((2,) + np.asarray(x).shape)
        z[0] = 1
        return z, dm
    
    def exact_solution(self, x):
        m = 2
        scale = special.mathieu_cem(m, self.q, 0)[0]
        y, yp = special.mathieu_cem(m, self.q, 180/np.pi * x)
        a = special.mathieu_a(m, self.q)
        return np.array([y/scale, yp/scale, np.ones(x.shape)*a]).T


class Problem10(TwoPointBVP):
    """
    A big trivial test problem::

       y'' = 0
       y(0) = 0, y(1) = 1
    """

    m = [2]*256

    a = 0
    b = 1
    n_a = 256
    linear = False
    homogenous = True
    vectorized = True

    def f(self, x, u):
        return np.zeros((self.n_a, x.size))

    def df(self, x, u):
        return np.zeros((self.n_a, 2*self.n_a, x.size))

    def g(self, u_a, u_b):
        return np.concatenate([u_a[::2] - 0, u_b[::2] - 1])

    def dg(self, u_a, u_b):
        du_a = np.zeros((2*self.n_a, 2*self.n_a))
        du_b = np.zeros((2*self.n_a, 2*self.n_a))
        i = np.arange(0, self.n_a)
        j = np.arange(0, 2*self.n_a, 2)
        du_a[i,j] = 1
        du_b[i+self.n_a,j] = 1
        return (du_a, du_b)

    guess = None

    def exact_solution(self, x):
        x = np.asarray(x)
        v = np.zeros((x.size, self.n_a*2))
        v[:,::2] = x[:,None]
        v[:,1::2] = 1
        return v
