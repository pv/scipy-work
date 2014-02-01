"""
Benchmark the Levinson-Durbin implementation.

This algorithm solves Toeplitz systems.

"""
from __future__ import division, print_function, absolute_import

import time

import numpy as np
from numpy.testing import assert_allclose
import scipy.linalg


def bench_levinson_durbin():
    np.random.seed(1234)
    print()
    print('               Levinson-Durbin vs. generic solver')
    print('          T x = y; T.shape == (n, n); y.shape == (n, m)')
    print('==============================================================')
    print('   n   |   m   |        dtype       | L-D time  | generic time')
    print('       |       |                    | (seconds) |             ')
    print('--------------------------------------------------------------')
    fmt = ' %5d | %5d | %18s | %9.3g | %9.3g '
    for dtype in (np.float64, np.complex128):
        for n in (100, 300, 1000):
            for m in (1, 10, 100, 1000, 10000):
                # Sample a random Toeplitz matrix representation and rhs.
                c = np.random.randn(n)
                r = np.random.randn(n)
                y = np.random.randn(n, m)
                if dtype == np.complex128:
                    c = c + 1j*np.random.rand(n)
                    r = r + 1j*np.random.rand(n)
                    y = y + 1j*np.random.rand(n, m)

                # generic solver
                T = scipy.linalg.toeplitz(c, r=r)
                count = 0
                tm = time.clock()
                while time.clock() - tm < 0.1 or count == 0:
                    x_generic = scipy.linalg.solve(T, y)
                    count += 1
                nseconds_gen = (time.clock() - tm)/count

                # toeplitz-specific solver
                count = 0
                tm = time.clock()
                while time.clock() - tm < 0.1 or count == 0:
                    x_toeplitz = scipy.linalg.solve_toeplitz(c, r=r, y=y)
                    count += 1
                nseconds_ld = (time.clock() - tm)/count

                print(fmt % (n, m, T.dtype, nseconds_ld, nseconds_gen))

                # Check that the solutions are the same.
                assert_allclose(x_generic, x_toeplitz, atol=1e-8*abs(x_generic).max(),
                                err_msg=repr(abs(x_generic - x_toeplitz).max()))
