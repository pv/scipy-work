"""Test functions for the sparse.linalg.norm module
"""

from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_raises, assert_allclose, TestCase

from scipy import sparse
from scipy.sparse.linalg import norm


class TestNorm(TestCase):
    def test_norm(self):
        a = np.arange(9) - 4
        b = a.reshape((3, 3))
        b = sparse.csr_matrix(b)

        # Frobenius norm is the default
        assert_allclose(norm(b), 7.745966692414834)        
        assert_allclose(norm(b, 'fro'), 7.745966692414834)

        assert_allclose(norm(b, np.inf), 9)
        assert_allclose(norm(b, -np.inf), 2)
        assert_allclose(norm(b, 1), 7)
        assert_allclose(norm(b, -1), 6)
        
        # _multi_svd_norm is not implemented for sparse matrix
        assert_raises(NotImplementedError, norm, b, 2)
        assert_raises(NotImplementedError, norm, b, -2)
                
    def test_norm_axis(self):
        a = np.array([[1, 2, 3],
                      [-1, 1, 4]])

        c = sparse.csr_matrix(a)

        for ord in (None, 'f', 'fro'):
            assert_allclose(norm(c, ord, axis=0), np.sqrt(np.power(np.asmatrix(a), 2).sum(axis=0)))
            assert_allclose(norm(c, ord, axis=1), np.sqrt(np.power(np.asmatrix(a), 2).sum(axis=1)))

        assert_allclose(norm(c, 'f', axis=0), np.sqrt(np.power(np.asmatrix(a), 2).sum(axis=0)))
        assert_allclose(norm(c, 'fro', axis=1), np.sqrt(np.power(np.asmatrix(a), 2).sum(axis=1)))

        assert_allclose(norm(c, axis=0), np.sqrt(np.power(np.asmatrix(a), 2).sum(axis=0)))
        assert_allclose(norm(c, axis=1), np.sqrt(np.power(np.asmatrix(a), 2).sum(axis=1)))

        assert_allclose(norm(c, np.inf, axis=0), max(abs(np.asmatrix(a)).sum(axis=0)))
        assert_allclose(norm(c, np.inf, axis=1), max(abs(np.asmatrix(a)).sum(axis=1)))

        assert_allclose(norm(c, -np.inf, axis=0), min(abs(np.asmatrix(a)).sum(axis=0)))
        assert_allclose(norm(c, -np.inf, axis=1), min(abs(np.asmatrix(a)).sum(axis=1)))

        assert_allclose(norm(c, 1, axis=0), abs(np.asmatrix(a)).sum(axis=0))
        assert_allclose(norm(c, 1, axis=1), abs(np.asmatrix(a)).sum(axis=1))

        assert_allclose(norm(c, -1, axis=0), min(abs(np.asmatrix(a)).sum(axis=0)))
        assert_allclose(norm(c, -1, axis=1), min(abs(np.asmatrix(a)).sum(axis=1)))

        assert_allclose(norm(c, 0, axis=0), np.sum(np.asmatrix(a) != 0, axis=0))
        assert_allclose(norm(c, 0, axis=1), np.sum(np.asmatrix(a) != 0, axis=1))

        # _multi_svd_norm is not implemented for sparse matrix
        assert_raises(NotImplementedError, norm, c, 2, 0)
        # assert_raises(NotImplementedError, norm, c, -2, 0)
