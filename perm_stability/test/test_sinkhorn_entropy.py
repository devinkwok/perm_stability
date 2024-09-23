import unittest
import numpy as np

from itertools import product
from perm_stability.test.base_test import BaseTest
from perm_stability.sinkhorn_entropy import *


class TestSinkhornEntropy(BaseTest):

    def assert_entropy_curves_close(self, cost_a, cost_b):
        l_a, ent_a = entropy_curve(cost_a, max_l=0.5, n_points=5)
        l_b, ent_b = entropy_curve(cost_b, max_l=0.5, n_points=5)
        self.assert_array_equal(l_a, l_b)
        self.assert_array_close(ent_a, ent_b)

    def test_logspace(self):
        # sanity check: logspace computation is equal to regular computation
        for A, B in self.matrix_pairs:
            for reg in self.lambdas:
                p = sinkhorn_ot(A, B, reg=reg, logspace=False)
                p_log = sinkhorn_ot(A, B, reg=reg, logspace=True)
                self.assert_array_close(p, p_log)

    def test_entropy_bounds(self):
        # sanity check: any permutation matrix gives min entropy
        for n, m in product(self.n, self.m):
            p = np.eye(n)[np.random.permutation(n)]
            self.assertAlmostEqual(normalized_entropy(p), 0)
            # sanity check: if matrix has all identical vectors, should have max entropy
            for X in [np.ones([n, m]), np.repeat(np.random.randn(m), n).reshape(m, n).T]:
                [self.assert_array_equal(x, X[0]) for x in X]
                p = sinkhorn_ot(X, X)
                self.assert_array_close(p, np.full_like(p, 1 / n))
                self.assertAlmostEqual(normalized_entropy(p), 1)
        # sanity check: lambda -> 0 gives max entropy
        for A, B in self.matrix_pairs:
            p = sinkhorn_ot(A, A, reg=1e-7)
            self.assert_array_close(p, np.full_like(p, 1 / len(A)))
            self.assertAlmostEqual(normalized_entropy(p), 1)

    def test_marginal_shift(self):
        # sanity check: adding a row or col constant to the cost matrix doesn't affect l2 cost
        for A, B in self.matrix_pairs:
            n = len(A)
            for C in [np.ones([n, n]), np.repeat(np.random.randn(n), n).reshape(n, n), np.repeat(np.random.randn(n), n).reshape(n, n).T]:
                M = normalized_cost(A, B, "linear", True)
                p = sinkhorn_fp(self.l * M)
                p_c = sinkhorn_fp(self.l * (M + C))
                self.assert_array_close(p, p_c)

    def test_is_symmetric(self):
        # sanity check: sinkhorn entropy is symmetric
        for A, B in self.matrix_pairs:
            for cost in COST_FN_AND_STD.keys():
                C = normalized_cost(A, B, cost)
                C_T = normalized_cost(B, A, cost)
                self.assert_entropy_curves_close(C, C_T)

    def test_self_upper_bound(self):
        # sanity check: entropy(A, A) is lower bound on entropy(A, B)
        for A, B in self.matrix_pairs:
            ent_self_X = normalized_entropy(sinkhorn_ot(A, A, reg=self.l))
            ent_self_Y = normalized_entropy(sinkhorn_ot(B, B, reg=self.l))
            entropy = normalized_entropy(sinkhorn_ot(A, B, reg=self.l))
            self.assertLessEqual(ent_self_X, entropy)
            self.assertLessEqual(ent_self_Y, entropy)

    def test_l2_vs_dot_cost(self):
        # sanity check: L2 cost should be equivalent to -2*dot product similarity for entropy
        for A, B in self.matrix_pairs:
            C = 2 * l2_cost(A, B)
            C_true = np.sum((np.expand_dims(A, -1) - np.expand_dims(B.T, 0))**2, axis=1)
            self.assert_entropy_curves_close(C, C_true)


if __name__ == "__main__":
    unittest.main()
