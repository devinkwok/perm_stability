import unittest
import numpy as np

from itertools import product
from perm_stability.test.base_test import BaseTest
from perm_stability.sinkhorn_entropy import *


class TestSinkhornEntropy(BaseTest):

    def assert_entropy_curves_close(self, cost_a, cost_b):
        # increase tolerance so entropy_curve doesn't return None
        curve_a = entropy_curve(cost_a, max_reg=0.5, n_points=self.n_points, rtol=1, atol=1)
        curve_b = entropy_curve(cost_b, max_reg=0.5, n_points=self.n_points, rtol=1, atol=1)
        self.assert_curves_close(curve_a, curve_b)

    def assert_curves_close(self, curve_a, curve_b):
        l_a, ent_a = list(zip(*curve_a))
        l_b, ent_b = list(zip(*curve_b))
        self.assert_array_equal(l_a, l_b)
        self.assert_array_close(ent_a, ent_b)

    def test_logspace(self):
        # sanity check: logspace computation is equal to regular computation
        for A, B in self.matrix_pairs.values():
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
        for A, B in self.matrix_pairs.values():
            p = sinkhorn_ot(A, A, reg=1e-7)
            self.assert_array_close(p, np.full_like(p, 1 / len(A)))
            self.assertAlmostEqual(normalized_entropy(p), 1)

    def test_marginal_shift(self):
        # sanity check: adding a row or col constant to the cost matrix doesn't affect l2 cost
        for A, B in self.matrix_pairs.values():
            n = len(A)
            for C in [np.ones([n, n]), np.repeat(np.random.randn(n), n).reshape(n, n), np.repeat(np.random.randn(n), n).reshape(n, n).T]:
                M = normalized_cost(A, B, "linear", True)
                p = sinkhorn_fp(self.l * M)
                p_c = sinkhorn_fp(self.l * (M + C))
                self.assert_array_close(p, p_c)

    def test_is_symmetric(self):
        # sanity check: sinkhorn entropy is symmetric
        for A, B in self.matrix_pairs.values():
            for cost in COST_FN_AND_STD.keys():
                C = normalized_cost(A, B, cost)
                C_T = normalized_cost(B, A, cost)
                self.assert_entropy_curves_close(C, C_T)

    def test_self_upper_bound(self):
        # sanity check: entropy(A, A) is lower bound on entropy(A, B)
        for A, B in self.matrix_pairs.values():
            ent_self_X = normalized_entropy(sinkhorn_ot(A, A, reg=self.l))
            ent_self_Y = normalized_entropy(sinkhorn_ot(B, B, reg=self.l))
            entropy = normalized_entropy(sinkhorn_ot(A, B, reg=self.l))
            self.assertLessEqual(ent_self_X, entropy)
            self.assertLessEqual(ent_self_Y, entropy)

    def test_cost_distribution(self):
        # sanity check: for self entropy, cost should be minimized along diagonal
        # meanwhile, the non-diagonal costs should be somewhat close to 0
        for pairname, (A, B) in self.matrix_pairs.items():
            if not ("self" in pairname):
                continue
            for cost_name in COST_FN_AND_STD.keys():
                C = normalized_cost(A, B, cost=cost_name)
                diag_mean = np.mean(np.diag(C)) / np.std(C)
                off_diag_mean = (np.sum(C) - np.sum(np.diag(C))) / C.size / np.std(C)
                self.assertLess(diag_mean, -3)
                self.assertAlmostEqual(off_diag_mean, 0, delta=0.1)

    def test_l2_vs_dot_cost(self):
        # sanity check: L2 cost should be equivalent to -2*dot product similarity for entropy
        for A, B in self.matrix_pairs.values():
            C = 2 * l2_cost(A, B)
            C_true = np.sum((np.expand_dims(A, -1) - np.expand_dims(B.T, 0))**2, axis=1)
            self.assert_entropy_curves_close(C, C_true)

    def test_translate_by_constant(self):
        # sanity check: entropy between A and interpolation of A to B increases as interpolant increases
        alphas = np.linspace(0, 1, 3)
        for k, (A, B) in self.matrix_pairs.items():
            entropies = []
            for a in alphas:
                if "self" in k:
                    B = np.random.randn(*A.shape) * 5
                C = (1 - a) * A + a * B
                entropies.append(normalized_entropy(sinkhorn_ot(A, C, reg=self.l)))
            ent_prev = entropies[0]
            for s, ent in zip(alphas, entropies):
                self.assertGreaterEqual(ent, ent_prev)

    def test_sinkhorn_entropies(self):
        for A, B in self.matrix_pairs.values():
            for cost in COST_FN_AND_STD.keys():
                curve = sinkhorn_entropies(A, B, cost=cost, n_points=self.n_points)
                curve_true = entropy_curve(normalized_cost(*normalize_weight(A, B), cost=cost), n_points=self.n_points)
                self.assert_curves_close(curve, curve_true)


if __name__ == "__main__":
    unittest.main()
