import unittest
from itertools import product
import numpy as np

from perm_stability.test.base_test import BaseTest
from perm_stability.sinkhorn_entropy import COST_FN_AND_STD, normalized_cost, normalize_weight
from perm_stability.initializations import auto_normalize_weights
from perm_stability.nn_entropy import *

from nnperm.align.kernel import get_kernel_from_name


class TestNNEntropy(BaseTest):

    def assert_perm_shapes_correct(self, perm_dict, perm_dims):
        self.assert_keys_match(perm_dict, perm_dims)
        # check shapes are n^2
        for k, v in perm_dict.items():
            self.assertEqual(len(v.shape), 2)
            self.assertEqual(v.shape[0], perm_dims[k])
            self.assertEqual(v.shape[1], perm_dims[k])

    def test_nn_normalize_costs(self):
        for cost_name, (cost_fn, _) in COST_FN_AND_STD.items():
            kernel_fn = get_kernel_from_name(cost_name)
            costs, m_sizes, ref_costs = {}, {}, {}
            for n, m in product(self.n, self.m):
                A = np.random.randn(n, m)
                B = np.random.randn(n, m)
                cost = cost_fn(A, B)
                cost_true = -1 * kernel_fn(A, B)
                self.assert_array_close(cost, cost_true)

                key = f"{n}, {m}"
                costs[key] = cost
                m_sizes[key] = m
                ref_costs[key] = normalized_cost(A, B, cost_name, normalize_m=True)

            normalized = nn_normalize_costs(costs, m_sizes, cost_name)
            self.assert_is_normalized(normalized)
            for k, v in normalized.items():
                self.assert_array_close(v, ref_costs[k])

    def test_cost_distribution(self):
        for pairname, ckpt_0, ckpt_1 in self.ckpt_pairs:
            if not ("self" in pairname):
                continue

            ckpt_0 = auto_normalize_weights(ckpt_0)
            ckpt_1 = auto_normalize_weights(ckpt_1)
            for cost_name in COST_FN_AND_STD.keys():
                costs = nn_cost_matrices(self.perm_spec, ckpt_0, ckpt_1, cost=cost_name, normalize_m=True)

            # sanity check: for self entropy, cost should be minimized along diagonal
            # meanwhile, the non-diagonal costs should be somewhat close to 0
            for k, v in costs.items():
                diag_mean = np.mean(np.diag(v)) / np.std(v)
                off_diag_mean = (np.sum(v) - np.sum(np.diag(v))) / v.size / np.std(v)
                self.assertLess(diag_mean, 0)
                self.assertLess(diag_mean, off_diag_mean)
                self.assertGreater(off_diag_mean, -2)
                self.assertLess(off_diag_mean, 0.5)

    def bernoulli_like(self, array, p):
        # Bernoulli X with probability p has variance Var[X] = E[X^2] - E[X]^2 = p - p^2
        # standardize to mean 0 and std 1
        mean = p
        std = (p - p**2)**(1/2)
        X = np.random.uniform(0, 1, size=array.shape) < p
        return (X - mean) / std

    def test_nn_cost_matrices(self):
        # only look at IID randomly initialized networks, see if their costs are normalized
        for pairname, ckpt_0, ckpt_1 in self.ckpt_pairs:
            if not ("init" in pairname and "iid" in pairname):
                continue

            ckpt_0 = auto_normalize_weights(ckpt_0)
            ckpt_1 = auto_normalize_weights(ckpt_1)

            for cost_name in COST_FN_AND_STD.keys():
                costs, perms = nn_cost_matrices(self.perm_spec, ckpt_0, ckpt_1, cost=cost_name, normalize_m=True, return_minimizing_permutation=True)
                # apply permutation so that we can get a matching cost manually
                ckpt_1 = self.perm_spec.apply_permutation(ckpt_1, perms)

                #TODO FIXME cosine cost has wrong scaling
                if cost_name == "cosine":
                    continue

                for group, layers in self.perm_spec.group_to_axes.items():
                    C = 0
                    m = 0
                    for layer, dim, _ in layers:
                        A, B = normalize_weight(ckpt_0[layer], ckpt_1[layer], perm_axis=dim)
                        C = C + normalized_cost(A, B, cost=cost_name, normalize_m=False)
                        m += A.shape[1]

                    C = C / COST_FN_AND_STD[cost_name][1](m)
                    self.assert_array_close(C, costs[group])
                ms = get_non_permuted_sizes(ckpt_0, self.perm_spec)
                nn_normalize_costs(costs, ms, cost_name)

                self.assert_is_normalized(costs)

    def test_nn_sinkhorn_entropies(self):
        for pairname, ckpt_0, ckpt_1 in self.ckpt_pairs:
            A = auto_normalize_weights(ckpt_0)
            B = auto_normalize_weights(ckpt_1)

            for cost_name, _ in COST_FN_AND_STD.items():
                costs = nn_cost_matrices(self.perm_spec, A, B, cost=cost_name)
                self.assert_perm_shapes_correct(costs, self.perm_dims)

                perms = nn_sinkhorn(costs)
                self.assert_perm_shapes_correct(perms, self.perm_dims)

                entropies = nn_normalized_entropy(perms)
                self.assert_keys_match(entropies, self.perm_dims)

                entropy_curve = nn_entropy_curve(costs, n_points=self.n_points)
                self.assert_keys_match(entropy_curve, self.perm_dims)
                for v in entropy_curve.values():
                    self.assertLessEqual(len(v), self.n_points)
                    for a, b in v:
                        self.assertIsInstance(a, float)
                        self.assertIsInstance(b, float)



if __name__ == "__main__":
    unittest.main()
