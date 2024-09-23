import unittest
from itertools import product
import numpy as np

from perm_stability.test.base_test import BaseTest
from perm_stability.sinkhorn_entropy import COST_FN_AND_STD, normalized_cost, normalize_weight
from perm_stability.initializations import nn_normalize_weights
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
            similarities, m_sizes = {}, {}
            for n, m in product(self.n, self.m):
                A = np.random.randn(n, m)
                B = np.random.randn(n, m)
                sim = -1 * cost_fn(A, B)
                sim_true = kernel_fn(A, B)
                self.assert_array_close(sim, sim_true)

                similarities[f"{n}, {m}"] = sim
                m_sizes[f"{n}, {m}"] = m

            normalized = nn_normalize_costs(similarities, m_sizes, cost_name)
            self.assert_is_normalized(normalized)

    def test_nn_cost_matrices(self):
        # only look at IID randomly initialized networks, see if their costs are normalized
        for pairname, ckpt_0, ckpt_1 in self.ckpt_pairs:
            if not ("init" in pairname and "iid" in pairname):
                continue

            ckpt_0 = nn_normalize_weights(ckpt_0)
            ckpt_1 = nn_normalize_weights(ckpt_1)

            for cost_name in COST_FN_AND_STD.keys():
                if cost_name == "linear":
                    continue  #TODO remove DEBUG
                costs = nn_cost_matrices(self.perm_spec, ckpt_0, ckpt_1, align_kernel=cost_name, normalize_m=True)

                #TODO FIXME wrong scale for cosine cost
                print(pairname, cost_name)
                for group, layers in self.perm_spec.group_to_axes.items():
                    C = 0
                    m = 0
                    for layer, dim, _ in layers:
                        A, B = normalize_weight(ckpt_0[layer], ckpt_1[layer], perm_axis=dim)
                        C = C - normalized_cost(A, B, cost=cost_name, normalize_m=False)
                        print("\t", layer, A.shape, np.std(A), np.mean(A), C.shape, np.std(C), np.mean(C))
                        m += A.shape[1]
                    print(group, len(C), m, np.std(C), np.mean(C), len(costs[group]), np.std(costs[group]), np.mean(costs[group]))

                self.assert_is_normalized(costs)

    def test_nn_sinkhorn_ot(self):
        for pairname, ckpt_0, ckpt_1 in self.ckpt_pairs:
            ckpt_0 = nn_normalize_weights(ckpt_0)
            ckpt_1 = nn_normalize_weights(ckpt_1)

            for cost_name, (cost_fn, _) in COST_FN_AND_STD.items():
                costs = nn_cost_matrices(self.perm_spec, ckpt_0, ckpt_1, align_kernel=cost_name)
                self.assert_perm_shapes_correct(costs, self.perm_dims)

                perms = nn_sinkhorn_ot(costs)
                self.assert_perm_shapes_correct(perms, self.perm_dims)

                entropies = nn_normalized_entropy(perms)
                self.assert_keys_match(entropies, self.perm_dims)

                lambdas, entropy_curve = nn_entropy_curve(perms)
                self.assert_keys_match(entropy_curve, self.perm_dims)
                for k, v in entropy_curve.items():
                    self.assertEqual(len(v), len(lambdas))


if __name__ == "__main__":
    unittest.main()
