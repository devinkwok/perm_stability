import unittest
from perm_stability.test.test_utils import *
from perm_stability.sinkhorn_entropy import normalize_weight
from perm_stability.initializations import is_normalization_weight

from nnperm.spec.perm_spec import PermutationSpec
from nnperm.align.kernel import get_kernel_from_name


class BaseTest(unittest.TestCase):

    def setUp(self) -> None:
        self.n = np.arange(10, 30, 7)
        self.m = np.arange(8, 40, 11)
        self.l = 1e-1
        # don't have too high lambda as Sinkhorn is quite unstable for small random matrices
        self.lambdas = np.exp(np.linspace(-3, 0, 5))

        matrix_pairs = list(yield_matrix_pairs())
        matrix_pairs = [matrix_pairs[i] for i in range(0, len(matrix_pairs), 5)][:10]
        self.matrix_pairs = {k: normalize_weight(a, b, init_std=std) for k, a, b, std in matrix_pairs}

        self.ckpt_init = get_test_ckpts()["init"][0]
        self.ckpt_trained = get_test_ckpts()["trained"][0]
        self.ckpt_pairs = list(yield_ckpt_pairs(runs=1))

        self.perm_spec = PermutationSpec.from_residual_model(self.ckpt_init)
        self.perm_dims = self.perm_spec.get_sizes(self.ckpt_init)
        self.n_points = 10

    def assert_array_close(self, a, b):
        np.testing.assert_allclose(a, b, rtol=5e-2, atol=5e-2)

    def assert_array_equal(self, a, b):
        np.testing.assert_array_equal(a, b)

    def assert_keys_match(self, dict_1, dict_2):
        keys_1 = {k: None for k in dict_1.keys()}
        keys_2 = {k: None for k in dict_2.keys()}
        self.assertDictEqual(keys_1, keys_2)

    def assert_is_normalized(self, array_dict):
        norm_weights = []
        for k, v in array_dict.items():
            if np.std(v) != 0:
                if is_normalization_weight(k, v.shape):
                    # stack all norm weights to get more samples
                    norm_weights.append(v)
                else:
                    delta = min(2 / v[0].size**(1/2), 0.5)
                    self.assertAlmostEqual(np.mean(v), 0, delta=delta)
                    self.assertAlmostEqual(np.std(v), 1, delta=delta)

        # do all norm weights at once for closer statistical approximation
        if len(norm_weights) > 0:
            norm_weights = np.concatenate(norm_weights)
            delta = min(2 / norm_weights.size**(1/2), 0.5)
            self.assertAlmostEqual(np.mean(norm_weights), 0, delta=delta)
            self.assertAlmostEqual(np.std(norm_weights), 1, delta=delta)


if __name__ == "__main__":
    unittest.main()
