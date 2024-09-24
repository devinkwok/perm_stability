import unittest
import numpy as np

from perm_stability.test.base_test import BaseTest
from perm_stability.initializations import *


class TestInitializations(BaseTest):

    def test_normalize_weights(self):
        # constant std and mean
        std = 2
        mean = -1
        normalized = auto_normalize_weights(self.ckpt_init, std_mean=(std, mean))
        for k, v in normalized.items():
            self.assertAlmostEqual(np.std(v), np.std(self.ckpt_init[k]) / std, delta=1e-2)
            self.assertAlmostEqual(np.mean(v), (np.mean(self.ckpt_init[k]) - mean) / std, delta=1e-2)

        # dict of stds and means
        std_mean = {k: (np.random.uniform() + 0.1, np.random.randn()) for k in self.ckpt_init.keys()}
        normalized = auto_normalize_weights(self.ckpt_init, std_mean=std_mean)

        for k, v in normalized.items():
            self.assertAlmostEqual(np.std(v), np.std(self.ckpt_init[k]) / std_mean[k][0], delta=1e-2)
            self.assertAlmostEqual(np.mean(v), (np.mean(self.ckpt_init[k]) - std_mean[k][1]) / std_mean[k][0], delta=1e-2)

        # infer std mean from network layers
        normalized = auto_normalize_weights(self.ckpt_init)
        self.assert_is_normalized(normalized)


if __name__ == "__main__":
    unittest.main()
