import numpy as np
from sinkhorn_entropy import lp_perm


# compute empirical P_\sigma
def empirical_P(X, Y, sigma=1, n_samples=1000, add_noise_to_both=False):
    n = len(X)
    P_sigma = np.zeros((n, n))
    for _ in range(n_samples):
        Z = np.random.randn(*X.shape) * sigma
        Y_noise = Y + np.random.randn(*Y.shape) * sigma if add_noise_to_both else Y
        P = lp_perm(X + Z, Y_noise)
        P_sigma += P
    return P_sigma / n_samples
