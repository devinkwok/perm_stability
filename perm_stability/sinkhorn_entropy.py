from typing import Union, List, Tuple
import warnings
import numpy as np
import scipy.optimize
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore


def normalize_weight(*args, init_std=1, perm_axis=0):
    output = []
    for A in args:
        A = np.moveaxis(A, perm_axis, 0)
        A = A.reshape([A.shape[0], -1])
        A = A.astype(np.float128)
        A = A / init_std  # get rid of effect of initial standard deviation
        output.append(A)
    return output


def l2_cost(A, B):
    M = A @ B.T  # minimizing 1/2 squared L2 distance is equivalent to maximizing dot product
    # multiply by -1 to turn similarity into cost
    return -1 * M


def cos_cost(A, B):
    return -1 * cosine_similarity(A, B)


COST_FN_AND_STD = {
    "linear": (l2_cost, lambda m: m**(1/2)),  # sum of m IID linear costs has std \sqrt(m)
    "cosine": (cos_cost, lambda m: 1 / m**(1/2)),  # sum of m IID cosine costs has std 1 / \sqrt(m)
}


def normalized_cost(A, B, cost="linear", normalize_m=True):
    """Divide by function of m, where m is non-permuted dimensions,
    to get consistent cost scale regardless of m.

    This is based on normalizing in terms of standard deviation of cost over IID random matrices.
    """
    cost_fn, cost_std = COST_FN_AND_STD[cost]
    cost = cost_fn(A, B)
    if normalize_m:
        cost = cost / cost_std(A.shape[1])
    return cost


def lp_perm(A, B, cost="linear"):
    M = normalized_cost(A, B, cost, False)
    _, c = scipy.optimize.linear_sum_assignment(M.astype(np.float64))
    return np.argsort(c)  # turn into row indices


def logsum(x, axis=0):
    # computes \log sum_{i=1}^n exp(x_i)
    m = np.max(x, axis=axis)  # shift by max
    return m + np.log(np.sum(np.exp(x - m), axis=axis))


def normalize_first_dim(M, stop_rtol, stop_atol, logspace):
    M_sum = logsum(M, axis=0) if logspace else np.sum(M, axis=0)
    is_normalized = np.allclose(M_sum, 0 if logspace else 1, rtol=stop_rtol, atol=stop_atol)
    normalized_value = M - M_sum if logspace else M / M_sum
    return normalized_value, is_normalized


def sinkhorn_fp(P, max_iter=100, stop_rtol=1e-4, stop_atol=1e-4, logspace=True):
    P = P if logspace else np.exp(P)
    for i in range(max_iter):
        P, done_cols = normalize_first_dim(P.T, stop_rtol=stop_rtol, stop_atol=stop_atol, logspace=logspace)
        P, done_rows = normalize_first_dim(P.T, stop_rtol=stop_rtol, stop_atol=stop_atol, logspace=logspace)
        if done_cols and done_rows:
            break
    return np.exp(P) if logspace else P


def sinkhorn_ot(A, B, reg=1, cost="linear", max_iter=100, stop_rtol=1e-4, stop_atol=1e-4, logspace=True, normalize_m=True):
    M = normalized_cost(A, B, cost, normalize_m)
    return sinkhorn_fp(-reg * M, max_iter=max_iter, stop_rtol=stop_rtol, stop_atol=stop_atol, logspace=logspace)


def sinkhorn_perm(A, B, reg=1e2, logspace=True, normalize_m=True):
    P = sinkhorn_ot(A, B, reg=reg, logspace=logspace, normalize_m=normalize_m)
    return np.argmax(P, axis=0)


def normalized_entropy(P, rtol=3e-2, atol=3e-2) -> Union[float, None]:
    n = len(P)
    P = P / n  # normalize to sum to 1
    if not (np.allclose(np.sum(P), 1, rtol=rtol, atol=atol)
            and np.allclose(np.sum(P, axis=0), 1/n, rtol=3e-2, atol=3e-2)
            and np.allclose(np.sum(P, axis=1), 1/n, rtol=3e-2, atol=3e-2)
            ):
        warnings.warn("normalized_entropy: permutation distribution is not doubly stochastic")
        return None  # Sinkhorn output is too off from doubly stochastic

    # by convention when computing entropy, 0 log(0) = 1 log(1) = 0
    P[P == 0] = 1
    h = -P * np.log(P)
    # KL from uniform is h(r) + h(c) - h(P), divide by maximum possible KL
    kl = (2*np.log(n) - np.sum(h)) / np.log(n)
    # reverse so we get 0 to 1 as increasing entropy (decreasing permutation stability)
    return (1 - kl).astype(np.float64)


def entropy_curve(C, reg=None, min_reg=0.01, max_reg=100, n_points=40, max_iter=100, rtol=3e-2, atol=3e-2, logspace=True, return_permutation=False):
    if reg is None:
        reg = np.exp(np.linspace(np.log(min_reg), np.log(max_reg), n_points))

    permutations = [sinkhorn_fp(-reg * C, max_iter=max_iter, logspace=logspace) for reg in reg]
    entropies = [normalized_entropy(p, rtol=rtol, atol=atol) for p in permutations]

    # remove entropies if Sinkhorn failed to reach a doubly stochastic matrix
    permutations = [(reg, p) for reg, p, ent in zip(reg, permutations, entropies) if ent is not None]
    entropies = [(reg, ent) for reg, ent in zip(reg, entropies) if ent is not None]

    if return_permutation:
        return entropies, permutations
    return entropies


# draws a random permutation with at least fixed_points_fraction fixed points
def randperm(n, fixed_points_fraction):
    # expected number of fixed points in random perm is 1, so subtract 1 from guaranteed fixed points
    n_permuted = int(np.round(n * (1 - fixed_points_fraction))) + 1
    n_permuted = n if n_permuted > n else n_permuted
    p = np.random.permutation(n_permuted)
    id = np.arange(n)
    # randomly distribute permuted elements among fixed points
    idx_to_permute = np.random.permutation(n)[:n_permuted]
    id[idx_to_permute] = idx_to_permute[p]
    return id


def fixed_points(p, q=None):
    n = len(p)
    q = np.arange(n) if q is None else q
    return np.count_nonzero(p == q) / n


def l2_distance(a, b=None):
    b = np.zeros_like(a) if b is None else b
    return np.linalg.norm((a - b).flatten())
