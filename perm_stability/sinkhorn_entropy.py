import numpy as np
import scipy.optimize
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore


def normalize_weight(*args, init_std=1, perm_axis=0):
    output = []
    for X in args:
        X = np.moveaxis(X, perm_axis, 0)
        X = X.reshape([X.shape[0], -1])
        X = X.astype(np.float128)
        X = X / init_std  # get rid of effect of initial standard deviation
        output.append(X)
    return output


def l2_cost(X, Y):
    M = X @ Y.T  # minimizing 1/2 squared L2 distance is equivalent to maximizing dot product
    # multiply by -1 to turn similarity into cost
    return -1 * M


def cos_cost(X, Y):
    return -1 * cosine_similarity(X, Y)


COST_FN_AND_STD = {
    "linear": (l2_cost, lambda m: m**(1/2)),  # sum of m IID linear costs has std \sqrt(m)
    "cosine": (cos_cost, lambda m: 1 / m**(1/2)),  # sum of m IID cosine costs has std 1 / \sqrt(m)
}


def normalized_cost(X, Y, cost="linear", normalize_m=True):
    """Divide by function of m, where m is non-permuted dimensions,
    to get consistent cost scale regardless of m.

    This is based on normalizing in terms of standard deviation of cost over IID random matrices.
    """
    cost_fn, cost_std = COST_FN_AND_STD[cost]
    cost = cost_fn(X, Y)
    if normalize_m:
        cost = cost / cost_std(X.shape[1])
    return cost


def lp_perm(X, Y, cost="linear"):
    M = normalized_cost(X, Y, cost, False)
    _, c = scipy.optimize.linear_sum_assignment(M.astype(np.float64))
    return np.argsort(c)  # turn into row indices


def logsum(x, axis=0):
    # computes \log sum_{i=1}^n exp(x_i)
    m = np.max(x, axis=axis)  # shift by max
    return m + np.log(np.sum(np.exp(x - m), axis=axis))


def rescale(M, rtol, atol, logspace):
    M_sum = logsum(M, axis=0) if logspace else np.sum(M, axis=0)
    return M - M_sum if logspace else M / M_sum, np.allclose(M_sum, 0 if logspace else 1, rtol=rtol, atol=atol)


def sinkhorn_fp(P, max_iter=100, rtol=1e-4, atol=1e-4, logspace=True):
    P = P if logspace else np.exp(P)
    for i in range(max_iter):
        P, done_cols = rescale(P.T, rtol=rtol, atol=atol, logspace=logspace)
        P, done_rows = rescale(P.T, rtol=rtol, atol=atol, logspace=logspace)
        if done_cols and done_rows:
            break
    return np.exp(P) if logspace else P


def sinkhorn_ot(X, Y, reg=1, cost="linear", max_iter=100, rtol=1e-4, atol=1e-4, logspace=True, normalize_m=True):
    M = normalized_cost(X, Y, cost, normalize_m)
    return sinkhorn_fp(-reg * M, max_iter=max_iter, rtol=rtol, atol=atol, logspace=logspace)


def sinkhorn_perm(X, Y, reg=1e2, logspace=True, normalize_m=True):
    P = sinkhorn_ot(X, Y, reg=reg, logspace=logspace, normalize_m=normalize_m)
    return np.argmax(P, axis=0)


def normalized_entropy(P):
    n = len(P)
    P = P / n  # normalize to sum to 1
    # assert abs(np.sum(P) - 1) < 1e-4, np.sum(P)
    # assert np.allclose(np.sum(P, axis=0), 1/n, rtol=3e-2, atol=3e-2), np.sum(P, axis=0)
    # assert np.allclose(np.sum(P, axis=1), 1/n, rtol=3e-2, atol=3e-2), np.sum(P, axis=1)
    P[P == 0] = 1  # by convention when computing entropy, 0 log(0) = 1 log(1) = 0
    h = -P * np.log(P)
    # KL from uniform is h(r) + h(c) - h(P), divide by maximum possible KL
    # reverse so we get 0 to 1 as increasing entropy (decreasing permutation stability)
    return 1 - (2*np.log(n) - np.sum(h)) / np.log(n)


def entropy_curve(C, min_l=0.01, max_l=100, n_points=40):
    lambdas = np.exp(np.linspace(np.log(min_l), np.log(max_l), n_points))
    entropies = [normalized_entropy(sinkhorn_fp(-reg * C)) for reg in lambdas]
    return lambdas, entropies


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
