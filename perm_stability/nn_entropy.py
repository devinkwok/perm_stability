from copy import deepcopy
from typing import Dict, Union, Tuple, List
import numpy as np
import torch
import torch.nn as nn

from nnperm.align.weight_align import WeightAlignment
from nnperm.align.activation_align import ActivationAlignment
from nnperm.spec.perm_spec import PermutationSpec

from perm_stability.sinkhorn_entropy import sinkhorn_fp, normalized_entropy, entropy_curve, COST_FN_AND_STD
from perm_stability.initializations import auto_normalize_weights


def nn_sinkhorn_entropies(
        perm_spec: PermutationSpec,
        A: Union[Dict[str, torch.Tensor], nn.Module],
        B: Union[Dict[str, torch.Tensor], nn.Module] = None,
        reg: Union[None, float, List[float]] = None,
        cost: str = "linear",
        std_mean: Union[Tuple[float, float], Dict[str, Tuple[float, float]], None] = None,
        align_obj=None,
        align_type: str = "weight",
        dataloader: Union[torch.utils.data.DataLoader, None] = None,
        min_reg=0.01,
        max_reg=100,
        n_points=40,
        max_iter=100,
        rtol=3e-2,
        atol=3e-2,
        logspace=True,
        normalize_m: bool = True,
        return_permutation: bool = False,
):
    """Computes Sinkhorn entropies for a neural network and itself, or two neural networks,
    over a range of regularization terms $\lambda$.

    Args:
        perm_spec (PermutationSpec): from nnperm, specifies how to assign parameters and dims to permutations
        A (Union[Dict[str, torch.Tensor], nn.Module]): first model or state dict
        B (Union[Dict[str, torch.Tensor], nn.Module], optional): second model or state dict. If None, use first model here. Defaults to None.
        reg (Union[None, float, List[float]], optional): Value of regularizer $\lambda$, either a single value,
        cost (str, optional): Method for computing cost that the permutations minimize between X and Y. Defaults to "linear".
        std_mean (Union[Tuple[float, float], Dict[str, Tuple[float, float]], None], optional):
            Normalize all layers by subtracting mean and dividing std, where mean and std are fixed per layer.
            If None, automatically means and stds based on layer type and their values at random initialization.
            If a tuple of floats, use these as the means and stds for all layers.
            If a dictionary, assign means and stds by key to corresponding named parameters.
            Set to (1, 0) to avoid normalizing. Defaults to None.
        align_obj (Literal[&quot;weight&quot;, &quot;activation&quot;], optional): Provide custom alignment object that implements
            fit(A, B) -> Tuple[Dict[str, List[int]], Dict[List[nn.ndarray]]],
            where the return values are the permutations and similarity matrices per iteration of the alignment algorithm.
        align_type (Literal[&quot;weight&quot;, &quot;activation&quot;], optional): Alignment algorithm to use. Defaults to "weight".
        dataloader (Union[torch.utils.data.DataLoader, None], optional): Data for computing activations if doing activation alignment. Defaults to None.
        min_reg (float, optional): If reg is None, this is the smallest reg to include. Defaults to 0.01.
        max_reg (int, optional): If reg is None, this is the largest reg to include. Defaults to 100.
        n_points (int, optional): If reg is None, this is the number of reg to compute
            (exponential scaling between min_reg and max_reg). Defaults to 40.
        max_iter (int, optional): Maximum Sinkhorn iterations. Defaults to 100.
        rtol (float, optional): Relative tolerance allowed to compute entropy from doubly stochastic permutation matrix. Defaults to 1e-4.
        atol (float, optional): Absolute tolerance allowed to compute entropy from doubly stochastic permutation matrix. Defaults to 1e-4.
        logspace (bool, optional): Do Sinkhorn algorithm in logspace. Defaults to True.
        normalize_m (bool, optional): Normalize cost by dividing out effect of summing non-permuted dimensions. Defaults to True.
        return_permutation (bool, optional): Also return permutation distribution for given lambdas. Defaults to False.

    Returns:
        Dict[List[Tuple[float, float]]]: Dictionary of lambda and corresponding Sinkhorn entropies.
            If return_permutation, permutation(s) are returned in the same format as the second tuple element.
    """
    A = auto_normalize_weights(A, std_mean=std_mean)
    B = deepcopy(A) if B is None else auto_normalize_weights(B, std_mean=std_mean)

    costs = nn_cost_matrices(perm_spec, A, B, cost=cost, align_obj=align_obj, align_type=align_type,
                             dataloader=dataloader, normalize_m=normalize_m)

    if reg is None or isinstance(reg, list):
        return nn_entropy_curve(costs, reg=reg, min_reg=min_reg, max_reg=max_reg, n_points=n_points,
                                rtol=rtol, atol=atol, return_permutation=return_permutation)
    else:
        perms = nn_sinkhorn(costs, reg=reg, max_iter=max_iter, logspace=logspace)
        entropies = nn_normalized_entropy(perms, rtol=rtol, atol=atol)
        if return_permutation:
            return entropies, perms
        return entropies


def nn_cost_matrices(
        perm_spec: PermutationSpec,
        A: Union[Dict[str, torch.Tensor], nn.Module],
        B: Union[Dict[str, torch.Tensor], nn.Module] = None,
        cost: str = "linear",
        align_obj=None,
        align_type: str = "weight",
        dataloader: Union[torch.utils.data.DataLoader, None] = None,
        normalize_m: bool = True,
        return_minimizing_permutation: bool = False,
) -> Dict[str, np.ndarray]:
    """IMPORTANT: A, B must be normalized if cost="linear" and align_type="weight"
    """
    B = A if B is None else B

    # align and get similarity matrices
    if align_obj is None:
        if align_type == "activation":
            assert isinstance(A, nn.Module) and isinstance(B, nn.Module)
            assert dataloader is not None
            align_obj = ActivationAlignment(perm_spec, dataloader, A, B, kernel=cost)
        else:
            align_obj = WeightAlignment(perm_spec, kernel=cost)

    params_A = A.state_dict() if isinstance(A, nn.Module) else A
    params_B = B.state_dict() if isinstance(B, nn.Module) else B

    # normalize, then align (this will change the alignment somewhat vs not normalizing)
    permutations, similarity_matrices = align_obj.fit(params_A, params_B)

    costs = {}
    for k in perm_spec.group_to_axes.keys():
        # similarity_matrices includes multiple iterations, we only want the last one
        # multiply by -1 to turn into cost, reorder to match perm_spec
        # increase precision to float128 as this makes Sinkhorn more stable
        costs[k] = -1 * (similarity_matrices[k][-1]).astype(np.float128)

    if normalize_m:
        non_permuted_sizes = get_non_permuted_sizes(params_A, perm_spec)
        costs = nn_normalize_costs(costs, non_permuted_sizes, cost)

    if return_minimizing_permutation:
        return costs, permutations
    return costs


def nn_normalize_costs(
        cost_matrices: Dict[str, np.ndarray],
        non_permuted_sizes: Dict[str, int],
        cost: str,
):
    _, cost_std = COST_FN_AND_STD[cost]
    normalized = {}
    for k, v in cost_matrices.items():
        normalized[k] = v / cost_std(non_permuted_sizes[k])
    return normalized


def get_non_permuted_sizes(state_dict, perm_spec):
    non_permuted_sizes = {}
    # for each permutation in perm_spec, sum up the non-permuted axes of the related parameters
    for perm_k, params in perm_spec.group_to_axes.items():
        m = 0
        for param_name, permuted_dim, is_inverse in params:
            shape = state_dict[param_name].shape
            # find size over all dimensions except the permuted one
            m += np.product(shape) // shape[permuted_dim]
        non_permuted_sizes[perm_k] = m
    return non_permuted_sizes


def apply_fn_to_dict(dictionary, apply_fn):
    return {k: apply_fn(v) for k, v in dictionary.items()}


def nn_sinkhorn(
        cost_matrices: Dict[str, np.ndarray],
        regularization=1,
        max_iter=100,
        stop_rtol=1e-4,
        stop_atol=1e-4,
        logspace=True
):
    def partial_sinkhorn_fp(c):
        return sinkhorn_fp(-regularization * c, max_iter=max_iter, stop_rtol=stop_rtol, stop_atol=stop_atol, logspace=logspace)

    return apply_fn_to_dict(cost_matrices, partial_sinkhorn_fp)


def nn_normalized_entropy(perm_matrices, rtol=3e-2, atol=3e-2):

    def partial_normalized_entropy(p):
        return normalized_entropy(p, rtol=rtol, atol=atol)

    return apply_fn_to_dict(perm_matrices, partial_normalized_entropy)


def nn_entropy_curve(cost_matrices, reg=None, min_reg=0.01, max_reg=100, n_points=40, rtol=3e-2, atol=3e-2, return_permutation=False):

    def partial_entropy_curve(c):
        return entropy_curve(c, reg=reg, min_reg=min_reg, max_reg=max_reg, n_points=n_points,
                             rtol=rtol, atol=atol, return_permutation=return_permutation)

    entropies = apply_fn_to_dict(cost_matrices, partial_entropy_curve)
    if return_permutation:  # pull out permutations into its own dict
        entropies_only = {k: v[0] for k, v in entropies.items()}
        perms = {k: v[1] for k, v in entropies.items()}
        return entropies_only, perms
    return entropies
