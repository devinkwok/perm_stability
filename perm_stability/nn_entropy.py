
"""
recipe for computing sinkhorn entropy for neural networks
1. weight or activation align the networks
2. get cost matrices for each permutation
3. normalize cost matrices in terms of init std and non-permuted dimension m
4. compute sinkhorn entropy for cost matrices with fixed lambda
"""
from typing import Dict, Union
import numpy as np
import torch
import torch.nn as nn

from nnperm.align.weight_align import WeightAlignment
from nnperm.align.activation_align import ActivationAlignment
from nnperm.spec.perm_spec import PermutationSpec

from perm_stability.sinkhorn_entropy import sinkhorn_fp, normalized_entropy, entropy_curve, COST_FN_AND_STD


def nn_cost_matrices(
        perm_spec: PermutationSpec,
        A: Union[Dict[str, torch.Tensor], nn.Module],
        B: Union[Dict[str, torch.Tensor], nn.Module] = None,
        align_obj=None,
        align_type: str = "weight",
        align_kernel: str = "linear",
        dataloader: Union[torch.utils.data.DataLoader, None] = None,
        normalize_m: bool = True,
        return_permutations: bool = False,
) -> Dict[str, np.ndarray]:
    """IMPORTANT: A, B must be normalized if align_type="weight" and align_kernel="linear"

    Args:
        A (Union[Dict[str, torch.Tensor], nn.Module]): first model or state dict
        B (Union[Dict[str, torch.Tensor], nn.Module], optional): second model or state dict. If None, use first model here. Defaults to None.
        perm_spec (PermutationSpec): from nnperm, specifies how to assign parameters and dims to permutations
        align_obj (Literal[&quot;weight&quot;, &quot;activation&quot;], optional): Provide custom alignment that returns fit() and predict().
        align_type (Literal[&quot;weight&quot;, &quot;activation&quot;], optional): Alignment algorithm to use. Defaults to "weight".
        align_kernel (Literal[&quot;linear&quot;, &quot;cosine&quot;], optional): Similarity metric that the permutation maximizes. Defaults to "linear".
        dataloader (Union[torch.utils.data.DataLoader, None], optional): Data for computing activations if doing activation alignment. Defaults to None.

    Returns:
        Dict[str, np.ndarray]: dictionary of permutations and their associated cost matrices.
    """
    B = A if B is None else B

    # align and get similarity matrices
    if align_obj is None:
        if align_type == "activation":
            assert isinstance(A, nn.Module) and isinstance(B, nn.Module)
            assert dataloader is not None
            align_obj = ActivationAlignment(perm_spec, dataloader, A, B, kernel=align_kernel)
        else:
            align_obj = WeightAlignment(perm_spec, align_kernel)

    params_A = A.state_dict() if isinstance(A, nn.Module) else A
    params_B = B.state_dict() if isinstance(B, nn.Module) else B
    
    # normalize, then align (this will change the alignment somewhat vs not normalizing)
    permutations, similarity_matrices = align_obj.fit(params_A, params_B)

    # similarity_matrices includes multiple iterations, we only want the last one
    # multiply by -1 to turn into cost
    costs = {k: -1 * v[-1] for k, v in similarity_matrices.items()}

    if normalize_m:
        non_permuted_sizes = get_non_permuted_sizes(params_A, perm_spec)
        costs = nn_normalize_costs(costs, non_permuted_sizes, align_kernel)

    if return_permutations:
        return costs, permutations
    return costs


def nn_normalize_costs(
        similarity_matrices: Dict[str, np.ndarray],
        non_permuted_sizes: Dict[str, int],
        align_kernel: str,
):
    _, cost_std = COST_FN_AND_STD[align_kernel]
    normalized = {}
    for k, v in similarity_matrices.items():
        normalized[k] = -1 * v / cost_std(non_permuted_sizes[k])
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


def nn_sinkhorn_ot(
        cost_matrices: Dict[str, np.ndarray],
        regularization=1,
        max_iter=100,
        rtol=1e-4,
        atol=1e-4,
        logspace=True
):
    def apply_fn(c):
        return sinkhorn_fp(-regularization * c, max_iter=max_iter, rtol=rtol, atol=atol, logspace=logspace)

    return apply_fn_to_dict(cost_matrices, apply_fn)


def nn_normalized_entropy(perm_matrices):
    return apply_fn_to_dict(perm_matrices, normalized_entropy)


def nn_entropy_curve(cost_matrices, min_l=0.01, max_l=100, n_points=40):

    def partial_entropy_curve(c):
        return entropy_curve(c, min_l=min_l, max_l=max_l, n_points=n_points)

    def apply_fn(c):
        return partial_entropy_curve(c)[1]

    lambdas, _ = partial_entropy_curve(np.ones([2, 2]))  # get the lambdas separately with dummy costs
    return lambdas, apply_fn_to_dict(cost_matrices, apply_fn)
