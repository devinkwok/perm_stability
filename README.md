# Permutation Stability

A library for computing Sinkhorn entropy: a measure of permutation stability between matrices or neural networks.


## Sinkhorn Entropy

The recipe for computing Sinkhorn entropy for neural networks is:

1. Normalize weights based on their mean and std *at initialization*. I am matching mine to open_lth so e.g. normalization weights are uniform in (0, 1), and fully connected bias is based on the uniform init described here: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
2. Align the networks.
3. Get the cost matrices for the alignment.
4. Normalize the cost matrices in terms of the sum over non-permuted dimensions, again assuming that the costs are computed from 0 mean and 1 std IID random variables. For L2 cost, this means dividing by sqrt(m), where m is all non-permuted dimensions (e.g. if there is a 16x8x3x3 conv weight, 16x1 bias, and 32x16x3x3 next conv layer permuted over the 16 dim channel, then m = 8x8x3+1+32x3x3).
5. Compute Sinkhorn entropy for each of the cost matrices, using a single lambda or a range of lambda regularization parameters.


## Usage

`sinkhorn_entropy` contains functions for computing Sinkhorn entropies for individual matrices. For an all-in-one solution, use the function `sinkhorn_entropy.sinkhorn_entropies`.

`nn_entropy` contains functions for computing Sinkhorn entropies for neural networks. For an all-in-one solution, use the function `nn_entropy.nn_sinkhorn_entropies`.

To align networks, `nn_entropy` uses [nnperm](https://github.com/devinkwok/nnperm/).
If you do not want to use `nnperm` for alignment, you have 2 options:

1. Provide an `align_obj` to `nn_sinkhorn_entropies` modelled after `WeightAlignment` or `ActivationAlignment`.
The `align_obj` should implement the scikit-like method `fit(A, B) -> permutations, similarity_matrices`.
2. Manually provide cost matrices to `nn_sinkhorn`, and call `nn_normalized_entropy` or `nn_entropy_curve`.

Run tests with `python -m unittest discover perm_stability`.

## Cost magnitude

All comparable cost matrices should share a fixed magnitude.
Extremely large or small cost matrices can cause numerical instability in the Sinkhorn algorithm.

By default, `perm_stability` chooses a rescaling such that for matrices or networks
where *all layers are initialized to be IID random*,
the cost matrices always have mean 0 and standard deviation 1.
Note this may not reflect real initializations where some parameters are uniformly zero, etc,
however this does give a consistent standard on which the Sinkhorn algorithm is also stable.

To match this functionality in matrices, call `sinkhorn_entropy.normalize_weight`
on the input pair and `sinkhorn_entropy.normalized_cost` to get the cost matrix.
In neural networks, call `initializations.auto_normalize_weights` on the input pair, get the cost matrix,
and call `nn_entropy.nn_normalize_costs` on the cost matrix.

## Known issues

Cosine cost is not correctly normalized in `nn_entropy_curve`.
