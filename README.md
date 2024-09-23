# Permutation Stability

`perm_stability` computes Sinkhorn-entropy: a measure of permutation stability between matrices or neural networks.

## Usage

`perm_stability` uses [nnperm](https://github.com/devinkwok/nnperm/) to align networks by weights or activations.

If not using `nnperm` for alignment, you can manually provide cost matrices to
 `nn_sinkhorn_ot`, `nn_normalized_kl`, and `nn_kl_curve` in `nn_entropy`.

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
In neural networks, call `nn_normalize_weights` on the input pair, get the cost matrix,
and call `nn_normalize_costs` on the cost matrix.
