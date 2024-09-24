import torch
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from initializations import is_conv_weight


def is_running_statistic(key, shape):
    return ((len(shape) == 1) and ".running_" in key) or \
        ((len(shape) == 0) and key.endswith(".num_batches_tracked"))


def remove_running_statistics(state_dict):
    return {k: v for k, v in state_dict.items() if not is_running_statistic(k, v.shape)}


def kaiming_normal_std(X):
    return np.sqrt(2 / np.prod(X.shape[1:]))


def load(ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    # for convenience, remove running stats as alignment won't use them
    state_dict = remove_running_statistics(state_dict)
    return {k: v.detach().numpy() for k, v in state_dict.items()}


def get_test_ckpts():
    inits = [load(f) for f in Path("./perm_stability/test/ckpts/init/").glob("*.pth")]
    trained = [load(f) for f in Path("./perm_stability/test/ckpts/trained/").glob("*.pth")]
    return {
        "init": inits,
        "trained": trained,
    }


def yield_ckpt_pairs(runs=2):
    ckpts = get_test_ckpts()
    for time1, time2 in [('init', 'init'), ('init', 'trained'), ('trained', 'trained')]:
        for pair in ['self', 'iid']:
            for i in np.arange(runs):
                run_1 = i*2
                run_2 = run_1 if pair == 'self' else run_1 + 1
                pairname = f"{time1}_{time2}-{pair}-{i}"
                yield pairname, ckpts[time1][run_1], ckpts[time2][run_2]


def yield_matrix_pairs():
    ckpts = get_test_ckpts()

    layer_keys = [k for k, v in ckpts["init"][0].items() if is_conv_weight(k, v.shape)]
    for name, ckpt_a, ckpt_b in yield_ckpt_pairs():
        for layer, k in enumerate(layer_keys):
            std = kaiming_normal_std(ckpt_a[k])
            a = ckpt_a[k] / std
            b = ckpt_b[k] / std
            if 'fc' in k:  # output of last layer can't be permuted, only input
                a, b = a.T, b.T
            pairname = f"{layer}-{k}-{name}"
            yield pairname, a, b, std


def format_entropy_curve_plot():
    plt.xlim(0.01, 100)
    plt.ylim(0, 1.1)
    plt.xscale("log")
    plt.show()
