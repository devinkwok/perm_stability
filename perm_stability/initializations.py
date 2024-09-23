from typing import Union, Tuple, Dict


def is_conv_weight(key, shape):
    return (len(shape) > 1) and key.endswith(".weight")


def is_normalization_weight(key, shape):
    return (len(shape) == 1) and key.endswith(".weight")


# conv weights are initialized Kaiming normal with variance 2 / fan_in,
# where fan_in is all non-output dimensions
def init_kaiming_normal_std_mean(param, output_dim=0):
    variance = 2 / (param.size / param.shape[output_dim])
    return variance**(1/2), 0


# normalization layer weights are initialized uniform in [0, 1]
def init_uniform_std_mean(min_range, max_range):
    mean = (max_range + min_range) / 2
    std = (max_range - min_range) / 12**(1/2)
    return std, mean


def nn_normalize_weights(state_dict, std_mean: Union[Tuple[float, float], Dict[str, Tuple[float, float]], None] = None):
    # normalize using provided stds and means
    if std_mean is not None:
        if isinstance(std_mean, tuple):
            std, mean = std_mean
            return {k: (v - mean) / std for k, v in state_dict.items()}
        return {k: (v - std_mean[k][1]) / std_mean[k][0] for k, v in state_dict.items()}

    # automatically infer stds and means from parameter keys and shapes
    normalized = {}
    for k, v in state_dict.items():
        if is_conv_weight(k, v.shape):
            std, mean = init_kaiming_normal_std_mean(v)
        elif is_normalization_weight(k, v.shape):
            std, mean = init_uniform_std_mean(0, 1)
        elif k == "fc.bias":
            # linear bias is initialized from uniform(-\sqrt{k}, \sqrt{k})
            # where k = 1 / in_features
            in_features = state_dict["fc.weight"].shape[1]
            fc_range = 1 / in_features**(1/2)
            std, mean = init_uniform_std_mean(-fc_range, fc_range)
        else:
            normalized[k] = v
            continue
        normalized[k] = (v - mean) / std
    return normalized
