import json
import torch
import numpy as np


def to_onehot_np(y, num_classes=None, axis=0, dtype="float32"):
    """Converts a class numpy.ndarray (integers) to a one hot numpy.ndarray.

    Modified from: https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py#L9

    Arguments:
        y (numpy.ndarray): array of integer values in the range
            [0, num_classes - 1] to be one hot encoded.
        num_classes (int, optional): total number of classes. If set to None,
            num_classes = max(y) + 1. Default: None.
        axis (int, optional): the axis where the one hot classes are encoded.
            E.g. when set to 1 and the size of y is (5, 5) the output is
            (5, num_classes, 5). Default: 0.
        dtype (torch.dtype, optional): the output data type, as a string (float32,
            float64, int32...). Default: float32.

    Returns:
        A one hot representation of the input numpy.ndarray.
    """
    y = np.array(y, dtype="int")
    if not num_classes:
        num_classes = np.max(y) + 1
    elif np.amax(y) > num_classes - 1 or np.amin(y) < 0:
        raise ValueError("y values outside range [0, {}]".format(num_classes - 1))

    input_shape = y.shape
    y = y.ravel()
    n = y.shape[0]
    output_shape = list(input_shape)
    output_shape.append(num_classes)
    axis_order = list(range(len(input_shape)))
    axis_order.insert(axis, -1)

    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    categorical = np.reshape(categorical, output_shape)

    return np.transpose(categorical, axis_order)


def to_onehot_tensor(tensor, num_classes=None, axis=0, dtype=torch.float):
    """Converts a class tensor (integers) to a one hot tensor.

    Arguments:
        tensor (torch.Tensor): tensor of integer values in the range
            [0, num_classes - 1] to be converted into a one hot tensor.
        num_classes (int, optional): total number of classes. If set to None,
            num_classes = max(tensor) + 1. Default: None.
        axis (int, optional): the axis where the one hot classes are encoded.
            E.g. when set to 1 and the size of tensor is (5, 5) the output is
            (5, num_classes, 5). Default: 0.
        dtype (torch.dtype, optional): the output data type. Default: torch.float.

    Returns:
        A one hot representation of the input tensor.
    """
    tensor = torch.tensor(tensor, dtype=torch.long)
    if not num_classes:
        num_classes = torch.max(tensor).item() + 1
    elif tensor.max() > num_classes - 1 or tensor.min() < 0:
        raise ValueError("tensor values outside range [0, {}]".format(num_classes - 1))

    out_shape = list(tensor.size())
    out_shape.insert(axis, num_classes)

    tensor = tensor.unsqueeze(axis)
    onehot = torch.zeros(out_shape, dtype=dtype)
    onehot.scatter_(axis, tensor, 1)

    return onehot


def save_config(filepath, config):
    with open(filepath, "w") as outfile:
        json.dump(config, outfile, indent=4, sort_keys=True)


def load_config(filepath):
    with open(filepath, "r") as infile:
        config = json.load(infile)

    return config
