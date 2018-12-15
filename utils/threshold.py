import torch
import numpy as np
from copy import deepcopy
import core


def sigmoid_threshold(tensor, threshold=0.5):
    """Applies the sigmoid function to the tensor and thresholds the values

    out_tensor = sigmoid(tensor) > threshold

    Arguments:
        tensor (torch.Tensor): the tensor to threshold.
        threshold (scalar or array-like): the threshold value or values. Can be a list,
            tuple, NumPy ndarray, scalar, and other types. If array-like, the size must
            match the size of `tensor`. Default: 0.5.

    Returns:
        torch.Tensor: same shape as the input with values {0, 1}.
    """
    threshold = torch.tensor(threshold, dtype=torch.float).to(tensor.device)
    out = torch.sigmoid(tensor)

    return out > threshold


def find_threshold(model, dataloader, metric, device=None, num_thresholds=100):
    """Searches for the decision threshold that yields the best metric.

    Arguments:
        model (torch.nn.Module): wrapped model.
        dataloader (torch.utils.data.DataLoader): validation set data loader.
        metric (metric.Metric): metric to monitor.
        device (str or torch.device, optional): a string ("cpu" or "cuda") with an
            optional ordinal for the device type (e.g. "cuda:X", where is the ordinal).
            Alternatively, can be an object representing the device on which the
            computation will take place. Default: None, uses the same device as `model`.
        num_thresholds (int, optional): the number of thresholds to test between 0
            and 1. Default: 100.

    Returns:
        int: the best threshold value.

    """
    # Instead of generating predictions for every threshold, we'll get the logits and
    # targets from the predict function; then the thresholds are applied to the logits
    logits, targets = core.predict(model, dataloader, device=device, ret_targets=True)

    # thresholds is a vector that contains all the thresholds to be tested.
    # best_thresholds will store the threshold that yields the highest metric.
    thresholds = np.linspace(0, 1, num_thresholds, dtype=np.float)
    best_threshold = None
    best_metric = None
    for idx, th in enumerate(thresholds):
        outputs = sigmoid_threshold(logits, threshold=th)
        metric.reset()
        metric.add(outputs, targets)
        if idx == 0 or metric.value() > best_metric:
            best_metric = metric.value()
            best_threshold = th

    return best_threshold


def find_class_threshold(model, dataloader, metric, device=None, num_thresholds=100):
    """Searches for the decision threshold that yields the best metric for each class.

    Arguments:
        model (torch.nn.Module): wrapped model.
        dataloader (torch.utils.data.DataLoader): validation set data loader.
        metric (metric.Metric): metric to monitor.
        device (str or torch.device, optional): a string ("cpu" or "cuda") with an
            optional ordinal for the device type (e.g. "cuda:X", where is the ordinal).
            Alternatively, can be an object representing the device on which the
            computation will take place. Default: None, uses the same device as `model`.
        num_thresholds (int, optional): the number of thresholds to test between 0
            and 1. Default: 100.

    Returns:
        numpy.ndarray: the best threshold value per class. Shape: (num_classes,)

    """
    # Instead of generating predictions for every threshold, we'll get the logits and
    # targets from the predict function; then the thresholds are applied to the logits
    logits, targets = core.predict(model, dataloader, device=device, ret_targets=True)
    num_classes = targets.size(1)

    # thresholds is a vector that contains all the thresholds to be tested. Best
    # thresholds is an array that stores the best threshold found for each class
    thresholds = np.linspace(0, 1, num_thresholds, dtype=np.float)
    best_thresholds = np.zeros((num_classes,))
    for class_idx in range(num_classes):
        # For each class all thresholds are tested. The threshold that yields the
        # highest metric is stored in best_thresholds
        best_metric = None
        class_thresholds = best_thresholds.copy()
        for idx, th in enumerate(thresholds):
            # th is the current threshold value for this class; class_thresholds is an
            # array that contains the threshold value for all classes
            class_thresholds[class_idx] = th
            outputs = sigmoid_threshold(logits, threshold=class_thresholds)
            metric.reset()
            metric.add(outputs, targets)
            if idx == 0 or metric.value() > best_metric:
                best_metric = metric.value()
                best_thresholds[class_idx] = th

    return best_thresholds
