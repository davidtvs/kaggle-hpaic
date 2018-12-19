import torch
import numpy as np
from core import predict


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


def find_single_threshold(
    model,
    dataloader,
    metric,
    device=None,
    min_threshold=0.2,
    max_threshold=0.8,
    num_thresholds=100,
):
    """Searches for the decision threshold that yields the best metric.

    Arguments:
        model (torch.nn.Module): wrapped model.
        dataloader (torch.utils.data.DataLoader): validation set data loader.
        metric (metric.Metric): metric to monitor.
        device (str or torch.device, optional): a string ("cpu" or "cuda") with an
            optional ordinal for the device type (e.g. "cuda:X", where is the ordinal).
            Alternatively, can be an object representing the device on which the
            computation will take place. Default: None, uses the same device as `model`.
        min_threshold (float, optional): the lowest threshold to be tested.
            Default: 0.2.
        max_threshold (float, optional): the highest threshold to be tested.
            Default: 0.8.
        num_thresholds (int, optional): the number of thresholds to test between
            ``min_threshold```and ``max_threshold``. Default: 100.

    Returns:
        float: the best threshold value.

    """
    # Instead of generating predictions for every threshold, we'll get the logits and
    # targets from the predict function; then the thresholds are applied to the logits
    logits, targets = predict(model, dataloader, device=device, ret_targets=True)

    # thresholds is a vector that contains all the thresholds to be tested.
    thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)
    best_metric = None

    # If several thresholds yield the highest metric, then they are stored in
    # highscore_thresholds and the final threshold is the median of highscore_thresholds
    highscore_thresholds = []
    for idx, th in enumerate(thresholds):
        outputs = sigmoid_threshold(logits, threshold=th)
        metric.reset()
        metric.add(outputs, targets)
        if idx == 0 or metric.value() > best_metric:
            best_metric = metric.value()
            highscore_thresholds = [th]
        elif metric.value() == best_metric:
            highscore_thresholds.append(th)

    return np.median(highscore_thresholds)


def find_class_threshold(
    model,
    dataloader,
    metric,
    device=None,
    min_threshold=0.2,
    max_threshold=0.8,
    num_thresholds=100,
):
    """Searches for the decision threshold that yields the best metric for each class.

    Arguments:
        model (torch.nn.Module): wrapped model.
        dataloader (torch.utils.data.DataLoader): validation set data loader.
        metric (metric.Metric): metric to monitor.
        device (str or torch.device, optional): a string ("cpu" or "cuda") with an
            optional ordinal for the device type (e.g. "cuda:X", where is the ordinal).
            Alternatively, can be an object representing the device on which the
            computation will take place. Default: None, uses the same device as `model`.
        min_threshold (float, optional): the lowest threshold to be tested.
            Default: 0.2.
        max_threshold (float, optional): the highest threshold to be tested.
            Default: 0.8.
        num_thresholds (int, optional): the number of thresholds to test between
            ``min_threshold```and ``max_threshold``. Default: 100.

    Returns:
        list: the best threshold value per class. Same length as the number of classes

    """
    # Instead of generating predictions for every threshold, we'll get the logits and
    # targets from the predict function; then the thresholds are applied to the logits
    logits, targets = predict(model, dataloader, device=device, ret_targets=True)
    num_classes = targets.size(1)

    # thresholds is a vector that contains all the thresholds to be tested. Best
    # thresholds is an array that stores the best threshold found for each class
    thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)
    best_thresholds = np.zeros((num_classes,))
    for class_idx in range(num_classes):
        # For each class all thresholds are tested. The threshold that yields the
        # highest metric is stored in best_thresholds.
        best_metric = None
        class_thresholds = best_thresholds.copy()

        # If several thresholds yield the highest metric, then they are stored in
        # highscore_thresholds and the final threshold that is added to best_thresholds
        # is the median of highscore_thresholds.
        highscore_thresholds = []
        for idx, th in enumerate(thresholds):
            # th is the current threshold value for this class; class_thresholds is an
            # array that contains the threshold value for all classes
            class_thresholds[class_idx] = th
            outputs = sigmoid_threshold(logits, threshold=class_thresholds)
            metric.reset()
            metric.add(outputs, targets)
            if idx == 0 or metric.value() > best_metric:
                best_metric = metric.value()
                highscore_thresholds = [th]
            elif metric.value() == best_metric:
                highscore_thresholds.append(th)

        best_thresholds[class_idx] = np.median(highscore_thresholds)

    return best_thresholds.tolist()


def multi_find_threshold(models, dataloaders, metric, device=None, num_thresholds=1000):
    """Searches for the best single and per-class decision thresholds for each model

    Wrapper function around ``find_single_threshold```and ``find_class_threshold``
    built to handle multiple models and return the single and per-class best decision
    thresholds for each pair in the array-like objects ``models`` and ``dataloaders``.

    Arguments:
        models (array-like of torch.nn.Module): an array of models.
        dataloader (array-like of torch.utils.data.DataLoader): an array of dataloaders
            for validation sets.
        metric (metric.Metric): metric to monitor.
        device (str or torch.device, optional): a string ("cpu" or "cuda") with an
            optional ordinal for the device type (e.g. "cuda:X", where is the ordinal).
            Alternatively, can be an object representing the device on which the
            computation will take place. Default: None, uses the same device as `model`.
        num_thresholds (int, optional): the number of thresholds to test between 0
            and 1. Default: 1000.

    Returns:
        Generator object that yields:
            list: the best single decision threshold value for each model. Same length
                as ``models``.
            list: the best per-class decision thresholds for each model. Same length as
                ``models``.

    """
    for model, loader in zip(models, dataloaders):
        # Single threshold
        single_threshold = find_single_threshold(
            model, loader, metric, device=device, num_thresholds=num_thresholds
        )

        # Per-class
        class_thresholds = find_class_threshold(
            model, loader, metric, device=device, num_thresholds=num_thresholds
        )

        yield single_threshold, class_thresholds
