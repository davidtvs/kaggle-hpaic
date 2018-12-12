import json
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import data
import model
import metric


def save_config(filepath, config):
    with open(filepath, "w") as outfile:
        json.dump(config, outfile, indent=4, sort_keys=True)


def load_config(filepath):
    with open(filepath, "r") as infile:
        config = json.load(infile)

    return config


def get_sampler(sampler_name):
    """Creates the sampling partial function to apply to the training labels"""
    if sampler_name:
        sampler_name = sampler_name.lower()
        if sampler_name == "mean":
            sampler = partial(data.utils.freq_weighted_sampler, mode="mean")
        elif sampler_name == "median":
            sampler = partial(data.utils.freq_weighted_sampler, mode="median")
        else:
            raise ValueError("invalid sampling technique: {}".format(sampler_name))
    else:
        sampler = None

    return sampler


def get_weights(weighing_name, labels, device):
    """Computes class weights given the name of the weighing technique."""
    if weighing_name:
        weighing_name = weighing_name.lower()
        if weighing_name == "fb":
            weights = data.utils.frequency_balancing(labels)
        elif weighing_name == "median_fb":
            weights = data.utils.frequency_balancing(labels, scaling="median")
        elif weighing_name == "majority_fb":
            weights = data.utils.frequency_balancing(labels, scaling="majority")
        elif weighing_name == "minority_fb":
            weights = data.utils.frequency_balancing(labels, scaling="minority")
        else:
            raise ValueError("invalid weighing: {}".format(weighing_name))
    else:
        weights = np.ones((labels.shape[-1],))

    weights = torch.tensor(weights, dtype=torch.float, device=device)

    return weights


def get_criterion(criterion_name, weight=None):
    criterion_name = criterion_name.lower()
    if criterion_name == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif criterion_name == "bce_w":
        criterion = nn.BCEWithLogitsLoss(weight=weight)
    elif criterion_name == "bce_pw":
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    elif criterion_name == "bfl":
        criterion = model.BinaryFocalWithLogitsLoss()
    elif criterion_name == "f1":
        criterion = model.FBetaWithLogitsLoss(1)
    else:
        raise ValueError("invalid loss: {}".format(criterion_name))

    return criterion


def get_metric_list(dataset):
    """Initialize metrics

    The main metric is the macro F1 score. Additionally, the F1 score for each class,
    and the overall accuracy are also computed.
    """
    metrics = [metric.FBetaScore(1, name="f1_macro")]
    for label in sorted(dataset.label_to_name):
        name = "f1_" + dataset.label_to_name[label]
        metrics.append(metric.FBetaScore(1, labels=label, name=name))

    metrics.append(metric.Accuracy())

    return metric.MetricList(metrics)
