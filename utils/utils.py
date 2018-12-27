import os
from copy import deepcopy
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from functools import partial
import data
import model
import metric


def save_json(data, filepath, indent=4, sort_keys=False):
    with open(filepath, "w") as outfile:
        json.dump(data, outfile, indent=indent, sort_keys=sort_keys)


def load_json(filepath):
    with open(filepath, "r") as infile:
        data = json.load(infile)

    return data


def get_partial_sampler(sampler_name):
    """Creates the sampling partial function to apply to the training labels"""
    if sampler_name:
        sampler_name = sampler_name.lower()
        if sampler_name == "mean":
            sampler = partial(data.utils.frequency_weighted_sampler, mode="mean")
        elif sampler_name == "median":
            sampler = partial(data.utils.frequency_weighted_sampler, mode="median")
        else:
            raise ValueError("invalid sampling technique: {}".format(sampler_name))
    else:
        sampler = None

    return sampler


def get_weights(scaling, labels, sample_weights, min_clip, max_clip, damping_r, device):
    """Computes class weights given the type scaling for frequency balancing."""
    if scaling:
        scaling = scaling.lower()
        weights = data.utils.frequency_balancing(
            labels,
            scaling=scaling,
            sample_weights=sample_weights,
            min_clip=min_clip,
            max_clip=max_clip,
            damping_r=damping_r,
        )
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


def load_kfold_models(net, checkpoint_dir):
    # Load the model weights from the checkpoint. It's asssumed that the directory
    # contains one subdirectory per fold named fold_x, where x is the fold number
    knets = []
    fold_idx = 1
    fold_checkpoint = os.path.join(checkpoint_dir, "fold_" + str(fold_idx))
    while os.path.isdir(fold_checkpoint):
        # Each fold subdirectory contains the checkpoint in a file named "model.pth"
        model_path = os.path.join(fold_checkpoint, "model.pth")
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        net.load_state_dict(checkpoint["model"])
        knets.append(deepcopy(net))

        # Update fold index and expected checkpoint directory
        fold_idx += 1
        fold_checkpoint = os.path.join(checkpoint_dir, "fold_" + str(fold_idx))

    # If "fold_1" doesn't exist then no valid model checkpoints were found
    if fold_idx == 1:
        raise FileNotFoundError("fold checkpoint not found")

    return knets


def make_submission(bin_predictions, sample_ids, path):
    # Iterate over each output in the batch and convert from binary to text
    predictions = []
    for idx, bin_pred in enumerate(bin_predictions):
        # Need to convert from binary format to integer format
        # The number of classes is bin_pred.shape[-1]
        bin_pred = bin_pred.astype(bool)
        int_output = np.arange(bin_pred.shape[-1])[bin_pred]

        # Then, join the integer labels seperated by an empty space
        str_output = [str(x) for x in int_output.tolist()]
        text_output = " ".join(str_output)
        predictions.append(text_output)

    # Build the submission data frame and save it as a csv file
    df = pd.DataFrame({"Id": sample_ids, "Predicted": predictions})
    df.to_csv(path, index=False)


def tta_ensembler(predictions, tta_predictions, tta_weight=0.6, threshold=0.5):
    tta_predictions = np.array(tta_predictions)
    tta_predictions = np.mean(tta_predictions, axis=0)
    weighted_pred = (1 - tta_weight) * predictions + tta_weight * tta_predictions

    return (weighted_pred > threshold).astype(int)


def ensembler(predictions, threshold=0.5):
    ensemble = np.array(predictions)
    ensemble = np.mean(ensemble, axis=0)

    return (ensemble > threshold).astype(int)


def fill_empty_predictions(predictions, class_idx):
    """Change empty predictions to output the specified class

    Arguments:
        predictions (numpy.ndarray): the model predictions in binary format.
            Shape: (N, K), where N is the number of samples and K is the number of
            classes.
        class_idx (int): the class to output when the model makes no
            prediction. Expects an integer in the range [0, K-1].

    Returns:
        numpy.ndarray: the predictions
    """
    if not 0 <= class_idx < predictions.shape[1]:
        raise ValueError(
            "class_idx is not in the expected range: [{}, {}], got {}".format(
                0, predictions.shape[1], class_idx
            )
        )
    empty_mask = np.sum(predictions, axis=1) == 0
    predictions[empty_mask, class_idx] = 1

    return predictions
