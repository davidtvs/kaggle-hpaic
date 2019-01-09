import os
from copy import deepcopy
import json
import torch
import torch.nn as nn
import torch.optim as optim
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


def get_partial_sampler(mode, class_weights):
    """Creates the sampling partial function to apply to the training labels"""
    if mode is not None:
        sampler = partial(
            data.utils.frequency_weighted_sampler,
            class_weights=class_weights,
            mode=mode.lower(),
        )
    else:
        sampler = None

    return sampler


def get_weights(
    labels, scaling, min_clip, max_clip, damping_r, device, sample_weights=None
):
    """Computes class weights given the type scaling for frequency balancing."""
    if scaling is not None:
        weights = data.utils.frequency_balancing(
            labels,
            scaling=scaling.lower(),
            sample_weights=sample_weights,
            min_clip=min_clip,
            max_clip=max_clip,
            damping_r=damping_r,
        )
        weights = torch.tensor(weights, dtype=torch.float, device=device)
    else:
        weights = None

    return weights


def get_model(name, num_classes, dropout_p):
    name = name.lower()
    if name == "resnet18":
        net = model.resnet18(num_classes, dropout_p=dropout_p)
    elif name == "resnet34":
        net = model.resnet34(num_classes, dropout_p=dropout_p)
    elif name == "resnet50":
        net = model.resnet50(num_classes, dropout_p=dropout_p)
    elif name == "resnet101":
        net = model.resnet101(num_classes, dropout_p=dropout_p)
    elif name == "resnet152":
        net = model.resnet152(num_classes, dropout_p=dropout_p)
    elif name == "inceptionv3":
        net = model.inception_v3(
            pretrained=True,
            num_classes=num_classes,
            dropout_p=dropout_p,
            aux_logits=False,
        )
    elif name == "bninception":
        net = model.bninception(num_classes, dropout_p=dropout_p)
    elif name == "se_resnet50":
        net = model.se_resnet50(num_classes, dropout_p=dropout_p)
    else:
        raise ValueError("invalid model: {}".format(name))

    return net


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


def get_optimizer(model, name, lr, weight_decay):
    name = name.lower()
    if name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise ValueError("invalid optimizer: {}".format(name))

    return optimizer


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


def ensembler(array_list, weights=None):
    ensemble = np.array(array_list)
    ensemble = np.average(ensemble, axis=0, weights=weights)

    return ensemble


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
