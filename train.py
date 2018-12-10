import os
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import core
import data
import data.transforms as tf
import model
import metric
import utils


def arguments():
    parser = ArgumentParser(
        description="Human Protein Atlas Image Classification training script"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/example_kfold.json",
        help="Path to the JSON configuration file. Default: config/example_kfold.json",
    )

    return parser.parse_args()


def sigmoid_threshold(tensor, threshold=0.5, high=1, low=0):
    """Applies the sigmoid function to the tensor and thresholds the values

    out_tensor(i) = low(i) if sigmoid(tensor(i)) <= threshold(i)
                  = high(i) if sigmoid(tensor(i)) > threshold(i)

    Arguments:
        tensor (torch.Tensor): the tensor to threshold.
        threshold (scalar or array-like): the threshold value or values. Can be a list,
            tuple, NumPy ndarray, scalar, and other types. If array-like, the size must
            match the size of `tensor`. Default: 0.5.
        high (scalar or array-like): the assigned value when the sigmoid of the tensor
            is greater than `threshold`. Can be a list, tuple, NumPy ndarray, scalar,
            and other types. If array-like, the size must match the size of `tensor`.
            Default: 1.
        high (scalar or array-like): the assigned value when the sigmoid of the tensor
            is less than or equal to `threshold`. Can be a list, tuple, NumPy ndarray,
            scalar, and other types. If array-like, the size must match the size of
            `tensor`. Default: 0.

    Returns:
        torch.Tensor: same shape as the input with values {low, high}.
    """
    threshold = torch.tensor(threshold).to(tensor.device)
    high = torch.tensor(high).to(tensor.device)
    low = torch.tensor(low).to(tensor.device)

    out = torch.sigmoid(tensor)

    return torch.where(out > threshold, high, low)


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
        weights = np.ones((num_classes,))

    weights = torch.tensor(weights, dtype=torch.float, device=device)

    return weights


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


if __name__ == "__main__":
    # Get script arguments and JSON configuration
    args = arguments()
    config = utils.load_config(args.config)

    # Configs that are used multiple times
    device = torch.device(config["device"])
    random_state = config["random_state"]

    # Data transformations
    image_size = (config["img_h"], config["img_w"])
    if config["aug"]:
        tf_train = tf.Augmentation(image_size)
    else:
        tf_train = tf.Resize(image_size)

    # Initialize the dataset
    dataset = data.HPADataset(
        config["dataset_dir"],
        config["image_mode"],
        transform=tf_train,
        subset=config["subset"],
        random_state=random_state,
    )
    num_classes = len(dataset.label_to_name)

    # Intiliaze the sampling strategy
    train_sampler = get_sampler(config["sampler"])

    # Compute class weights
    weights = get_weights(config["weighing"], dataset.targets, device)
    print(weights)

    # Initialize the model
    net = model.resnet(
        config["resnet_size"], num_classes, dropout_p=config["dropout_p"]
    )
    print(net)

    # Select loss function
    loss_name = config["loss"].lower()
    if loss_name == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif loss_name == "bce_w":
        criterion = nn.BCEWithLogitsLoss(weight=weights)
    elif loss_name == "bce_pw":
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    elif loss_name == "bfl":
        criterion = model.BinaryFocalWithLogitsLoss()
    elif loss_name == "f1":
        criterion = model.FBetaWithLogitsLoss(1)
    else:
        raise ValueError("invalid loss: {}".format(config["loss"]))

    # The standard stuff is all done, now lets handle the different operation modes
    mode_name = config["mode"].lower()
    if mode_name == "find_lr":
        # Run the learning rate finder
        # Create the data sampler by passing the labels of the training set to the
        # partial function that is wrapping the sampler
        shuffle = train_sampler is None
        if train_sampler is None:
            sampler = train_sampler
        else:
            sampler = train_sampler(dataset.targets)

        # Initialize the dataloader
        train_loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=config["workers"],
        )

        # Optimizer
        optimizer = optim.Adam(
            net.parameters(),
            lr=config["initial_lr"],
            weight_decay=config["weight_decay"],
        )
        lr_finder = utils.LRFinder(net, optimizer, criterion, device=device)
        lr_finder.range_test(
            train_loader,
            end_lr=config["end_lr"],
            num_iter=config["num_iter"],
            step_mode=config["step_mode"],
        )
        lr_finder.plot()
    elif mode_name == "kfold":
        # K-fold training
        # Split dataset into k-sets and get one dataloader for each set
        train_loaders, val_loaders = data.utils.kfold_loaders(
            dataset,
            config["n_splits"],
            config["batch_size"],
            tf_train=tf_train,
            train_sampler=train_sampler,
            num_workers=config["workers"],
            random_state=random_state,
        )

        # Optimizer
        optimizer = optim.Adam(
            net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )

        # Get list of metrics
        metrics = get_metric_list(dataset)

        # Location where the model checkpoints will be saved
        checkpoint_dir = os.path.join(config["checkpoint_dir"], config["name"])

        trainer = core.KFoldTrainer(
            net,
            config["epochs"],
            optimizer,
            criterion,
            metrics,
            checkpoint_dir=checkpoint_dir,
            mode="max",
            stop_patience=config["stop_patience"],
            lr_patience=config["lr_patience"],
            lr_factor=config["lr_factor"],
            min_lr=config["min_lr"],
            device=device,
        )
        if config["resume"] and os.path.isdir(config["resume"]):
            trainer.resume(config["resume"])
        trainer.fit(train_loaders, val_loaders, output_fn=sigmoid_threshold)
    else:
        raise ValueError("invalid mode in configuration file")
