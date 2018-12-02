import os
import torch
import torch.nn as nn
import torch.optim as optim
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
        default="config/example_train.json",
        help="Path to the JSON configuration file. Default: config/example_train.json",
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


if __name__ == "__main__":
    random_state = 92
    num_classes = 28

    # Get script arguments and JSON configuration
    args = arguments()
    config = utils.load_config(args.config)

    # Initialize the dataset and get the K-fold dataloaders
    image_size = (config["img_h"], config["img_w"])
    dataset = data.HPADataset(
        config["dataset_dir"],
        config["image_mode"],
        transform=tf.Augmentation(image_size),
        subset=config["subset"],
        random_state=92,
    )
    ksets = data.kfold_split(dataset, config["n_splits"], random_state=92)
    dataloaders = data.kfold_loader(
        ksets, config["batch_size"], num_workers=config["workers"]
    )

    # Initialize the model
    net = model.resnet(config["resnet_size"], num_classes)
    print(net)

    # Initialize loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr_rate"])

    # Initialize metrics: the main metric is the macro F1 score; additionally, we'll
    # compute the F1 score for each class, and the accuracy
    metrics = [metric.FBetaScore(name="f1_macro")]
    for label in range(num_classes):
        name = "f1_" + dataset.label_to_name[label]
        metrics.append(metric.FBetaScore(labels=label, name=name))
    metrics.append(metric.Accuracy())
    metrics = metric.MetricList(metrics)

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
        device=config["device"],
    )
    if config["resume"] and os.path.isdir(config["resume"]):
        trainer.resume(config["resume"])
    trainer.fit(dataloaders, output_fn=sigmoid_threshold)
