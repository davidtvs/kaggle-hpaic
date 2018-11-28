import os
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

    net = model.resnet(config["resnet_size"], num_classes)
    print(net)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr_rate"])
    metrics = metric.MetricList([metric.Accuracy()])

    # Location where the model checkpoints will be saved
    checkpoint_dir = os.path.join(config["checkpoint_dir"], config["name"])
    checkpoint_path = os.path.join(checkpoint_dir, config["name"] + ".pth")

    train = core.KFoldTrainer(
        net,
        config["epochs"],
        optimizer,
        criterion,
        metrics,
        checkpoint_path=checkpoint_path,
        mode="max",
        stop_patience=config["stop_patience"],
        device=config["device"],
    )
    best, _ = train.fit(dataloaders, output_fn=lambda x: x.max(dim=1, keepdim=True)[1])
