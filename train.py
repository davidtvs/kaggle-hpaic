import os
import numpy as np
import torch
import torch.optim as optim
from argparse import ArgumentParser
import core
import data
import data.transforms as tf
import model
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


if __name__ == "__main__":
    # Get script arguments and JSON configuration
    args = arguments()
    config = utils.load_config(args.config)

    # Configs that are used multiple times
    device = torch.device(config["device"])
    random_state = config["random_state"]
    print("Device:", device)
    print("Random state:", random_state)

    # Data transformations
    image_size = (config["img_h"], config["img_w"])
    if config["aug"]:
        tf_train = tf.Augmentation(image_size)
        tf_val = tf.Resize(image_size)
    else:
        tf_train = tf.Resize(image_size)
        tf_val = tf.Resize(image_size)

    print("Image size:", image_size)
    print("Sample transform when training:", tf_train)
    print("Sample transform when validation:", tf_val)

    # Initialize the dataset
    dataset = data.HPADatasetHDF5(
        config["dataset_dir"],
        config["image_mode"],
        subset=config["subset"],
        random_state=random_state,
    )
    num_classes = len(dataset.label_to_name)
    print("No. classes:", num_classes)
    print("Training set size:", len(dataset))

    # Intiliaze the sampling strategy
    train_sampler = utils.get_sampler(config["sampler"])
    print("Training sampler instance:", train_sampler)

    # Compute class weights
    weights = utils.get_weights(config["weighing"], dataset.targets, device)
    print("Class weights:", weights)

    # Initialize the model
    net = model.resnet(
        config["resnet_size"], num_classes, dropout_p=config["dropout_p"]
    )
    print(net)

    # Select loss function
    criterion = utils.get_criterion(config["loss"], weight=weights)
    print("Criterion:", criterion)

    # K-fold training
    # Split dataset into k-sets and get one dataloader for each set
    train_loaders, val_loaders = data.utils.kfold_loaders(
        dataset,
        config["n_splits"],
        config["batch_size"],
        tf_train=tf_train,
        tf_val=tf_val,
        train_sampler=train_sampler,
        num_workers=config["workers"],
        random_state=random_state,
    )
    print("Training dataloaders:", train_loaders)
    print("Validation dataloaders:", val_loaders)

    # Optimizer
    optimizer = optim.Adam(
        net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    print("Optimizer:", optimizer)

    # Get list of metrics
    metrics = utils.get_metric_list(dataset)

    # Location where the model checkpoints will be saved
    checkpoint_dir = os.path.join(config["checkpoint_dir"], config["name"])
    print("Checkpoint directory:", checkpoint_dir)

    # Create a new KFoldTrainer instance and check if there is a checkpoint to resume
    # from
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

    scores, checkpoints = trainer.fit(
        train_loaders, val_loaders, output_fn=utils.sigmoid_threshold
    )

    # Compute the cross-validation score (average of all folds)
    avg_scores_train = np.mean(scores[0], axis=0)
    avg_scores_val = np.mean(scores[1], axis=0)
    print(
        "K-fold average training metrics: {}".format(
            np.round(avg_scores_train, 4).tolist()
        )
    )
    print(
        "K-fold average validation CV metrics: {}".format(
            np.round(avg_scores_val, 4).tolist()
        )
    )

    # Find the best threshold
    print()
    print("-" * 80)
    print("Searching for the best thresholds")
    print("-" * 80)
    for fold, (checkpoint, val_loader) in enumerate(zip(checkpoints, val_loaders)):
        print()
        print("-" * 80)
        print("Fold {}/{}".format(fold + 1, len(checkpoints)))
        print("-" * 80)
        print()
        net.load_state_dict(checkpoint["model"])
        threshold = utils.find_threshold(
            net, val_loader, metrics[0], device=device, num_thresholds=1000
        )
        print("Best threshold:\n", threshold)

        threshold = utils.find_class_threshold(
            net, val_loader, metrics[0], device=device, num_thresholds=500
        )
        print("Best thresholds per class:\n", threshold)
