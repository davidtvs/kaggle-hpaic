import os
import numpy as np
import torch
import torchvision.transforms as tf
from argparse import ArgumentParser
import core
import data
import data.transforms as m_tf
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
    config = utils.load_json(args.config)

    # Configs that are used multiple times
    device = torch.device(config["device"])
    random_state = config["random_state"]
    print("Device:", device)
    print("Random state:", random_state)

    # Data transformations for training and validation
    image_size = (config["img_h"], config["img_w"])
    if config["aug"]:
        # Input image augmentations
        tf_train = tf.Compose(
            [
                tf.Resize(image_size),
                tf.RandomHorizontalFlip(),
                tf.RandomVerticalFlip(),
                m_tf.Transpose(),
                tf.RandomApply([tf.RandomRotation(20)]),
                tf.RandomApply(
                    [tf.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25)]
                ),
                tf.ToTensor(),
            ]
        )
    else:
        tf_train = tf.Compose([tf.Resize(image_size), tf.ToTensor()])

    # Validation (no augmentation)
    tf_val = tf.Compose([tf.Resize(image_size), tf.ToTensor()])
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
    train_sampler = utils.get_partial_sampler(config["sampler"])
    print("Training sampler instance:", train_sampler)

    # Compute class weights
    if train_sampler is None:
        sample_weights = None
    else:
        # Get the sample weight from the sampler; need to unsqueeze the last dimension
        # so numpy can broadcast the array when computing the weights
        sample_weights = train_sampler(dataset.targets).weights.unsqueeze(-1).numpy()
        class_w = np.mean(dataset.targets * sample_weights, axis=0)
        freq = class_w / np.sum(class_w)
        print("Sampler class frequency:\n", freq)

    weights = utils.get_weights(
        config["weighing"],
        dataset.targets,
        sample_weights,
        config["min_clip"],
        config["max_clip"],
        config["damping_r"],
        device,
    )
    print("Frequency balancing mode:", config["weighing"])
    print("Minimum clip:", config["min_clip"])
    print("Maximum clip:", config["max_clip"])
    print("Damping ratio:", config["damping_r"])
    print("Class weights:\n", weights)

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
    optimizer = utils.get_optimizer(
        config["optim"], net, config["lr"], config["weight_decay"]
    )
    print("Optimizer:", optimizer)

    # Get list of metrics
    metrics = utils.get_metric_list(dataset)

    # Location where the model checkpoints will be saved
    checkpoint_dir = os.path.join(config["checkpoint_dir"], config["name"])
    print("Checkpoint directory:", checkpoint_dir)

    # Create a new KFoldTrainer instance and check if there is a checkpoint to resume
    # from
    ktrainer = core.KFoldTrainer(
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
    load_checkpoint = config["load_checkpoint"]
    if load_checkpoint and os.path.isdir(load_checkpoint):
        print("Loading from checkpoint:", load_checkpoint)
        print("Loading only weights from checkpoint?", config["weights_only"])
        ktrainer.load_checkpoint(load_checkpoint, config["weights_only"])

    scores = ktrainer.fit(train_loaders, val_loaders, output_fn=utils.sigmoid_threshold)

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
