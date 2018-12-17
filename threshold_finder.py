import os
from functools import partial
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

    # Data transformations
    image_size = (config["img_h"], config["img_w"])
    tf_val = tf.Resize(image_size)
    print("Image size:", image_size)
    print("Sample transform when validation:", tf_val)

    # Split dataset into k-sets and get one dataloader for each set. Only the validation
    # sets are needed
    _, val_loaders = data.utils.kfold_loaders(
        dataset,
        config["n_splits"],
        config["batch_size"],
        tf_val=tf_val,
        train_sampler=train_sampler,
        num_workers=config["workers"],
        random_state=random_state,
    )
    print("Validation dataloaders:", val_loaders)

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

    # Optimizer
    optimizer = optim.Adam(
        net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    print("Optimizer:", optimizer)

    # Get list of metrics
    metrics = utils.get_metric_list(dataset)

    # Location where the model checkpoints are saved
    checkpoint_dir = os.path.join(config["checkpoint_dir"], config["name"])
    print("Checkpoint directory:", checkpoint_dir)

    # Create a new KFoldTrainer instance and load the Trainer objects from the
    # checkpoint to get the best model object for each fold
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
    ktrainer.resume(checkpoint_dir)
    kmodels = [trainer.model for trainer in ktrainer.trainers]

    # Search for the best thresholds for each fold
    print()
    print("-" * 80)
    print("Searching for the best decision thresholds")
    print("-" * 80)
    th_search = utils.find_threshold(kmodels, val_loaders, metrics[0], device=device)
    for idx, (single_th, class_th) in enumerate(th_search):
        print()
        print("-" * 80)
        print("Fold {}".format(idx + 1))
        print("-" * 80)
        print()

        # Score the model using the standard decision threshold (0.5) used during
        # training
        print("Evaluating using a threshold of 0.5 for reference")
        metrics = core.evaluate(
            kmodels[idx], val_loaders[idx], metrics, output_fn=utils.sigmoid_threshold
        )
        print(metrics)
        print()

        # Display the best overall decision threshold and evaluate the model again. This
        # will show the improvement over the default threshold
        print("Best overall threshold:\n", single_th)
        output_fn = partial(utils.sigmoid_threshold, threshold=single_th)
        metrics = core.evaluate(
            kmodels[idx], val_loaders[idx], metrics, output_fn=output_fn
        )
        print(metrics)
        print()

        # Same as above but now with per-class thresholds
        print("Best thresholds per class:\n", class_th)
        output_fn = partial(utils.sigmoid_threshold, threshold=class_th)
        metrics = core.evaluate(
            kmodels[idx], val_loaders[idx], metrics, output_fn=output_fn
        )
        print(metrics)
        print()
