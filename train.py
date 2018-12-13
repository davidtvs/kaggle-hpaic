import os
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

    # Start training
    trainer.fit(train_loaders, val_loaders, output_fn=sigmoid_threshold)
