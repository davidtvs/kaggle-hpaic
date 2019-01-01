import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as tf
from argparse import ArgumentParser
from core import LRFinder
from data import HPADatasetHDF5
import data.transforms as m_tf
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
    parser.add_argument(
        "--initial-lr",
        type=float,
        default=1e-6,
        help="The minimum learning rate to test",
    )
    parser.add_argument(
        "--end-lr", type=float, default=10, help="The maximum learning rate to test"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="The number of iterations over which the test occurs",
    )
    parser.add_argument(
        "--step-mode",
        choices=["exp", "linear"],
        default="exp",
        help=(
            "The learning rate schedule: exp: the learning rate increases "
            "exponentially; linear: the learning rate increases linearly. Exponential "
            "generally yields better results while linear performs well with small "
            "ranges"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Get script arguments and JSON configuration
    args = arguments()
    config = utils.load_json(args.config)

    # Device to be used
    device = torch.device(config["device"])
    print("Device:", device)

    # Data transformations for training
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

    print("Image size:", image_size)
    print("Training data transformation:", tf_train)

    # Initialize the dataset
    dataset = HPADatasetHDF5(**config["dataset"], transform=tf_train)
    num_classes = len(dataset.label_to_name)
    print("No. classes:", num_classes)
    print("Training set size:", len(dataset))

    # Intiliaze the sampling strategy
    print("Sampler config:\n", config["sampler"])
    sampler_weights = utils.get_weights(
        dataset.targets, device=device, **config["sampler"]["weights"]
    )
    train_sampler = utils.get_partial_sampler(
        config["sampler"]["mode"], sampler_weights
    )
    if train_sampler is not None:
        train_sampler = train_sampler(dataset.targets)
    print("Sampler instance:\n", train_sampler)

    # Initialize the dataloader
    dl_cfg = config["dataloader"]
    print("Dataloader config:\n", dl_cfg)
    train_loader = DataLoader(
        dataset,
        batch_size=dl_cfg["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=dl_cfg["workers"],
    )
    print("Dataloader:", train_loader)

    # Initialize the model
    net_cfg = config["model"]
    print("Model config:\n", net_cfg)
    net = utils.get_model(net_cfg["name"], num_classes, dropout_p=net_cfg["dropout_p"])
    print(net)

    # Create the loss criterion which can be weighted or not
    if train_sampler is None:
        sample_weights = None
    else:
        # Get the sample weight from the sampler; need to unsqueeze the last dimension
        # so numpy can broadcast the array when computing the weights
        sample_weights = train_sampler.weights.unsqueeze(-1).numpy()

        # Logging purposes only
        class_w = np.mean(dataset.targets * sample_weights, axis=0)
        freq = class_w / np.sum(class_w)
        print("Sampler class frequency:\n", freq)

    print("Criterion config:\n", config["criterion"])
    weights = utils.get_weights(
        dataset.targets,
        sample_weights=sample_weights,
        device=device,
        **config["criterion"]["weights"]
    )
    criterion = utils.get_criterion(config["criterion"]["name"], weight=weights)
    print("Criterion class weights:\n", weights)
    print("Criterion:", criterion)

    # Optimizer with learning rate set to the lower limit of the learning rate range
    # to test
    print("Criterion config:\n", config["optim"])
    optimizer = utils.get_optimizer(net, **config["optim"])
    print("Optimizer:", optimizer)

    # Run the learning rate finder (fastai version)
    lr_finder = LRFinder(net, optimizer, criterion, device=device)
    lr_finder.range_test(
        train_loader,
        end_lr=args.end_lr,
        num_iter=args.iterations,
        step_mode=args.step_mode,
    )
    lr_finder.plot()
