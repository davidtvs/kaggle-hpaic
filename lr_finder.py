import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as tf
from argparse import ArgumentParser
from core import LRFinder
from data import HPADatasetHDF5
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

    # Configs that are used multiple times
    device = torch.device(config["device"])
    random_state = config["random_state"]
    print("Device:", device)
    print("Random state:", random_state)

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
    dataset = HPADatasetHDF5(
        config["dataset_dir"],
        config["image_mode"],
        transform=tf_train,
        subset=config["subset"],
        random_state=random_state,
    )
    num_classes = len(dataset.label_to_name)
    print("No. classes:", num_classes)
    print("Training set size:", len(dataset))

    # Intiliaze the sampling strategy
    train_sampler = utils.get_partial_sampler(config["sampler"])
    if train_sampler is not None:
        train_sampler = train_sampler(dataset.targets)
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

    # Initialize the dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config["workers"],
    )

    # Optimizer with learning rate set to the lower limit of the learning rate range
    # to test
    optimizer = optim.Adam(
        net.parameters(), lr=args.initial_lr, weight_decay=config["weight_decay"]
    )

    # Run the learning rate finder (fastai version)
    lr_finder = LRFinder(net, optimizer, criterion, device=device)
    lr_finder.range_test(
        train_loader,
        end_lr=args.end_lr,
        num_iter=args.iterations,
        step_mode=args.step_mode,
    )
    lr_finder.plot()
