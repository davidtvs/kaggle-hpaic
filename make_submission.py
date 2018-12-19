import os
import errno
from argparse import ArgumentParser
from functools import partial
import torch
from torch.utils.data import DataLoader
from core import predict
import data
import data.transforms as tf
import model
import utils


def arguments():
    parser = ArgumentParser(
        description="Human Protein Atlas Image Classification submission script"
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

    device = torch.device(config["device"])
    print("Device:", device)

    # Data transformations
    image_size = (config["img_h"], config["img_w"])
    transform = tf.Resize(image_size)
    print("Image size:", image_size)
    print("Sample transform:", transform)

    # Initialize the dataset
    dataset = data.HPADatasetHDF5(
        config["dataset_dir"],
        config["image_mode"],
        is_training=False,
        transform=transform,
    )
    num_classes = len(dataset.label_to_name)
    print("No. classes:", num_classes)
    print("Training set size:", len(dataset))

    # Initialize the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["workers"],
    )

    # Get list of metrics
    metrics = utils.get_metric_list(dataset)

    # Initialize the model
    net = model.resnet(
        config["resnet_size"], num_classes, dropout_p=config["dropout_p"]
    )
    print(net)

    # Load the models from the specified checkpoint location
    checkpoint_dir = os.path.join(config["checkpoint_dir"], config["name"])
    print("Checkpoint directory:", checkpoint_dir)
    knets = utils.load_kfold_models(net, checkpoint_dir)
    print("No. of models loaded from checkpoint:", len(knets))

    # Load decision thresholds from threshold json file
    json_path = os.path.join(checkpoint_dir, "threshold.json")
    threshold_dict = utils.load_json(json_path)
    print("Loaded threshold dictionary from JSON:", threshold_dict.keys())

    # Create the submission directory
    submission_dir = os.path.join(checkpoint_dir, "submission")
    try:
        os.makedirs(submission_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    print("Submissions will be saved in:", submission_dir)

    # Ready to make predictions for each fold model
    print()
    print("-" * 80)
    print("Generating predictions")
    print("-" * 80)
    for idx, net in enumerate(knets):
        print()
        print("-" * 80)
        print("Fold {}".format(idx + 1))
        print("-" * 80)
        print()

        # threshold_dict is a dictionary with two nested dictionaries. The first key
        # specifies the fold, the second sepcifies the type of threshold, and the third
        # contains the actual threshold in the "threshold" key and the validation
        # metrics evaluated with that threshold in the "metrics" key.
        fold_key = "fold_" + str(idx + 1)
        for th_key in threshold_dict[fold_key].keys():
            threshold = threshold_dict[fold_key][th_key]["threshold"]
            output_fn = partial(utils.sigmoid_threshold, threshold=threshold)
            print("Decision threshold:\n", threshold)

            # Make predictions and store them in a dictionary for later use
            predictions = predict(net, dataloader, output_fn=output_fn, device=device)
            predictions = predictions.cpu().numpy()

            # Construct the filename of the submission file using the dictionary keys
            csv_name = "{}_{}.csv".format(fold_key, th_key)
            csv_path = os.path.join(submission_dir, csv_name)
            utils.make_submission(predictions, dataset.sample_names, csv_path)
            print("Saved submission in: {}".format(csv_path))
            print()