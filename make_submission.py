import os
import errno
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as tf
from core import predict
import data
from data.transforms import get_tta
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
    parser.add_argument(
        "--fill-empty",
        type=int,
        default=25,
        help=(
            "Samples where the model makes no prediction are changed to output the "
            "class specified in this argument. If not set, empty predictions are "
            "unchanged."
        ),
    )
    parser.add_argument(
        "--no-tta",
        dest="tta",
        action="store_false",
        help="Test time augmentation is not performed",
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        help=(
            "Number of test time augmentations to apply. If not set, all possible TTA "
            "combinations are used"
        ),
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default="0.10",
        help=(
            "How much to jitter brightness. Set to 0 to disable this transformation. "
            "Default: 0.10"
        ),
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default="0",
        help=(
            "How much to jitter contrast. Set to 0 to disable this transformation. "
            "Default: 0"
        ),
    )
    parser.add_argument(
        "--hue",
        type=float,
        default="0",
        help=(
            "How much to jitter hue. Set to 0 to disable this transformation. "
            "Default: 0"
        ),
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default="0",
        help=(
            "How much to jitter saturation. Set to 0 to disable this transformation. "
            "Default: 0"
        ),
    )
    parser.add_argument(
        "--degrees",
        type=float,
        default="0",
        help=(
            "Rotation angle in degrees. Set to 0 to disable this transformation. "
            "Default: 0"
        ),
    )
    parser.add_argument(
        "--tta-weight",
        type=float,
        default="0.6",
        help=(
            "The weight of the average TTA predictions. The weight of the regular "
            "predictions is set to (1 - value) predictions. Default: 0.6"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Get script arguments and JSON configuration
    args = arguments()
    config = utils.load_json(args.config)
    print("Config file:", args.config)
    print("Fill empty:", args.fill_empty)

    device = torch.device(config["device"])
    print("Device:", device)

    # Data transformations for testing
    image_size = (config["img_h"], config["img_w"])
    transform = tf.Compose([tf.Resize(image_size), tf.ToTensor()])
    print("Image size:", image_size)
    print("Testing data transformation:", transform)

    # Initialize the dataset
    ds_cfg = config["dataset"]
    print("Dataset configuration:\n", ds_cfg)
    dataset = data.HPADatasetHDF5(
        ds_cfg["root_dir"], ds_cfg["filters"], is_training=False, transform=transform
    )
    num_classes = len(dataset.label_to_name)
    print("No. classes:", num_classes)
    print("Test set size:", len(dataset))

    # Initialize the dataloader
    dl_cfg = config["dataloader"]
    print("Dataloader config:\n", dl_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=dl_cfg["batch_size"],
        shuffle=False,
        num_workers=dl_cfg["workers"],
    )

    # Create dataloaders for TTA
    print("Test time augmentations:", args.tta)
    if args.tta:
        tta_tf = get_tta(
            image_size,
            n_aug=args.num_aug,
            brightness=args.brightness,
            contrast=args.contrast,
            hue=args.hue,
            saturation=args.saturation,
            degrees=args.degrees,
        )
        tta_loaders = data.utils.tta_loaders(
            dataset, dl_cfg["batch_size"], tta_tf, num_workers=dl_cfg["workers"]
        )
        print("TTA transformations:\n", tta_tf)
        print("Number of TTA dataloaders:", len(tta_loaders))

        # Create the array of weights to use when ensembling with TTA
        tta_weights = np.zeros(len(tta_tf) + 1) + args.tta_weight / len(tta_tf)
        tta_weights[0] = 1 - args.tta_weight
        assert tta_weights.sum() == 1, "sum of weights is not 1: sum of {} = {}".format(
            tta_weights, tta_weights.sum()
        )
        print("TTA ensembling weights:\n", tta_weights)

    # Get list of metrics
    metrics = utils.get_metric_list(dataset)

    # Initialize the model
    net_cfg = config["model"]
    print("Model config:\n", net_cfg)
    net = utils.get_model(net_cfg["name"], num_classes, dropout_p=net_cfg["dropout_p"])
    print(net)

    # Load the models from the specified checkpoint location
    checkpoint_dir = os.path.join(config["checkpoint_dir"], config["name"])
    print("Checkpoint directory:", checkpoint_dir)
    knets = utils.load_kfold_models(net, checkpoint_dir)
    print("No. of models loaded from checkpoint:", len(knets))

    # Load decision thresholds from threshold json file
    json_path = os.path.join(checkpoint_dir, "threshold.json")
    threshold_dict = utils.load_json(json_path)
    print("Loaded threshold dictionary from JSON:", list(threshold_dict.keys()))

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
    logits_dict = {}
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
            print("Decision threshold:\n", threshold)

            # Get the logits for the test set
            print("Test set")
            logits = predict(net, dataloader, device=device).cpu().numpy()

            if args.tta:
                # Get the logits for each TTA
                logits_list = [logits]
                for idx, loader in enumerate(tta_loaders):
                    print("TTA {}/{}".format(idx + 1, len(tta_loaders)))
                    tta_logits = predict(net, loader, device=device).cpu().numpy()
                    logits_list.append(tta_logits)

                # Ensemble test and TTA logits
                logits = utils.ensembler(logits_list, weights=tta_weights)

            # Store the final logits in a dictionary to ensemble later
            if th_key in logits_dict:
                logits_dict[th_key]["logits"].append(logits.copy())
                logits_dict[th_key]["threshold"].append(threshold)
            else:
                logits_dict[th_key] = {}
                logits_dict[th_key]["logits"] = [logits.copy()]
                logits_dict[th_key]["threshold"] = [threshold]

            # Compute predictions from logits and threshold
            logits = torch.tensor(logits)
            predictions = utils.sigmoid_threshold(logits, threshold=threshold)
            predictions = predictions.cpu().numpy()

            # Construct the filename of the submission file using the dictionary keys
            csv_name = "{}_{}.csv".format(fold_key, th_key)
            csv_path = os.path.join(submission_dir, csv_name)
            utils.make_submission(predictions, dataset.sample_names, csv_path)
            print("Saved submission in:", csv_path)

            # Change empty predictions to the specified class - see arguments
            # (a wrong prediction has the same cost as not predicting anything, so might
            # aswell predict something)
            if args.fill_empty is not None:
                print("Filling empty predictions with", args.fill_empty)
                predictions = utils.fill_empty_predictions(predictions, args.fill_empty)
                csv_name = "{}_{}_fill{}.csv".format(fold_key, th_key, args.fill_empty)
                csv_path = os.path.join(submission_dir, csv_name)
                utils.make_submission(predictions, dataset.sample_names, csv_path)
                print("Saved submission in:", csv_path)

            print()

    # For each type of threshold the loop below will make an ensemble of all folds and
    # and create a submission file
    if len(knets) > 1:
        for key, value_dict in logits_dict.items():
            logits_list = value_dict["logits"]
            logits = utils.ensembler(logits_list)

            # Get the  average threshold used with the logits
            threshold_list = value_dict["threshold"]
            threshold = np.mean(threshold_list, axis=0)
            print("Decision threshold:\n", threshold)

            # Compute predictions from logits and threshold
            logits = torch.tensor(logits)
            predictions = utils.sigmoid_threshold(logits, threshold=threshold)
            predictions = predictions.cpu().numpy()

            # Construct the filename of the submission file; using the dictionary key
            # guarantees that the filenames are unique
            csv_name = "ensemble_{}.csv".format(key)
            csv_path = os.path.join(submission_dir, csv_name)
            utils.make_submission(predictions, dataset.sample_names, csv_path)
            print("Saved ensemble submission in:", csv_path)

            if args.fill_empty is not None:
                print("Filling empty predictions with", args.fill_empty)
                predictions = utils.fill_empty_predictions(predictions, args.fill_empty)
                csv_name = "ensemble_{}_fill{}.csv".format(key, args.fill_empty)
                csv_path = os.path.join(submission_dir, csv_name)
                utils.make_submission(predictions, dataset.sample_names, csv_path)
                print("Saved submission in:", csv_path)

            print()
