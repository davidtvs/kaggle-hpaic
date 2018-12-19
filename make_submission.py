import os
from argparse import ArgumentParser
from functools import partial
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from core import predict_yield_batch
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
        default="config/example_submission.json",
        help="Path to the JSON configuration file. Default: config/example_submission.json",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Get script arguments and JSON configuration
    args = arguments()
    config = utils.load_config(args.config)

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

    # Load model weights from checkpoint
    print("Loading model weights from {}...".format(config["model_weights"]))
    checkpoint = torch.load(config["model_weights"], map_location=torch.device("cpu"))
    net.load_state_dict(checkpoint["model"])

    # Load decision thresholds from configuration file
    output_fn = partial(utils.sigmoid_threshold, threshold=config["threshold"])
    print("Decision threshold:", config["threshold"])
    print("Output function:", output_fn)

    print()
    print("-" * 80)
    print("Generating predictions")
    print("-" * 80)
    predictions = []
    predictor = predict_yield_batch(net, dataloader, output_fn=output_fn, device=device)
    for bin_output_batch in predictor:
        bin_output_batch = bin_output_batch.cpu().numpy().astype(bool)

        # Iterate over each output in the batch and convert from binary to text
        for idx, bin_output in enumerate(bin_output_batch):
            # Need to convert from binary format to integer format
            int_output = np.arange(num_classes)[bin_output]

            # Then, join the integer labels seperated by an empty space
            str_output = [str(x) for x in int_output.tolist()]
            text_output = " ".join(str_output)
            predictions.append(text_output)

    # Construct the path where the submission will be saved:
    # - same directory as the model weights;
    # - filename as specified in submission_name setting in the config file.
    csv_dir = os.path.dirname(config["model_weights"])
    csv_filename = config["name"] + ".csv"
    csv_path = os.path.join(csv_dir, csv_filename)

    # Build the submission data frame and save it as a csv file
    df = pd.DataFrame({"Id": dataset.sample_names, "Predicted": predictions})
    df.to_csv(csv_path, index=False)
    print()
    print("Done! Saved submission in: {}".format(csv_path))
