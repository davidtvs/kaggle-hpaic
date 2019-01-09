import os
import glob
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import utils


def arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        help="Path to the directory containing the CSV files to ensemble",
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

    return parser.parse_args()


def to_binary_target(target_str):
    bin_target = np.zeros(28)
    if isinstance(target_str, str):
        int_target = np.array(target_str.split(" "), dtype=int)
        bin_target[int_target] = 1

    return bin_target


if __name__ == "__main__":
    args = arguments()

    bin_target_list = []
    csv_paths = glob.glob(os.path.join(args.dir, "*.csv"))
    for csv in csv_paths:
        df = pd.read_csv(csv)
        targets = df["Predicted"].values
        targets = np.array(list(map(to_binary_target, targets)))
        bin_target_list.append(targets)

    ensemble = utils.ensembler(bin_target_list)
    predictions = (ensemble > 0.5).astype(int)
    if args.fill_empty is not None:
        ensemble = utils.fill_empty_predictions(predictions, args.fill_empty)

    # Construct the filename of the submission file; using the dictionary key
    # guarantees that the filenames are unique
    save_path = os.path.join(args.dir, "ensemble.csv")
    utils.make_submission(ensemble, df["Id"].values, save_path)
    print("Saved ensemble submission in: {}".format(save_path))
    print()
