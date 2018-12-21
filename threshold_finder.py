import os
from functools import partial
import torch
import torchvision.transforms as tf
from argparse import ArgumentParser
from core import evaluate
import data
import model
import utils


def arguments():
    parser = ArgumentParser(
        description=(
            "Human Protein Atlas Image Classification decision threshold search script"
        )
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

    # Data transformations for validation
    image_size = (config["img_h"], config["img_w"])
    tf_val = tf.Compose([tf.Resize(image_size), tf.ToTensor()])
    print("Image size:", image_size)
    print("Validation data transformation:", tf_val)

    # Initialize the dataset; no need to set the transformation because it'll be
    # overwritten when creating the dataloaders
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

    # Search for the best thresholds for each fold
    print()
    print("-" * 80)
    print("Searching for the best decision thresholds")
    print("-" * 80)
    results = {}
    th_search = utils.multi_find_threshold(
        knets, val_loaders, metrics[0], device=device
    )
    for idx, (single_th, class_th) in enumerate(th_search):
        print()
        print("-" * 80)
        print("Fold {}".format(idx + 1))
        print("-" * 80)
        print()

        # Create a new dictionary entry for each fold where the results will be stored
        # in a nested dictionary
        key = "fold_" + str(idx + 1)
        results[key] = {"default": {}, "single_best": {}, "class_best": {}}

        # Score the model using the standard decision threshold (0.5) used during
        # training
        print("Evaluating using a threshold of 0.5 for reference")
        metrics = evaluate(
            knets[idx], val_loaders[idx], metrics, output_fn=utils.sigmoid_threshold
        )
        results[key]["default"]["threshold"] = 0.5
        results[key]["default"]["metrics"] = str(metrics)
        print(metrics)
        print()

        # Display the best overall decision threshold and evaluate the model again. This
        # will show the improvement over the default threshold
        print("Best overall threshold:\n", single_th)
        output_fn = partial(utils.sigmoid_threshold, threshold=single_th)
        metrics = evaluate(knets[idx], val_loaders[idx], metrics, output_fn=output_fn)
        results[key]["single_best"]["threshold"] = single_th
        results[key]["single_best"]["metrics"] = str(metrics)
        print(metrics)
        print()

        # Same as above but now with per-class thresholds
        print("Best thresholds per class:\n", class_th)
        output_fn = partial(utils.sigmoid_threshold, threshold=class_th)
        metrics = evaluate(knets[idx], val_loaders[idx], metrics, output_fn=output_fn)
        results[key]["class_best"]["threshold"] = class_th
        results[key]["class_best"]["metrics"] = str(metrics)
        print(metrics)
        print()

    # Write the results dictionary to a json file inside checkpoint_dir
    json_path = os.path.join(checkpoint_dir, "threshold.json")
    utils.save_json(results, json_path)
