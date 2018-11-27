import os
import pandas as pd
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def get_kfold_loaders(dataset, batch_size, num_workers=4):
    """Returns a dictionary of training and validation dataloaders.

    Arguments:
        dataset (torch.utils.data.Dataset): dataset from which to load the data.
        batch_size (int): how many samples per batch to load.
        num_workers (int, optional): how many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process. Default: 4.

    Returns:
        dict: a disctionary with keys 'train' and 'val'. Each dictionary key contains a
        list of dataloaders, one for each K-fold split.
    """
    dataloaders = {"train": [], "val": []}

    # For each split get the training and validation dataloaders and append them to the
    # dictionary
    for k in range(dataset.n_splits):
        for mode, shuffle in [("train", True), ("val", False)]:
            dataset.set_mode(mode, k)

            # Pass a deepcopy of the dataset to the dataloader, otherwise all
            # dataloaders will point to the same object
            dataloader_tmp = DataLoader(
                deepcopy(dataset),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            )
            dataloaders[mode].append(dataloader_tmp)

    return dataloaders


class HPADataset(Dataset):
    # Dataset directories
    train_dir = "train"
    test_dir = "test"

    # CSV training ground-truth
    target_csv_file = "train.csv"

    # Class dictionary
    label_to_name = {
        0: "Nucleoplasm",
        1: "Nuclear membrane",
        2: "Nucleoli",
        3: "Nucleoli fibrillar center",
        4: "Nuclear speckles",
        5: "Nuclear bodies",
        6: "Endoplasmic reticulum",
        7: "Golgi apparatus",
        8: "Peroxisomes",
        9: "Endosomes",
        10: "Lysosomes",
        11: "Intermediate filaments",
        12: "Actin filaments",
        13: "Focal adhesion sites",
        14: "Microtubules",
        15: "Microtubule ends",
        16: "Cytokinetic bridge",
        17: "Mitotic spindle",
        18: "Microtubule organizing center",
        19: "Centrosome",
        20: "Lipid droplets",
        21: "Plasma membrane",
        22: "Cell junctions",
        23: "Mitochondria",
        24: "Aggresome",
        25: "Cytosol",
        26: "Cytoplasmic bodies",
        27: "Rods & rings",
    }

    def __init__(
        self,
        root_dir,
        mode,
        n_splits,
        split_index,
        transform=None,
        target_transform=None,
        data_slice=1.0,
        random_state=None,
    ):
        self.root_dir = root_dir
        self.mode = mode.lower()
        self.n_splits = n_splits
        self.split_index = split_index
        self.transform = transform
        self.target_transform = target_transform
        self.data_slice = data_slice
        self.random_state = random_state

        self.set_mode(mode, split_index)

    def set_mode(self, mode, split_index):
        self.mode = mode.lower()
        if 0 <= split_index < self.n_splits:
            self.split_index = split_index
        else:
            raise ValueError(
                "split_index out of range [0, {}], got {}".format(
                    self.n_splits - 1, split_index
                )
            )

        # Handle selected mode
        if self.mode in ("train", "val"):
            # Load training CSV
            csv_path = os.path.join(self.root_dir, self.target_csv_file)
            df = pd.read_csv(csv_path)

            # Split the data frame into two arrays: image names and target
            image_names = df["Id"].values
            targets = df["Target"].values
            targets = np.array(list(map(self._to_binary_target, targets)))

            # Split and stratify the training data
            mskf = MultilabelStratifiedKFold(
                n_splits=self.n_splits, random_state=self.random_state
            )
            kfold_splits = list(mskf.split(image_names, targets))
            train_indices = kfold_splits[split_index][0]
            val_indices = kfold_splits[split_index][1]

            # Set the appropriate data and targets
            if self.mode == "train":
                self.data_names = image_names[train_indices]
                self.targets = targets[train_indices]
            else:
                self.data_names = image_names[val_indices]
                self.targets = targets[val_indices]
        elif mode == "test":
            # Get the list of images from the test directory
            data_dir = os.path.join(self.root_dir, self.test_dir)
            self.data_names = sorted(os.listdir(data_dir))

            # The test set ground-truth is not public
            self.targets = None
        else:
            raise ValueError("invalid mode; supported modes are: train, val and test")

    def __getitem__(self, index):
        """Gets a single item from the dataset.

        Arguments:
            index (int): index of the item in the dataset

        Returns:
            dict: a dictionary with two keys, 'sample' and 'target'; 'sample' contains a
            `PIL.Image` object and 'target' a `torch.Tensor`.

        """
        raise NotImplementedError

    def __len__(self):
        """Returns the length of the data_pathset."""
        return len(self.data_names)

    def _to_binary_target(self, target_str):
        int_target = np.array(target_str.split(" "), dtype=int)
        bin_target = np.zeros(len(self.label_to_name))
        bin_target[int_target] = 1

        return bin_target


if __name__ == "__main__":
    dataset = HPADataset("../../dataset/", "train", 3, 0, random_state=92)
    dataloaders = get_kfold_loaders(dataset, 2)
    for dataloader in dataloaders["train"]:
        tmp = dataloader.dataset
        print("Fold: {}/{}".format(tmp.split_index + 1, tmp.n_splits))
        print(tmp.data_names)
        print(tmp.targets)
        print()
