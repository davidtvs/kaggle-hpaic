import os
import pandas as pd
import numpy as np
from copy import deepcopy
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .transforms import to_tensor
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)


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

    # If the dataset random_state is not set, set it now to guarantee that all folds are
    # generated in the same manner with the same data.
    if dataset.random_state is None:
        dataset.random_state = np.random.randint(2 ** 32 - 1)

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

    # Filter code to filter name
    filter_dict = {"r": "red", "g": "green", "b": "blue", "y": "yellow"}

    # Image extension
    ext = ".png"

    def __init__(
        self,
        root_dir,
        mode,
        filters,
        n_splits,
        split_index=0,
        transform=to_tensor,
        subset=1.0,
        random_state=None,
    ):
        self.root_dir = root_dir
        self.mode = mode.lower()
        self.filters = filters.lower()
        self.n_splits = n_splits
        self.split_index = split_index
        self.transform = transform
        self.subset = subset
        self.random_state = random_state

        self.data_names = None
        self.data_dir = None
        self.targets = None

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

            # Create a subset of the data
            image_names, targets = self._subset(image_names, targets)

            # Split and stratify the training data
            mskf = MultilabelStratifiedKFold(
                n_splits=self.n_splits, random_state=self.random_state
            )
            kfold_splits = list(mskf.split(image_names, targets))
            train_indices = kfold_splits[split_index][0]
            val_indices = kfold_splits[split_index][1]

            # Set the appropriate data and targets
            self.data_dir = os.path.join(self.root_dir, self.train_dir)
            if self.mode == "train":
                self.data_names = image_names[train_indices]
                self.targets = targets[train_indices]
            else:
                self.data_names = image_names[val_indices]
                self.targets = targets[val_indices]
        elif mode == "test":
            # Get the list of images from the test directory
            self.data_dir = os.path.join(self.root_dir, self.test_dir)
            self.data_names = sorted(os.listdir(self.data_dir))

            # The test set ground-truth is not public
            self.targets = None
        else:
            raise ValueError("invalid mode; supported modes are: train, val and test")

    def __getitem__(self, index):
        """Gets a single item from the dataset.

        Arguments:
            index (int): index of the item in the dataset

        Returns:
            dict: a dictionary with three keys, 'sample', 'target', and 'sample_name';
            'sample' contains the training sample, 'target' contains the ground-truth
            label, and 'sample_name' contains the sample filename.

        """
        data_name = self.data_names[index]
        image = self._get_image(data_name)
        target = self.targets[index]

        image, target = self.transform(image, target)

        return {"sample": image, "target": target, "sample_name": data_name}

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data_names)

    def _to_binary_target(self, target_str):
        """Converts a label from the CSV file to its binary representation.

        Arguments:
            target_str (string): the label from the CSV file.

        Returns:
            numpy.ndarray: the binary representation of `target_str`.
        """
        int_target = np.array(target_str.split(" "), dtype=int)
        bin_target = np.zeros(len(self.label_to_name))
        bin_target[int_target] = 1

        return bin_target

    def _subset(self, X, y):
        """Create a subset of the full dataset.

        Arguments:
            X (array-like): samples with shape (n_samples, n_features) where n_samples
                is the number of samples and n_features is the number of features.
            y (array-like): the target variable for supervised learning problems.
                Multilabel stratification is done based on the y labels.
                Shape: (n_samples, n_labels).

        Returns:
            (numpy.ndarray, numpy.ndarray): the subset of X and y.
        """
        if self.subset == 1:
            return X, y

        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            train_size=self.subset,
            test_size=None,
            random_state=self.random_state,
        )

        # Index 0 gives a tuple of the train and test indices. The seconand index 0
        # returns the train indices which corresponds to self.subset of the initial data
        subset_idx = list(msss.split(X, y))[0][0]
        return X[subset_idx], y[subset_idx]

    def _get_image(self, name):
        """Gets an image given its name.

        Arguments:
            name (string): the image filename.

        Returns:
            PIL.Image: the image with shape (H, W, C)
        """
        # Split the self.filters string character by character
        filters = list(self.filters)

        # Iterate over the split list of self.filters and load the filter that matches
        # the character to a list of numpy arrays
        img_filters = []
        for f in filters:
            img_name = name + "_" + self.filter_dict[f] + self.ext
            path = os.path.join(self.data_dir, img_name)
            img = Image.open(path)
            img_np = np.asarray(img, dtype=np.uint8)
            img_filters.append(img_np)

        # Mix the yellow filter with the red and green if self.filters is "rygb"
        if self.filters == "rygb":
            img_filters = self._mix_yellow(*img_filters)

        img_np = np.stack(img_filters, axis=-1).squeeze()

        return Image.fromarray(img_np)

    def _mix_yellow(self, r, y, g, b):
        """Mixes the yellow channel with the red and green channels.

        Arguments:
            r (numpy.ndarray): the red channel.
            y (numpy.ndarray): the yellow channel.
            g (numpy.ndarray): the green channel.
            b (numpy.ndarray): the blue channel.

        Returns:
            list: [r, g, b] where r, g, and b are numpy.ndarray.
        """
        # Set the red and green channels with the yellow filter values. Must be a copy
        # else it'll bind a new name to y
        r_mix = np.copy(y)
        g_mix = np.copy(y)

        # Override the red and green channels with the corresponding filters
        r_mix[r > 0] = r[r > 0]
        g_mix[g > 0] = g[g > 0]

        return [r_mix, g_mix, b]
