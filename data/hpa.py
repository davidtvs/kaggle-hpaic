import os
import h5py as h5
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from .transforms import ToTensor


class HPADataset(Dataset):
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
        filters,
        is_training=True,
        transform=ToTensor(),
        subset=1.0,
        random_state=None,
    ):
        self.root_dir = root_dir
        self.is_training = is_training
        self.filters = filters.lower()
        self.transform = transform
        self.subset = subset
        self.random_state = random_state

        self.sample_names = None
        self.targets = None
        self.subset_idx = None

        # Handle selected mode
        if self.is_training:
            # Load training CSV
            csv_path = os.path.join(self.root_dir, self.target_csv_file)
            df = pd.read_csv(csv_path)

            # Split the data frame in two arrays: image_names and targets
            image_names = df["Id"].values
            targets = df["Target"].values
            targets = np.array(list(map(self._to_binary_target, targets)))

            # Create a subset of the data
            self.subset_idx = self._subset(image_names, targets)
            self.sample_names = image_names[self.subset_idx]
            self.targets = targets[self.subset_idx]
        else:
            # Get the list of images from the test directory and remove the filter from
            # the file name
            sample_names = sorted(os.listdir(self.data_dir))
            sample_names = [name.rsplit("_", 1)[0] for name in sample_names]
            sample_names = sorted(set(sample_names))

            # Convert to numpy array of objects for consistency with the training set
            self.sample_names = np.array(sample_names, dtype=np.object)

            # The test set ground-truth is not public
            self.targets = None

    def __getitem__(self, index):
        """Gets a single item from the dataset.

        Arguments:
            index (int): index of the item in the dataset

        Returns:
            dict: a dictionary with three keys:
                - ``sample``: contains the training sample;
                - ``target``: contains the ground-truth label;
                - ``sample_name``: contains the sample filename.

        """
        # Get the image and target at the specified index
        image = self._get_image(index)
        target = self.targets[index]

        # Apply the specified transformation to the image and target
        image, target = self.transform(image, target)

        return {
            "sample": image,
            "target": target,
            "sample_name": self.sample_names[index],
        }

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.sample_names)

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
            numpy.ndarray: the training set indices for the subset.

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
        return list(msss.split(X, y))[0][0]

    def _get_image(self, index):
        """Gets an image given its index.

        Arguments:
            index (int): index of the item in the dataset

        Returns:
            PIL.Image: the image with shape (H, W, C)
        """
        raise NotImplementedError

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


class HPADatasetPNG(HPADataset):
    # Dataset directories
    train_dir = "train"
    test_dir = "test"

    # Filter code to filter name
    filter_dict = {"r": "red", "g": "green", "b": "blue", "y": "yellow"}

    def __init__(
        self,
        root_dir,
        filters,
        is_training=True,
        transform=ToTensor(),
        subset=1.0,
        random_state=None,
    ):
        super().__init__(
            root_dir,
            filters,
            is_training=is_training,
            transform=transform,
            subset=subset,
            random_state=random_state,
        )
        if self.is_training:
            self.storage = os.path.join(self.root_dir, self.train_dir)
        else:
            self.storage = os.path.join(self.root_dir, self.test_dir)

    def _get_image(self, index):
        """Gets an image given its index in ``self.sample_names``.

        Arguments:
            index (int): index of the item in the dataset

        Returns:
            PIL.Image: the image with shape (H, W, C)
        """
        name = self.sample_names[index]

        # Iterate over the split list of self.filters and load the filter that matches
        # the character to a list of numpy arrays
        img_filters = []
        for f in list(self.filters):
            # list(self.filters) splits the self.filters in chars. self.filter_dict then
            # matches the char to the filter name that we'll load here
            img_name = name + "_" + self.filter_dict[f] + ".png"
            path = os.path.join(self.storage, img_name)
            img = Image.open(path)
            img_np = np.asarray(img, dtype=np.uint8)
            img_filters.append(img_np)

        # Mix the yellow filter with the red and green if self.filters is "rygb"
        if self.filters == "rygb":
            img_filters = self._mix_yellow(*img_filters)

        img_np = np.stack(img_filters, axis=-1).squeeze()

        return Image.fromarray(img_np)


class HPADatasetHDF5(HPADataset):
    # HDF5 file names
    train_hdf5 = "train.hdf5"
    test_hdf5 = "test.hdf5"

    # Filter code to filter name
    filter_dict = {"r": 0, "g": 1, "b": 2, "y": 3}

    def __init__(
        self,
        root_dir,
        filters,
        is_training=True,
        transform=ToTensor(),
        subset=1.0,
        random_state=None,
    ):
        super().__init__(
            root_dir,
            filters,
            is_training=is_training,
            transform=transform,
            subset=subset,
            random_state=random_state,
        )
        if self.is_training:
            self.storage = os.path.join(self.root_dir, self.train_hdf5)
        else:
            self.storage = os.path.join(self.root_dir, self.test_hdf5)

    def _get_image(self, index):
        """Gets an image given its index in the HDF5 file.

        Arguments:
            index (int): index of the item in the dataset

        Returns:
            PIL.Image: the image with shape (H, W, C)
        """
        true_idx = self.subset_idx[index]

        # Open the hdf5 file in read mode
        with h5.File(self.storage, "r") as f:
            # Read the name of the sample at the given index and check if it matches the
            # expected name from the internal list of sample names
            name = f["names"][true_idx].astype(str)
            expected_name = self.sample_names[index]
            if name != expected_name:
                raise ValueError(
                    "internal sample name ({}) and hdf5 sample name ({}) do not match "
                    "at index {}".format(expected_name, name, index)
                )

            # Load the sample as a numpy array which contains the 4 filters
            image_np = f["images"][true_idx]

        # Because we might not need all channels, get only the ones specified in
        # self.filters and append to img_filters
        img_filters = []
        for f in list(self.filters):
            channel = self.filter_dict[f]
            img_filters.append(image_np[:, :, channel])

        # Mix the yellow filter with the red and green if self.filters is "rygb"
        if self.filters == "rygb":
            img_filters = self._mix_yellow(*img_filters)

        img_np = np.stack(img_filters, axis=-1).squeeze()

        return Image.fromarray(img_np)
