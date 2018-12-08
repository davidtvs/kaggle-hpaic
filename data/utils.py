import numpy as np
from torch.utils.data import DataLoader
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)
from .transforms import to_tensor
from .subset import Subset


def kfold_loaders(
    dataset,
    n_splits,
    batch_size,
    tf_train=to_tensor,
    tf_val=to_tensor,
    num_workers=4,
    random_state=None,
):
    """Splits the specified dataset into training and validation dataloaders for k-fold
    cross-validation.

    Each subset split from `dataset` is stratified such that they have the same label
    distribution.

    Arguments:
        dataset (torch.utils.data.Dataset): dataset to split into k-subsets of training
            and validation data for k-fold cross-validation.
        n_splits (int): number of folds.
        batch_size (int): how many samples per batch to load.
        tf_train (callable, optional): transformation to apply to the training datasets.
            Default: transforms.to_tensor.
        tf_val (callable, optional): transformation to apply to the validation datasets.
            Default: transforms.to_tensor.
        num_workers (int, optional): how many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process. Default: 4.
        random_state (int, optional): the seed used by the random number generator.
            If None, the random number generator is the RandomState instance used by
            np.random. Default: None.

    Returns:
        list: training dataloaders for each k-fold
        list: validation dataloaders for each k-fold

    """
    # Get samples and targets arrays
    X = dataset.sample_names
    y = dataset.targets

    # Split and stratify the training data into k-folds
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=random_state)

    # Iterate over the stratified folds indices, create subsets, and respective
    # dataloaders
    train = []
    val = []
    for train_indices, val_indices in mskf.split(X, y):
        # Create the training subset from the indices and the dataloader
        subset_train = Subset(dataset, train_indices, tf_train)
        loader_train = DataLoader(
            subset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        train.append(loader_train)

        # Same as above but for the validation set
        subset_val = Subset(dataset, val_indices, tf_val)
        loader_val = DataLoader(
            subset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        val.append(loader_val)

    return train, val


def train_val_loaders(
    dataset,
    val_size,
    batch_size,
    tf_train=to_tensor,
    tf_val=to_tensor,
    num_workers=4,
    random_state=None,
):
    """Splits the specified dataset into a training and validation dataloaders.

    The training and validation sets are stratified from `dataset` in such a way that
    they maintain the same label distribution.

    Arguments:
        dataset (torch.utils.data.Dataset): dataset to split into a training and
            validation.
        val_size (float): should be between 0.0 and 1.0 and represent the proportion of
            the dataset to include in the validation split.
        batch_size (int): how many samples per batch to load.
        tf_train (callable, optional): transformation to apply to the training datasets.
            Default: transforms.to_tensor.
        tf_val (callable, optional): transformation to apply to the validation datasets.
            Default: transforms.to_tensor.
        num_workers (int, optional): how many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process. Default: 4.
        random_state (int, optional): the seed used by the random number generator.
            If None, the random number generator is the RandomState instance used by
            np.random. Default: None.

    Returns:
        torch.utils.data.DataLoader: training dataloader
        torch.utils.data.DataLoader: validation dataloader

    """
    # Get samples and targets arrays
    X = dataset.sample_names
    y = dataset.targets

    # Stratified split; msss.split returns a single element generator
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state
    )
    train_indices, val_indices = list(msss.split(X, y))[0]

    # Create the training and validation subsets and dataloaders
    train_set = Subset(dataset, train_indices, tf_train)
    val_set = Subset(dataset, val_indices, tf_val)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader


def frequency_balancing(labels, scaling=None):
    """Computes class weights using the class frequency.

    Frequency balancing is described as the inverse of the class frequency:
        w = 1 / freq

    Arguments:
        labels (numpy.ndarray): array of labels with size (N, C) where N is the number
            of samples and C is the number of classes.
        scaling (string, optional): the scaling to apply to the weights:
            - `"median"`: the weights are multiplied by the median frequency;
            - `"majority"`: the weights are multiplied by the frequency of the majority
                class (maximum frequency);
            - `"minority"`: the weights are multiplied by the frequency of the minority
                class (minimum frequency);
            - `None` or `"none"`: the weights are not rescaled.

    Returns:
        numpy.ndarray: class weigths with shape (C,).

    """
    # Count the no. of positive labels for each class
    class_count = np.sum(labels, axis=0)

    # Compute the frequency and its median
    freq = class_count / labels.shape[0]

    if scaling is None or scaling.lower() == "none":
        w = 1 / freq
    elif scaling.lower() == "median":
        w = np.median(freq) / freq
    elif scaling.lower() == "majority":
        w = np.max(freq) / freq
    elif scaling.lower() == "minority":
        w = np.min(freq) / freq
    else:
        raise ValueError("invalid scaling mode: {}".format(scaling))

    return w
