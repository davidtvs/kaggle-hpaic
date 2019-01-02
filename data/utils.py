from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)
from torchvision.transforms import ToTensor
from .subset import Subset


def kfold_loaders(
    dataset,
    n_splits,
    batch_size,
    tf_train=ToTensor(),
    tf_val=ToTensor(),
    train_sampler=None,
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
            Default: transforms.ToTensor.
        tf_val (callable, optional): transformation to apply to the validation datasets.
            Default: transforms.ToTensor.
        train_sampler (functools.partial, optional): partial function object that takes
            the dataset targets as the only remaining argument. The function defines the
            strategy to draw samples from the dataset. If specified, ``shuffle`` must be
            False. Default: None.
        num_workers (int, optional): how many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process. Default: 4.
        random_state (int, optional): the seed used by the random number generator.
            If None, the random number generator is the RandomState instance used by
            np.random. Default: None.

    Returns:
        list: training dataloaders for each k-fold
        list: validation dataloaders for each k-fold

    """
    train_shuffle = train_sampler is None
    val_shuffle = False

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
        # Create the training subset from the indices
        subset_train = Subset(dataset, train_indices, tf_train)

        # Create the data sampler by passing the labels of the training sub set to the
        # partial function object
        if train_sampler is None:
            subsampler = train_sampler
        else:
            subsampler = train_sampler(dataset.targets[train_indices])

        # Initialize the training dataloader
        loader_train = DataLoader(
            subset_train,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=subsampler,
            drop_last=True,
            num_workers=num_workers,
        )
        train.append(loader_train)

        # Similar to the above, but the validation set doesn't use any sampling
        # technique
        subset_val = Subset(dataset, val_indices, tf_val)
        loader_val = DataLoader(
            subset_val,
            batch_size=batch_size,
            shuffle=val_shuffle,
            num_workers=num_workers,
        )
        val.append(loader_val)

    return train, val


def train_val_loaders(
    dataset,
    val_size,
    batch_size,
    tf_train=ToTensor(),
    tf_val=ToTensor(),
    train_sampler=None,
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
            Default: transforms.ToTensor.
        tf_val (callable, optional): transformation to apply to the validation datasets.
            Default: transforms.ToTensor.
        train_sampler (functools.partial, optional): partial function object that takes
            the dataset targets as the only remaining argument. The function defines the
            strategy to draw samples from the dataset. If specified, ``shuffle`` must be
            False. Default: None.
        num_workers (int, optional): how many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process. Default: 4.
        random_state (int, optional): the seed used by the random number generator.
            If None, the random number generator is the RandomState instance used by
            np.random. Default: None.

    Returns:
        torch.utils.data.DataLoader: training dataloader
        torch.utils.data.DataLoader: validation dataloader

    """
    train_shuffle = train_sampler is None
    val_shuffle = False

    # Get samples and targets arrays
    X = dataset.sample_names
    y = dataset.targets

    # Stratified split; msss.split returns a single element generator
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state
    )
    train_indices, val_indices = list(msss.split(X, y))[0]

    # Create the training subset from the stratified indices
    train_set = Subset(dataset, train_indices, tf_train)

    # Create the data sampler by passing the labels of the training sub set to the
    # partial function object
    if train_sampler is None:
        sampler = train_sampler
    else:
        sampler = train_sampler(dataset.targets[train_indices])

    # Initialize the training dataloader
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
    )

    # Similar to the above, but the validation set doesn't use any sampling
    # technique
    val_set = Subset(dataset, val_indices, tf_val)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=val_shuffle, num_workers=num_workers
    )

    return train_loader, val_loader


def tta_loaders(dataset, batch_size, tta_transforms, num_workers=4):
    loaders = []
    for tta in tta_transforms:
        tta_set = deepcopy(dataset)
        tta_set.transform = tta
        loader = DataLoader(tta_set, batch_size=batch_size, num_workers=num_workers)
        loaders.append(loader)

    return loaders


def frequency_balancing(
    labels, sample_weights=None, scaling=None, min_clip=None, max_clip=None, damping_r=1
):
    """Computes class weights using the class frequency.

    Frequency balancing is described as the inverse of the class frequency:
        w = 1 / freq

    Arguments:
        labels (numpy.ndarray): array of labels with size (N, C) where N is the number
            of samples and C is the number of classes.
        sample_weights (numpy.ndarray, optional): array of weights with shape (N, 1).
            When set to None, it's assumed that all sampled have the same weight.
            Default: None
        scaling (string, optional): the scaling to apply to the weights:
            - ``"median"``: the weights are multiplied by the median frequency;
            - ``"majority"``: the weights are multiplied by the frequency of the
                majority class (maximum frequency);
            - ``"minority"``: the weights are multiplied by the frequency of the
                minority class (minimum frequency);
            - ``"log"``: log-damped class frequency w = ln(damping_r / freq)
            - ``None`` or ``"none"``: the weights are not rescaled.
        min_clip (float, optional): minimum value. If None, clipping is not performed on
            lower interval edge. Default: None.
        max_clip (float, optional): maximum value. If None, clipping is not performed on
            upper interval edge. Default: None.
        damping_r (float, optional): damping ratio used when ``scaling = "log"``.
            Ignored for all other scaling modes. Default: 1.

    Returns:
        numpy.ndarray: class weigths with shape (C,).

    """
    if sample_weights is None:
        # Compute the class frequency
        class_count = np.sum(labels, axis=0)
        freq = class_count / np.sum(class_count)
    else:
        if labels.shape[0] != sample_weights.shape[0]:
            raise ValueError("number of samples mismatch")
        # When using sample_weights each label has a different probability of being used
        # to compute the class frequency we can multiply the sample weight by the binary
        # label which results in a matrix of weights per sample and per class. The mean
        # of that matrix over the samples gives us the overall weight of each class. The
        # class frequency is just the normalized of the latter
        sample_class_w = labels * sample_weights
        class_w = np.mean(sample_class_w, axis=0)
        freq = class_w / np.sum(class_w)

    if scaling is None or scaling.lower() == "none":
        w = 1 / freq
    elif scaling.lower() == "median":
        w = np.median(freq) / freq
    elif scaling.lower() == "majority":
        w = np.max(freq) / freq
    elif scaling.lower() == "minority":
        w = np.min(freq) / freq
    elif scaling.lower() == "log":
        w = np.log(damping_r / freq)
    else:
        raise ValueError("invalid scaling mode: {}".format(scaling))

    if min_clip is not None or max_clip is not None:
        w = np.clip(w, min_clip, max_clip)

    return w


def frequency_weighted_sampler(labels, class_weights, mode="mean"):
    samples_weight = labels * class_weights

    # To apply the specified operation we want to ignore the 0s; the simplest way of
    # achieving this goal is to set all 0s to NaN and use the operations that ignore NaN
    samples_weight[samples_weight == 0] = np.nan
    mode = mode.lower()
    if mode == "median":
        samples_weight = np.nanmedian(samples_weight, axis=1)
    elif mode == "mean":
        samples_weight = np.nanmean(samples_weight, axis=1)
    elif mode == "max":
        samples_weight = np.nanmax(samples_weight, axis=1)
    elif mode == "meanmax":
        sw_mean = np.nanmean(samples_weight, axis=1, keepdims=True)
        sw_max = np.nanmax(samples_weight, axis=1, keepdims=True)
        sw_meanmax = np.hstack((sw_mean, sw_max))
        samples_weight = np.mean(sw_meanmax, axis=1)
    else:
        raise ValueError("invalid mode: {}".format(mode))

    # Convert to torch tensor
    samples_weight = torch.from_numpy(samples_weight)

    return WeightedRandomSampler(samples_weight, len(samples_weight))
