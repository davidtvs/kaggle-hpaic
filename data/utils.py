from torch.utils.data import Subset
from torch.utils.data import DataLoader
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)


def kfold_loader(subsets, batch_size, num_workers=4):
    # The dataloaders will be stored in this dictionary
    dataloaders = {"train": [], "val": []}

    for train, val in zip(*subsets.values()):
        tmp = DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        dataloaders["train"].append(tmp)
        tmp = DataLoader(
            val, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        dataloaders["val"].append(tmp)

    return dataloaders


def kfold_split(dataset, n_splits, random_state=None):
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
    # Get samples and targets arrays
    X = dataset.sample_names
    y = dataset.targets

    # The subsets will be stored in this dictionary
    subsets = {"train": [], "val": []}

    # Split and stratify the training data into k-folds
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=random_state)
    for train_indices, val_indices in mskf.split(X, y):
        subsets["train"].append(Subset(dataset, train_indices))
        subsets["val"].append(Subset(dataset, val_indices))

    return subsets


def train_val_loader(train_set, val_set, batch_size, num_workers=4):
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader


def train_val_split(dataset, val_size=0.2, random_state=None):
    # Get samples and targets arrays
    X = dataset.sample_names
    y = dataset.targets

    # Stratified split; msss.split returns a single element generator
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state
    )
    train_indices, val_indices = list(msss.split(X, y))[0]

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
