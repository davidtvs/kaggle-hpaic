from torch.utils.data import Dataset
from copy import deepcopy


class Subset(Dataset):
    """Subset of a dataset at specified indices.

    Modified from: https://pytorch.org/docs/stable/data.html?highlight=subset#torch.utils.data.Subset

    Arguments:
        dataset (Dataset): the dataset
        indices (sequence): indices in the whole set selected for subset
        transform (callable, optional): transformation to apply to the dataset. If None,
            the dataset transformation is unchanged. Default: None.
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = deepcopy(dataset)
        self.indices = indices
        if transform is not None:
            self.dataset.transform = transform

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
