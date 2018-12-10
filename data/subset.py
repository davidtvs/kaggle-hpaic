from torch.utils.data import Dataset
from copy import deepcopy
from .transforms import ToTensor


class Subset(Dataset):
    """Subset of a dataset at specified indices.

    Modified from: https://pytorch.org/docs/stable/data.html?highlight=subset#torch.utils.data.Subset

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        transform (callable, optional): transformation to apply to the data samples and
            targets. Default: transforms.ToTensor.
    """

    def __init__(self, dataset, indices, transform=ToTensor()):
        self.dataset = deepcopy(dataset)
        self.indices = indices
        self.dataset.transform = transform

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
