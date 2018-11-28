import torch
import torchvision.transforms.functional as F


def to_tensor(input, target):
    input = F.to_tensor(input)
    target = torch.tensor(target)
    return input, target


class Augmentation(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, input, target):
        input = F.resize(input, self.size)
        return input, target
