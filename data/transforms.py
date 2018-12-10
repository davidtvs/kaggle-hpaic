import torch
import torchvision.transforms as transforms


class ToTensor(object):
    def __call__(self, input, target):
        input = transforms.ToTensor()(input)
        target = torch.tensor(target, dtype=torch.float)

        return input, target


class Augmentation(object):
    def __init__(
        self, size, degrees=20, brightness=0.25, contrast=0.25, saturation=0.25
    ):
        self.image_aug = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees),
                transforms.ColorJitter(
                    brightness=brightness, contrast=contrast, saturation=saturation
                ),
            ]
        )

    def __call__(self, input, target):
        input = self.image_aug(input)

        return ToTensor()(input, target)
