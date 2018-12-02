import torch
import torchvision.transforms as transform


def to_tensor(input, target):
    input = transform.ToTensor()(input)
    target = torch.tensor(target, dtype=torch.float)
    return input, target


class Augmentation(object):
    def __init__(
        self, size, degrees=20, brightness=0.25, contrast=0.25, saturation=0.25
    ):
        self.image_aug = transform.Compose(
            [
                transform.Resize(size),
                transform.RandomHorizontalFlip(),
                transform.RandomVerticalFlip(),
                transform.RandomRotation(degrees),
                transform.ColorJitter(
                    brightness=brightness, contrast=contrast, saturation=saturation
                ),
            ]
        )

    def __call__(self, input, target):
        input = self.image_aug(input)
        return to_tensor(input, target)
