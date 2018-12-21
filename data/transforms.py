import random
import torch
import torchvision.transforms as transforms
from PIL import Image


class Transpose(object):
    """Transposes the given PIL Image randomly with a given probability.

    Arguments:
        p (float): probability of the image being flipped. Default: 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Arguments:
            img (PIL.Image): Image to be transposed.

        Returns:
            PIL.Image: Randomly transposed image.
        """
        if random.random() < self.p:
            return img.transpose(Image.TRANSPOSE)
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class ToTensor(object):
    def __call__(self, input, target):
        input = transforms.ToTensor()(input)
        target = torch.tensor(target, dtype=torch.float)

        return input, target


class Resize(object):
    def __init__(self, size):
        self.input_tf = transforms.Resize(size)

    def __call__(self, input, target):
        input = self.input_tf(input)

        return ToTensor()(input, target)


class Augmentation(object):
    def __init__(
        self, size, degrees=20, brightness=0.25, contrast=0.25, saturation=0.25
    ):
        self.input_tf = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                Transpose(),
                transforms.RandomApply([transforms.RandomRotation(degrees)]),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=brightness,
                            contrast=contrast,
                            saturation=saturation,
                        )
                    ]
                ),
            ]
        )

    def __call__(self, input, target):
        input = self.input_tf(input)

        return ToTensor()(input, target)
