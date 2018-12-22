import random
from functools import partial
from itertools import combinations
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import Compose


def transpose(img):
    return img.transpose(Image.TRANSPOSE)


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
            return transpose(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


def get_tta(
    image_size,
    n_aug=32,
    brightness=0.15,
    contrast=0.15,
    hue=0.15,
    saturation=0.15,
    degrees=20,
):
    resize = partial(F.resize, size=image_size)
    transforms = [F.hflip, F.vflip, transpose]
    if brightness != 0:
        partial_f = partial(F.adjust_brightness, brightness_factor=1 + brightness)
        transforms.append(partial_f)
    if contrast != 0:
        partial_f = partial(F.adjust_contrast, contrast_factor=1 + contrast)
        transforms.append(partial_f)
    if hue != 0:
        partial_f = partial(F.adjust_hue, hue_factor=1 + hue)
        transforms.append(partial_f)
    if saturation != 0:
        partial_f = partial(F.adjust_saturation, saturation_factor=1 + saturation)
        transforms.append(partial_f)
    if degrees != 0:
        partial_f = partial(F.rotate, angle=degrees)
        transforms.append(partial_f)

    tta = []
    for length in range(1, len(transforms) + 1):
        for aug in combinations(transforms, length):
            aug = (resize,) + aug + (F.to_tensor,)
            tta.append(Compose(aug))

    if n_aug == len(tta):
        return tta
    elif n_aug < len(tta):
        return random.sample(tta, n_aug)
    else:
        raise ValueError(
            "generated {} augmentations but {} were requested".format(len(tta), n_aug)
        )
