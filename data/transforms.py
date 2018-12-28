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
    image_size, n_aug=None, brightness=0.10, contrast=0, hue=0, saturation=0, degrees=0
):
    resize = partial(F.resize, size=image_size)
    transforms = [F.hflip, F.vflip, transpose]

    # Some transformations cannot be applied together or they will cancel each other;
    # This group of transformations are added to the list as tuples and only one of the
    # tuple elements is chosen when composing the final list transformations
    if brightness != 0:
        partial_f_n = partial(F.adjust_brightness, brightness_factor=1 - brightness)
        partial_f_p = partial(F.adjust_brightness, brightness_factor=1 + brightness)
        transforms.append((partial_f_n, partial_f_p))
    if contrast != 0:
        partial_f_n = partial(F.adjust_contrast, contrast_factor=1 - contrast)
        partial_f_p = partial(F.adjust_contrast, contrast_factor=1 + contrast)
        transforms.append((partial_f_n, partial_f_p))
    if hue != 0:
        partial_f_n = partial(F.adjust_hue, hue_factor=1 - hue)
        partial_f_p = partial(F.adjust_hue, hue_factor=1 + hue)
        transforms.append((partial_f_n, partial_f_p))
    if saturation != 0:
        partial_f_n = partial(F.adjust_saturation, saturation_factor=1 - saturation)
        partial_f_p = partial(F.adjust_saturation, saturation_factor=1 + saturation)
        transforms.append((partial_f_n, partial_f_p))
    if degrees != 0:
        partial_f_n = partial(F.rotate, angle=-degrees)
        partial_f_p = partial(F.rotate, angle=degrees)
        transforms.append((partial_f_n, partial_f_p))

    tta = []
    for length in range(1, len(transforms) + 1):
        for aug in combinations(transforms, length):
            # For each tuple of transformations in aug, select one of the tuple elements
            tmp = []
            for tf in aug:
                if isinstance(tf, tuple):
                    tmp.append(random.choice(tf))
                else:
                    tmp.append(tf)
            aug = tmp

            # Insert the image resize and append the transformation to tensor
            aug.insert(0, resize)
            aug.append(F.to_tensor)

            # Wrap this augmentation in Compose and add to the list of TTAs
            tta.append(Compose(aug))

    if n_aug is None or n_aug == len(tta):
        return tta
    elif n_aug < len(tta):
        return tta[:n_aug]
    else:
        raise ValueError(
            "generated {} augmentations but {} were requested".format(len(tta), n_aug)
        )
