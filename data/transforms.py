import random
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
