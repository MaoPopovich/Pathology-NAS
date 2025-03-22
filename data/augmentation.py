import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import ImageFilter, ImageOps, Image
import random


def random_horizontal_flip(img1, img2, p=0.5):
    """
    img1: Cropped image patch
    img2: Ground truth mask

    img1 and img2 should be augmentated simultaneously.
    """

    do_it = random.random() <= p
    if not do_it:
        return img1, img2
    img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
    img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
    return img1, img2


def random_vertical_flip(img1, img2, p=0.5):
    """
    img1: Cropped image patch
    img2: Ground truth mask

    img1 and img2 should be augmentated simultaneously.
    """
    do_it = random.random() <= p
    if not do_it:
        return img1, img2
    img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
    img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
    return img1, img2


def random_rotate(img1, img2):
    """
    img1: Cropped image patch
    img2: Ground truth mask

    img1 and img2 should be augmentated simultaneously.
    """
    rand = random.random()
    if rand < 0.25:
        angle = 0
    elif rand < 0.5:
        angle = 90
    elif rand < 0.75:
        angle = 180
    else:
        angle = 270
    if angle == 0:
        return img1, img2
    img1 = img1.rotate(angle)
    img2 = img2.rotate(angle)
    return img1, img2


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


class PairedRandomResizedCrop:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
        if not isinstance(size, tuple):
            size = (size, size)
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img1, img2):
        # Get crop parameters for the first image
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img1, scale=self.scale, ratio=self.ratio
        )

        # Apply the same crop to both images
        img1_cropped = TF.resized_crop(img1, i, j, h, w, self.size, Image.BILINEAR)
        img2_cropped = TF.resized_crop(img2, i, j, h, w, self.size, Image.BILINEAR)

        return img1_cropped, img2_cropped


class TranAugment:
    def __init__(self):

        self.transform1 = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=1.0),
                transforms.PILToTensor(),
            ]
        )

        self.transform2 = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
                transforms.RandomApply([Solarize()], p=0.2),
                transforms.PILToTensor(),
            ]
        )

    def __call__(self, x):
        x1 = self.transform1(x)
        x2 = self.transform2(x)
        return x1, x2


val_aug = transforms.Compose(
    [
        transforms.PILToTensor(),
    ]
)
