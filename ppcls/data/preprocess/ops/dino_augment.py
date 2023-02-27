from paddle.vision.transforms import Compose
from ppcls.data.preprocess.ops.operators import RandomHorizontalFlip, RandomGrayscale, ToTensor, Normalize, RawColorJitter
from ppcls.data.preprocess.ops.operators import RandomResizedCrop
import random
import paddle
from PIL import ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return


class RandomApply(object):
    def __init__(self, transforms: list, p=0.8):
        super(RandomApply, self).__init__()
        self.p = p
        self.transforms = transforms

    def _apply_image(self, img):
        if self.p < paddle.rand([1]):
            return img
        for t in self.transforms:
            img = t(img)
        return img


class AugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = Compose([
            RandomHorizontalFlip(prob=0.5),
            RandomApply(
                p=0.8,
                transforms=[RawColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)]
            ),
            RandomGrayscale(p=0.2),
        ])
        normalize = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = Compose([
            RandomResizedCrop(224, scale=global_crops_scale, interpolation="bicubic"),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = Compose([
            RandomResizedCrop(224, scale=global_crops_scale, interpolation="bicubic"),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = Compose([
            RandomResizedCrop(96, scale=local_crops_scale, interpolation="bicubic"),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
