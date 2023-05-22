import random

import paddle
from paddle.vision import BaseTransform, transforms
from PIL import ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
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
            return img


class RandomApply(BaseTransform):
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


class RandomGrayscale(BaseTransform):
    def __init__(self, prob=0.2):
        super().__init__()
        self.prob = prob

    def _apply_image(self, img):
        if paddle.rand([1]) < self.prob:
            nc = len(img.split())
            return transforms.to_grayscale(img, num_output_channels=nc)
        return img


class IBOTAugmentation(object):
    def __init__(self,
                 global_crops_scale,
                 local_crops_scale,
                 global_crops_number,
                 local_crops_number):
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(prob=0.5),
                RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                RandomGrayscale(prob=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.global_crops_number = global_crops_number
        # first global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation="bicubic"
                ),
                flip_and_color_jitter,
                GaussianBlur(1.0),
                normalize,
            ]
        )

        # second global crop
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation="bicubic"
                ),
                flip_and_color_jitter,
                GaussianBlur(0.1),
                Solarization(0.2),
                normalize,
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    96, scale=local_crops_scale, interpolation="bicubic"
                ),
                flip_and_color_jitter,
                GaussianBlur(p=0.5),
                normalize,
            ]
        )


    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
