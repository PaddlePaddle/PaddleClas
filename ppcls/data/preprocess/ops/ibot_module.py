import random

from paddle.vision import transforms
from paddle.vision.transforms import *
from paddle.vision.transforms import transforms as T
from PIL import ImageFilter, ImageOps

from ppcls.data.preprocess import RandomApply, RandomGrayscale


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
            return img


class IBOTAugmentation(object):
    def __init__(self, 
                 transform_ops_global,
                 transform_ops_local,
                 global_crops_number, 
                 local_crops_number):
        self.global_number = global_crops_number
        self.local_number = local_crops_number

        ts_global = self._create_ops(transform_ops_global)
        ts_global2 = ts_global.copy()
        _ = ts_global2.pop(-1)
        self.trans_global = T.Compose(ts_global)
        self.trans_global2 = T.Compose(ts_global2)

        ts_local = self._create_ops(transform_ops_local)
        self.trans_local = T.Compose(ts_local)

    def _create_ops(self, transform):
        ts = []
        for t in transform:
            for key in t.keys():
                if t[key] is not None:
                    ts.append(eval(key)(**t[key]))
                else:
                    ts.append(eval(key)())
        return ts

    def __call__(self, img):
        global_crops_imgs = [self.trans_global(img) 
                             for o in range(self.global_number - 1)]
        global_crops_imgs.insert(0, self.trans_global2(img))
        local_crops_imgs = [self.trans_local(img) 
                             for o in range(self.local_number)]

        return global_crops_imgs + local_crops_imgs