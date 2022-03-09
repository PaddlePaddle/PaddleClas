from functools import partial
import six
import math
import random
import cv2
import numpy as np
import importlib
from PIL import Image

from utils import logger


class PreProcesser(object):
    def __init__(self, config):
        """Image PreProcesser

        Args:
            config (list): A list consisting of Dict object that describe an image processer operator.
        """
        super().__init__()
        self.ops = self.create_ops(config)

    def create_ops(self, config):
        if not isinstance(config, list):
            msg = "The preprocess config should be a list consisting of Dict object."
            logger.error(msg)
            raise Exception(msg)
        mod = importlib.import_module(__name__)
        ops = []
        for op_config in config:
            name = list(op_config)[0]
            param = {} if op_config[name] is None else op_config[name]
            op = getattr(mod, name)(**param)
            ops.append(op)
        return ops

    def __call__(self, img, img_info=None):
        if img_info:
            for op in self.ops:
                img, img_info = op(img, img_info)
            return img, img_info
        else:
            for op in self.ops:
                img = op(img)
            return img


class DecodeImage(object):
    """ decode image """

    def __init__(self, to_rgb=True, to_np=False, channel_first=False):
        self.to_rgb = to_rgb
        self.to_np = to_np  # to numpy
        self.channel_first = channel_first  # only enabled when to_np is True

    def __call__(self, img, img_info=None):
        if six.PY2:
            assert type(img) is str and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        else:
            assert type(img) is bytes and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        data = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(data, 1)
        if self.to_rgb:
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (
                img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        if img_info:
            img_info["im_shape"] = np.array(img.shape[:2], dtype=np.float32)
            img_info["scale_factor"] = np.array([1., 1.], dtype=np.float32)
            return img, img_info
        else:
            return img


class UnifiedResize(object):
    def __init__(self, interpolation=None, backend="cv2"):
        _cv2_interp_from_str = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'area': cv2.INTER_AREA,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        _pil_interp_from_str = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'box': Image.BOX,
            'lanczos': Image.LANCZOS,
            'hamming': Image.HAMMING
        }

        def _pil_resize(src, size, resample):
            pil_img = Image.fromarray(src)
            pil_img = pil_img.resize(size, resample)
            return np.asarray(pil_img)

        if backend.lower() == "cv2":
            if isinstance(interpolation, str):
                interpolation = _cv2_interp_from_str[interpolation.lower()]
            # compatible with opencv < version 4.4.0
            elif interpolation is None:
                interpolation = cv2.INTER_LINEAR
            self.resize_func = partial(cv2.resize, interpolation=interpolation)
        elif backend.lower() == "pil":
            if isinstance(interpolation, str):
                interpolation = _pil_interp_from_str[interpolation.lower()]
            self.resize_func = partial(_pil_resize, resample=interpolation)
        else:
            logger.warning(
                f"The backend of Resize only support \"cv2\" or \"PIL\". \"f{backend}\" is unavailable. Use \"cv2\" instead."
            )
            self.resize_func = cv2.resize

    def __call__(self, src, size):
        return self.resize_func(src, size)


class ResizeImage(object):
    """ resize image """

    def __init__(self,
                 size=None,
                 resize_short=None,
                 interpolation=None,
                 backend="cv2"):
        if resize_short is not None and resize_short > 0:
            self.resize_short = resize_short
            self.w = None
            self.h = None
        elif size is not None:
            self.resize_short = None
            self.w = size if type(size) is int else size[0]
            self.h = size if type(size) is int else size[1]
        else:
            raise Exception("invalid params for ReisizeImage for '\
                'both 'size' and 'resize_short' are None")

        self._resize_func = UnifiedResize(
            interpolation=interpolation, backend=backend)

    def __call__(self, img, img_info=None):
        img_h, img_w = img.shape[:2]
        if self.resize_short is not None:
            percent = float(self.resize_short) / min(img_w, img_h)
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))
        else:
            w = self.w
            h = self.h
        img = self._resize_func(img, (w, h))
        if img_info:
            img_info["input_shape"] = img.shape[:2]
            img_info["scale_factor"] = np.array(
                [img.shape[0] / img_h, img.shape[1] / img_w]).astype("float32")
            return img, img_info
        else:
            return img


class CropImage(object):
    """ crop image """

    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

    def __call__(self, img, img_info=None):
        w, h = self.size
        img_h, img_w = img.shape[:2]

        if img_h < h or img_w < w:
            raise Exception(
                f"The size({h}, {w}) of CropImage must be greater than size({img_h}, {img_w}) of image. Please check image original size and size of ResizeImage if used."
            )

        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        img = img[h_start:h_end, w_start:w_end, :]
        if img_info:
            img_info["input_shape"] = img.shape[:2]
            # TODO(gaotingquan): im_shape is needed to update?
            img_info["im_shape"] = np.array(img.shape[:2], dtype=np.float32)
            return img, img_info
        else:
            return img


class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self,
                 scale=None,
                 mean=None,
                 std=None,
                 order='chw',
                 output_fp16=False,
                 channel_num=3):
        if isinstance(scale, str):
            scale = eval(scale)
        assert channel_num in [
            3, 4
        ], "channel number of input image should be set to 3 or 4."
        self.channel_num = channel_num
        self.output_dtype = 'float16' if output_fp16 else 'float32'
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        self.order = order
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if self.order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img, img_info=None):
        if isinstance(img, Image.Image):
            img = np.array(img)

        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"

        img = (img.astype('float32') * self.scale - self.mean) / self.std

        if self.channel_num == 4:
            img_h = img.shape[1] if self.order == 'chw' else img.shape[0]
            img_w = img.shape[2] if self.order == 'chw' else img.shape[1]
            pad_zeros = np.zeros(
                (1, img_h, img_w)) if self.order == 'chw' else np.zeros(
                    (img_h, img_w, 1))
            img = (np.concatenate(
                (img, pad_zeros), axis=0)
                   if self.order == 'chw' else np.concatenate(
                       (img, pad_zeros), axis=2))
        img = img.astype(self.output_dtype)
        if img_info:
            return img, img_info
        else:
            return img


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self):
        pass

    def __call__(self, img, img_info=None):
        if isinstance(img, Image.Image):
            img = np.array(img)

        img = img.transpose((2, 0, 1))
        if img_info:
            return img, img_info
        else:
            return img
