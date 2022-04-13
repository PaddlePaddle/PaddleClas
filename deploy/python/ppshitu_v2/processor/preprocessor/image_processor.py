from functools import partial
import cv2
import numpy as np
import importlib
from PIL import Image

from utils import logger
from processor.base_processor import BaseProcessor


class ImageProcessor(BaseProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.processors = []
        mod = importlib.import_module(__name__)
        for processor_config in config.get("ops"):
            name = list(processor_config)[0]
            param = {} if processor_config[name] is None else processor_config[
                name]
            op = getattr(mod, name)(**param)
            self.processors.append(op)

    def process(self, input_data):
        for processor in self.processors:
            input_data = processor.process(input_data)
        return input_data


class BoxCrop(BaseProcessor):
    def __init__(self, config=None):
        super().__init__(config)
        if self.input_keys is None:
            self.input_keys = ["boxes", "input_image"]
        if self.output_keys is None:
            self.ouput_keys = ["image"]

    def process(self, input_data):
        xmin, ymin, xmax, ymax = input_data[self.input_keys[0]].astype("int")
        image = input_data[self.input_keys[1]]
        crop_img = image[ymin:ymax, xmin:xmax, :].copy()
        input_data[self.output_keys[0]] = crop_img
        return input_data

class GetShapeInfo(BaseProcessor):
    def __init__(self, config=None, order="chw"):
        super().__init__(config)
        self.order = order
        if self.input_keys is None:
            self.input_keys = ["input_image", "image"]
        if self.output_keys is None:
            self.output_keys = ["im_shape", "scale_factor", "input_shape"]

    def process(self, input_data):
        input_image = input_data[self.input_keys[0]]
        image = input_data[self.input_keys[1]]
        if self.order == "hwc":
            input_data[self.output_keys[0]] = np.array(
                (image.shape[:2], ), dtype=np.float32)
            input_data[self.output_keys[1]] = np.array(
                [
                    image.shape[0] / input_image.shape[0],
                    image.shape[1] / input_image.shape[1]
                ],
                dtype=np.float32)
        else:
            input_data[self.output_keys[0]] = np.array(
                (image.shape[1:], ), dtype=np.float32)
            input_data[self.output_keys[1]] = np.array(
                [
                    image.shape[2] / input_image.shape[0],
                    image.shape[1] / input_image.shape[1]
                ],
                dtype=np.float32)
        input_data[self.output_keys[2]] = np.array(image.shape[:2], dtype=np.float32)
        return input_data


class ProxyProcessor(BaseProcessor):
    def __init__(self, config=None):
        super().__init__(config)
        if self.input_keys is None:
            self.input_keys = ["image"]
        if self.output_keys is None:
            self.output_keys = ["image"]

    def process(self, input_data):
        img = input_data[self.input_keys[0]]
        output = self(img)
        input_data[self.output_keys[0]] = output
        return input_data

    def __call__(self, img):
        return img


class ToBatch(ProxyProcessor):
    def __call__(self, img):
        img = img[np.newaxis, :, :, :]
        return img


class ToRGB(ProxyProcessor):
    def __call__(self, img):
        img = img[:, :, ::-1]
        return img


class ToCHWImage(ProxyProcessor):
    def __call__(self, img, img_info=None):
        img = img.transpose((2, 0, 1))
        return img


class ResizeImage(ProxyProcessor):
    def __init__(self,
                 config=None,
                 size=None,
                 resize_short=None,
                 interpolation=None,
                 backend="cv2"):
        super().__init__(config)
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
                f"The backend of Resize only support \"cv2\" or \"PIL\". \"f{backend}\" is unavailable. "
                f"Use \"cv2\" instead.")
            self.resize_func = cv2.resize

    def __call__(self, img):
        img_h, img_w = img.shape[:2]
        if self.resize_short is not None:
            percent = float(self.resize_short) / min(img_w, img_h)
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))
        else:
            w = self.w
            h = self.h
        img = self.resize_func(img, (w, h))
        return img


class CropImage(ProxyProcessor):
    """ crop image """

    def __init__(self, config=None, size=0):
        super().__init__(config)
        if size == 0:
            raise KeyError("CropImage config size is wrong")
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

    def __call__(self, img):
        w, h = self.size
        img_h, img_w = img.shape[:2]

        if img_h < h or img_w < w:
            raise Exception(
                f"The size({h}, {w}) of CropImage must be greater than size({img_h}, {img_w}) of image. "
                f"Please check image original size and size of ResizeImage if used."
            )
        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        img = img[h_start:h_end, w_start:w_end, :]
        return img


class NormalizeImage(ProxyProcessor):
    def __init__(self,
                 config=None,
                 scale=None,
                 mean=None,
                 std=None,
                 order='chw',
                 output_fp16=False,
                 channel_num=3):
        super().__init__(config)
        if isinstance(scale, str):
            scale = eval(scale)
        assert channel_num in [3, 4], \
            "channel number of input image should be set to 3 or 4."
        self.channel_num = channel_num
        self.output_dtype = 'float16' if output_fp16 else 'float32'
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        self.order = order
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if self.order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
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
        return img
