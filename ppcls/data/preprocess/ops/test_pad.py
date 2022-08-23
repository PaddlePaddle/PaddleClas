import numpy as np

import paddle.vision.transforms as T
import cv2


class Pad(object):
    """
    Pads the given PIL.Image on all sides with specified padding mode and fill value.
    adapted from: https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#Pad
    """

    def __init__(self,
                 padding: int,
                 fill: int=0,
                 padding_mode: str="constant",
                 backend: str="pil"):
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode
        self.backend = backend
        assert backend in [
            "pil", "cv2"
        ], f"backend in Pad must in ['pil', 'cv2'], but got {backend}"

    def _parse_fill(self, fill, img, min_pil_version, name="fillcolor"):
        # Process fill color for affine transforms
        major_found, minor_found = (int(v)
                                    for v in PILLOW_VERSION.split('.')[:2])
        major_required, minor_required = (int(v) for v in
                                          min_pil_version.split('.')[:2])
        if major_found < major_required or (major_found == major_required and
                                            minor_found < minor_required):
            if fill is None:
                return {}
            else:
                msg = (
                    "The option to fill background area of the transformed image, "
                    "requires pillow>={}")
                raise RuntimeError(msg.format(min_pil_version))

        num_bands = len(img.getbands())
        if fill is None:
            fill = 0
        if isinstance(fill, (int, float)) and num_bands > 1:
            fill = tuple([fill] * num_bands)
        if isinstance(fill, (list, tuple)):
            if len(fill) != num_bands:
                msg = (
                    "The number of elements in 'fill' does not match the number of "
                    "bands of the image ({} != {})")
                raise ValueError(msg.format(len(fill), num_bands))

            fill = tuple(fill)

        return {name: fill}

    def __call__(self, img):
        if self.backend == "pil":
            opts = self._parse_fill(self.fill, img, "2.3.0", name="fill")
            if img.mode == "P":
                palette = img.getpalette()
                img = ImageOps.expand(img, border=self.padding, **opts)
                img.putpalette(palette)
                return img
            return ImageOps.expand(img, border=self.padding, **opts)
        else:
            img = cv2.copyMakeBorder(
                img,
                self.padding,
                self.padding,
                self.padding,
                self.padding,
                cv2.BORDER_CONSTANT,
                value=(self.fill, self.fill, self.fill))
            return img


img = np.random.randint(0, 255, [3, 4, 3], dtype=np.uint8)

for p in range(0, 10):
    for v in range(0, 10):
        img_1 = Pad(p, v, backend="cv2")(img)
        img_2 = T.Pad(p, (v, v, v))(img)
        print(f"{p} - {v}", np.allclose(img_1, img_2))
        if not np.allclose(img_1, img_2):
            print(img_1[..., 0], "\n", img_2[..., 0])
            print(img_1[..., 1], "\n", img_2[..., 1])
            print(img_1[..., 2], "\n", img_2[..., 2])
            exit(0)
