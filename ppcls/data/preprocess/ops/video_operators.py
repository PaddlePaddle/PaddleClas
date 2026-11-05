import cv2
import numpy as np


def _read_video(video_path, num_seg):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"No frame found in video: {video_path}")

    if total_frames >= num_seg:
        indices = np.linspace(0, total_frames - 1, num_seg, dtype=int)
    else:
        indices = list(range(total_frames))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            frame = frames[-1] if frames else np.zeros(
                (224, 224, 3), dtype=np.uint8)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) < num_seg:
        frames.extend([frames[-1]] * (num_seg - len(frames)))
    return np.stack(frames)


class DecodeVideo(object):
    """Decode a video and sample num_seg frames evenly."""

    def __init__(self, num_seg=16):
        self.num_seg = num_seg

    def __call__(self, data):
        video_path, label = data
        return [_read_video(video_path, self.num_seg), label]


class ResizeVideo(object):
    """Resize all frames so that the short side equals resize_short."""

    def __init__(self, resize_short=256):
        self.resize_short = resize_short

    def __call__(self, data):
        frames, label = data
        height, width = frames.shape[1:3]
        if height < width:
            new_h, new_w = self.resize_short, int(
                round(width * self.resize_short / height))
        else:
            new_h, new_w = int(round(height * self.resize_short / width)), \
                self.resize_short
        frames = np.stack([
            cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            for f in frames
        ])
        return [frames, label]


class RandCropVideo(object):
    """Randomly crop a region of the given size for the whole video."""

    def __init__(self, size=224):
        self.size = size

    def __call__(self, data):
        frames, label = data
        height, width = frames.shape[1:3]
        top = np.random.randint(0, height - self.size + 1)
        left = np.random.randint(0, width - self.size + 1)
        frames = frames[:, top:top + self.size, left:left + self.size, :]
        return [frames, label]


class CenterCropVideo(object):
    """Center crop a region of the given size for the whole video."""

    def __init__(self, size=224):
        self.size = size

    def __call__(self, data):
        frames, label = data
        height, width = frames.shape[1:3]
        top = (height - self.size) // 2
        left = (width - self.size) // 2
        frames = frames[:, top:top + self.size, left:left + self.size, :]
        return [frames, label]


class RandFlipVideo(object):
    """Randomly flip the whole video."""

    def __init__(self, flip_code=1):
        self.flip_code = flip_code

    def __call__(self, data):
        frames, label = data
        if np.random.rand() < 0.5:
            frames = np.stack(
                [cv2.flip(f, self.flip_code) for f in frames])
        return [frames, label]


class NormalizeVideo(object):
    """Normalize frames and convert them to shape [C, T, H, W]."""

    def __init__(self, scale=1.0 / 255.0, mean=None, std=None):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]
        self.mean = np.array(
            mean, dtype=np.float32).reshape(1, 1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 1, 3)

    def __call__(self, data):
        frames, label = data
        frames = (frames.astype(np.float32) * self.scale -
                  self.mean) / self.std
        frames = frames.transpose(3, 0, 1, 2)
        return [frames, label]
