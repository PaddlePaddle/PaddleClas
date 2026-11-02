import csv
import os

import numpy as np
from paddle.io import Dataset

from ppcls.data.dataloader.common_dataset import create_operators
from ppcls.data.preprocess import transform
from ppcls.utils import logger

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')


def _load_label_map(label_map_file):
    label_map = {}
    with open(label_map_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) != 2:
                continue
            label_map[parts[1].strip()] = int(parts[0])
    return label_map


class VideoDataset(Dataset):
    """Video classification dataset.

    Args:
        image_root (str): directory that contains one sub-directory per class.
        cls_label_path (str, optional): csv file with columns id,path,label.
            Paths are resolved relative to the csv file.
        label_map_file (str, optional): file with lines "index class_name".
            When given, class names are mapped to these indexes.
        transform_ops (list, optional): list of transform op(s).
    """

    def __init__(self,
                 image_root,
                 cls_label_path=None,
                 label_map_file=None,
                 transform_ops=None):
        self.image_root = image_root
        self.ops = create_operators(transform_ops)
        self.label_map = _load_label_map(
            label_map_file) if label_map_file else None

        self.videos = []
        raw_labels = []
        if cls_label_path:
            self._load_from_csv(cls_label_path, raw_labels)
        else:
            self._scan_directory(raw_labels)
        if not self.videos:
            raise ValueError(f"No video found in {image_root}")
        self.labels = self._encode_labels(raw_labels)

    def _load_from_csv(self, cls_label_path, raw_labels):
        csv_dir = os.path.dirname(os.path.abspath(cls_label_path))
        with open(cls_label_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                video_path = row['path']
                if not os.path.isabs(video_path):
                    video_path = os.path.normpath(
                        os.path.join(csv_dir, video_path))
                if not os.path.exists(video_path):
                    logger.warning(
                        f"Skip missing video: {video_path}")
                    continue
                self.videos.append(video_path)
                raw_labels.append(row['label'])

    def _scan_directory(self, raw_labels):
        for class_name in sorted(os.listdir(self.image_root)):
            class_dir = os.path.join(self.image_root, class_name)
            if class_name.startswith('.') or not os.path.isdir(class_dir):
                continue
            for file_name in sorted(os.listdir(class_dir)):
                if file_name.lower().endswith(VIDEO_EXTENSIONS):
                    self.videos.append(os.path.join(class_dir, file_name))
                    raw_labels.append(class_name)

    def _encode_labels(self, raw_labels):
        if self.label_map is None:
            label_names = sorted(set(raw_labels))
            self.label_map = {
                name: idx
                for idx, name in enumerate(label_names)
            }
        labels = []
        for name in raw_labels:
            key = name.replace('_', ' ')
            if key not in self.label_map:
                raise KeyError(
                    f"Label '{key}' is not in the label map file")
            labels.append(self.label_map[key])
        return labels

    def __getitem__(self, idx):
        try:
            data = [self.videos[idx], self.labels[idx]]
            if self.ops:
                data = transform(data, self.ops)
            return data[0], data[1]
        except Exception as ex:
            logger.error(
                f"Exception occured when parse {self.videos[idx]}: {ex}")
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.videos)

    @property
    def class_num(self):
        if not self.labels:
            return 0
        return max(self.labels) + 1
