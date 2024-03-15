import os
from typing import Optional, Callable, List

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from torchvision.io import read_image

from src.constants import Probe

class ImagePretrainDataset(Dataset):
    def __init__(
            self,
            img_records: pd.DataFrame,
            img_root_dir: str,
            mask_root_dir: str,
            transforms1: Optional[Callable],
            transforms2: Optional[Callable],
            img_ext: str = ".jpg"
    ):
        self.img_root_dir = img_root_dir
        self.mask_root_dir = mask_root_dir

        self.image_paths = [p.replace("\\", "/") for p in img_records["filepath"].tolist()]
        self.mask_paths = [f"{id}_mask{img_ext}" for id in img_records["id"].tolist()]
        self.keypoints = torch.from_numpy(img_records[["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]].values)
        self.probe_types = torch.tensor([Probe[p.replace(' ', '_')].value for p in img_records["probe_type"].tolist()])

        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.img_ext = img_ext
        self.cardinality = len(self.image_paths)

    def __len__(self):
        return self.cardinality

    def __getitem__(self, idx):

        # Load and copy image
        image_path = os.path.join(self.img_root_dir, self.image_paths[idx])
        x1 = read_image(image_path)
        x2 = torch.clone(x1)

        # Load ancillary inputs
        mask_path = os.path.join(self.mask_root_dir, self.mask_paths[idx])
        mask = read_image(mask_path)
        label = torch.tensor(-1)    # No label
        keypoints = self.keypoints[idx]
        probe_type = self.probe_types[idx]

        # Apply data augmentation transforms
        if self.transforms1:
            x1 = self.transforms1(x1, label, keypoints, mask, probe_type)[0]
        if self.transforms2:
            x2 = self.transforms2(x2, label, keypoints, mask, probe_type)[0]
        return x1, x2


class ImageClassificationDataset(Dataset):
    def __init__(
            self,
            img_root_dir: str,
            img_paths: List[str],
            labels: np.ndarray,
            n_classes: int,
            transforms: Optional[Callable],
            img_ext: str = ".jpg"
    ):
        assert len(img_paths) == len(labels), "Number of images and labels must match."
        self.image_paths = [p.replace("\\", "/") for p in img_paths]
        self.img_ext = img_ext
        self.img_root_dir = img_root_dir
        if n_classes > 2:
            self.labels = one_hot(torch.from_numpy(labels), num_classes=n_classes)
        else:
            self.labels = np.expand_dims(labels, axis=-1).astype(np.float32)
        self.label_freqs = np.bincount(labels[labels != -1]) / labels.shape[0]
        self.transforms = transforms
        self.cardinality = len(self.image_paths)

    def __len__(self):
        return self.cardinality

    def __getitem__(self, idx):

        # Load image
        image_path = os.path.join(
            self.img_root_dir,
            self.image_paths[idx]
        )
        x = read_image(image_path)

        # Apply data augmentation transforms
        if self.transforms:
            x = self.transforms(x)

        y = self.labels[idx]
        return x, y
