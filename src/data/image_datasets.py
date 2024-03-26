import os
from typing import Optional, Callable, List

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
from torchvision.transforms.v2 import Compose
from torchvision.io import read_image, ImageReadMode

from src.constants import Probe
from src.augmentations.pipelines import *

class ImagePretrainDataset(Dataset):
    def __init__(
            self,
            img_records: pd.DataFrame,
            img_root_dir: str,
            mask_root_dir: str,
            transforms1: Optional[Callable],
            transforms2: Optional[Callable],
            img_ext: str = ".jpg",
            channels: int = 3
    ):
        self.img_root_dir = img_root_dir
        self.mask_root_dir = mask_root_dir

        self.image_paths = [p.replace("\\", "/") for p in img_records["filepath"].tolist()]
        self.mask_paths = [f"{id}_mask{img_ext}" for id in img_records["id"].tolist()]
        self.keypoints = torch.from_numpy(img_records[["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]].values.astype(np.float32))
        self.probe_types = torch.tensor([Probe[p.replace(' ', '_')].value for p in img_records["probe_type"].tolist()])

        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.img_ext = img_ext
        if channels == 3:
            self.read_mode = ImageReadMode.RGB
        elif channels == 1:
            self.read_mode = ImageReadMode.GRAY
        else:
            self.read_mode = ImageReadMode.UNCHANGED

        self.cardinality = len(self.image_paths)

    def __len__(self):
        return self.cardinality

    def __getitem__(self, idx):

        # Load and copy image
        image_path = os.path.join(self.img_root_dir, self.image_paths[idx])
        x1 = read_image(image_path, self.read_mode)
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


def get_augmentation_transforms_pretrain(
        pipeline: str,
        height: int,
        width: int,
        **augment_kwargs
) -> (Compose, Compose):
    """Get augmentation transformation pipelines

    :param pipeline: Name of pipeline.
                     One of {'byol', 'august', or 'none'}
    :param height: Image height
    :param width: Image width
    :param pipeline_kwargs: Pipeline keyword arguments
    :return: Augmentation pipelines for first and second images
    """
    pipeline = pipeline.lower()
    if pipeline == "byol":
        return (
            get_byol_augmentations(height, width),
            get_byol_augmentations(height, width)
        )
    elif pipeline == "august":
        return (
            get_august_augmentations(**augment_kwargs),
            get_august_augmentations(**augment_kwargs)
        )
    else:
        if pipeline != "none":
            print(f"Unrecognized augmentation pipeline: {pipeline}.\n"
                            f"No augmentations will be applied.")
        return (
            get_validation_scaling(),
            get_validation_scaling(),
        )


def prepare_pretrain_dataloader(
        img_root: str,
        mask_root: str,
        file_df: pd.DataFrame,
        batch_size: int,
        width: int,
        height: int,
        augment_pipeline: str = "august",
        shuffle: bool = False,
        n_workers: int = 0,
        drop_last: bool = False,
        **preprocess_kwargs
) -> DataLoader:
    '''
    Constructs a dataset for a joint embedding self-supervised pretraining task.
    :param img_root: Root directory in which all images are stored. Will be prepended to path in frames table.
    :param file_df: A table of US image properties
    :param batch_size: Batch size for pretraining
    :param width: Desired width of US images
    :param height: Desired height of US images
    :param augment: If True, applies data augmentation transforms to the inputs
    :param shuffle: Flag indicating whether to shuffle the dataset
    :param channels: Number of channels
    :param max_time_delta: Maximum temporal separation of two frames
    :param n_workers: Number of workers for preloading batches
    :param world_size: Number of processes. If 1, then not using distributed mode
    :param drop_last: If True, drops the last batch in the data loader if smaller than the batch size
    :param preprocess_kwargs: Keyword arguments for preprocessing
    :return: A batched dataset ready for iterating over preprocessed batches
    '''

    # Construct the dataset
    augment1, augment2 = get_augmentation_transforms_pretrain(
        augment_pipeline,
        height,
        width,
        #**preprocess_kwargs["augmentation"]
    )

    dataset = ImagePretrainDataset(
        file_df,
        img_root,
        mask_root,
        transforms1=augment1,
        transforms2=augment2
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_workers,
        drop_last=drop_last,
        pin_memory=True
    )
    return data_loader


def load_data_for_pretrain(
        image_dir: str,
        mask_dir: str,
        width: int,
        height: int,
        splits_dir: str,
        batch_size: int,
        augment_pipeline: str = "august",
        use_unlabelled: bool = True,
        n_workers: int = 0,
        **preprocess_kwargs
) -> (DataLoader, pd.DataFrame):
    """
    Retrieve data, data splits, and returns an iterable preprocessed dataset for pretraining
    :param cfg: The config.yaml file dictionary
    :param batch_size: Batch size for datasets
    :param run: The wandb run object that is initialized
    :param data_artifact_name: Artifact name for raw data and files
    :param data_version: Artifact version for raw data
    :param splits_artifact_name: Artifact name for train/val/test splits
    :param splits_version: Artifact version for train/val/test splits
    :param redownload_data: Flag indicating whether the dataset artifact should be redownloaded
    :param augment_pipeline: Augmentation strategy identifier
    :param use_unlabelled: Flag indicating whether to use the unlabelled data in
    :param channels: Number of channels
    :param max_pixel_val: Maximum value for pixel intensity
    :param width: Desired width of images
    :param height: Desired height of images
    :param us_mode: US Mode. Either 'bmode' or 'mmode'.
    :param world_size: Number of processes. If 1, then not using distributed mode
    :param n_workers: Number of workers for preloading batches
    :param preprocess_kwargs: Keyword arguments for preprocessing
    :return: dataset for pretraining
    """

    # Load data for pretraining
    labelled_train_frames_path = os.path.join(splits_dir, f'train_set_frames.csv')
    labelled_train_clips_path = os.path.join(splits_dir, 'train_set_clips.csv')
    if os.path.exists(labelled_train_frames_path) and os.path.exists(labelled_train_clips_path):
        labelled_train_frames_df = pd.read_csv(labelled_train_frames_path)
        labelled_train_clips_df = pd.read_csv(labelled_train_clips_path)
    else:
        labelled_train_frames_df = pd.DataFrame()
        labelled_train_clips_df = pd.DataFrame()
    unlabelled_frames_path = os.path.join(splits_dir, f'unlabelled_frames.csv')
    unlabelled_clips_path = os.path.join(splits_dir, 'unlabelled_clips.csv')
    if os.path.exists(unlabelled_frames_path) and os.path.exists(unlabelled_clips_path):
        unlabelled_frames_df = pd.read_csv(unlabelled_frames_path)
        unlabelled_clips_df = pd.read_csv(unlabelled_clips_path)
    else:
        unlabelled_frames_df = pd.DataFrame()
        unlabelled_clips_df = pd.DataFrame()
    val_frames_path = os.path.join(splits_dir, f'val_set_frames.csv')
    val_clips_path = os.path.join(splits_dir, 'val_set_clips.csv')
    if os.path.exists(val_frames_path) and os.path.exists(val_clips_path):
        val_frames_df = pd.read_csv(val_frames_path)
        val_clips_df = pd.read_csv(val_clips_path)
    else:
        val_frames_df = pd.DataFrame()
        val_clips_df = pd.DataFrame()

    if use_unlabelled:
        train_frames_df = pd.concat([labelled_train_frames_df, unlabelled_frames_df])
        train_clips_df = pd.concat([labelled_train_clips_df, unlabelled_clips_df])
    else:
        train_frames_df = labelled_train_frames_df
        train_clips_df = labelled_train_clips_df

    # Add clip-wise keypoints and probe types to frame records
    train_frames_df = pd.merge(
        train_frames_df,
        train_clips_df[["id", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "probe_type"]],
        how='left',
        on='id'
    )
    val_frames_df = pd.merge(
        val_frames_df,
        val_clips_df[["id", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "probe_type"]],
        how='left',
        on='id'
    )

    train_loader = prepare_pretrain_dataloader(
        image_dir,
        mask_dir,
        train_frames_df,
        batch_size,
        width,
        height,
        augment_pipeline=augment_pipeline,
        shuffle=True,
        channels=3,
        n_workers=n_workers,
        drop_last=True,
        **preprocess_kwargs
    )
    if val_frames_df.shape[0] > 0:
        val_loader = prepare_pretrain_dataloader(
            image_dir,
            mask_dir,
            val_frames_df,
            batch_size,
            width,
            height,
            augment_pipeline="none",
            shuffle=False,
            channels=3,
            n_workers=1,
            drop_last=False,
            **preprocess_kwargs
        )
    else:
        val_loader = None

    return train_loader, val_loader