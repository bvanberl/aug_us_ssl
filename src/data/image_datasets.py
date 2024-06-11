import os
from typing import Optional, Callable
import sys
import traceback

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
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
            height: int = 224,
            width: int = 224,
            channels: int = 3,
            device: str = 'cpu'
    ):
        self.device = device
        self.img_root_dir = img_root_dir
        self.mask_root_dir = mask_root_dir

        self.image_paths = [p.replace("\\", "/") for p in img_records["filepath"].tolist()]
        self.mask_paths = [f"{id}_mask{img_ext}" for id in img_records["id"].tolist()]
        self.keypoints = img_records[["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]].values.astype(np.float32)
        self.probe_types = np.array([Probe[p.replace(' ', '_').upper()].value for p in img_records["probe_type"].tolist()])

        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.img_ext = img_ext
        if channels == 3:
            self.read_mode = ImageReadMode.RGB
        elif channels == 1:
            self.read_mode = ImageReadMode.GRAY
        else:
            self.read_mode = ImageReadMode.UNCHANGED

        self.height = height
        self.width = width
        self.cardinality = len(self.image_paths)

    def __len__(self):
        return self.cardinality

    def __getitem__(self, idx):

        # Load and copy image
        image_path = os.path.join(self.img_root_dir, self.image_paths[idx])
        x1 = read_image(image_path, self.read_mode).to(self.device)
        x2 = torch.clone(x1)

        # Load ancillary inputs
        mask_path = os.path.join(self.mask_root_dir, self.mask_paths[idx])
        mask = read_image(mask_path).to(self.device)
        label = torch.tensor(-1)    # No label
        keypoints = torch.tensor(self.keypoints[idx], device=self.device)
        probe_type = torch.tensor(self.probe_types[idx], device=self.device)


        # Apply data augmentation transforms
        try:
            if self.transforms1:
                x1 = self.transforms1(x1, label, keypoints, mask, probe_type)[0]
            if self.transforms2:
                x2 = self.transforms2(x2, label, keypoints, mask, probe_type)[0]
        except Exception as e:
            print(e, traceback.format_exc(), image_path, keypoints)
        return x1, x2


class ImageClassificationDataset(Dataset):
    def __init__(
            self,
            img_root_dir: str,
            img_paths: List[str],
            labels: np.ndarray,
            n_classes: int,
            transforms: Optional[Callable],
            img_ext: str = ".jpg",
            device: str = "cpu"
    ):
        assert len(img_paths) == len(labels), "Number of images and labels must match."
        self.image_paths = [p.replace("\\", "/") for p in img_paths]
        self.img_ext = img_ext
        self.img_root_dir = img_root_dir
        self.n_classes = n_classes
        if n_classes == 2:
            self.labels = torch.from_numpy(labels. astype(np.float32)).unsqueeze(-1)
        else:
            self.labels = torch.from_numpy(labels)
        self.label_freqs = np.unique(labels, return_counts=True)[1] / len(labels)
        self.transforms = transforms
        self.device = device
        self.cardinality = len(self.image_paths)

    def __len__(self):
        return self.cardinality

    def __getitem__(self, idx):

        # Load image
        image_path = os.path.join(
            self.img_root_dir,
            self.image_paths[idx]
        )
        x = read_image(image_path).to(self.device)

        # Apply data augmentation transforms
        if self.transforms:
            x = self.transforms(x)

        y = self.labels[idx]
        return x, y


def get_augmentation_transforms(
        pipeline: str,
        height: int,
        width: int,
        resize: bool = True,
        exclude_idx: int = -1,
        square_roi: bool = True,
        **augment_kwargs
) -> Compose:
    """Get augmentation transformation pipelines

    :param pipeline: Name of pipeline.
                     One of {'byol', 'august', 'supervised', or 'none'}
    :param height: Image height
    :param width: Image width
    :param pipeline_kwargs: Pipeline keyword arguments
    :return: Augmentation pipelines for first and second images
    """
    pipeline = pipeline.lower()
    if pipeline == "byol_original":
        return get_original_byol_augmentations(height, width, resize=resize, exclude_idx=exclude_idx)
    if pipeline == "byol_grayscale":
        return get_grayscale_byol_augmentations(height, width, resize=resize, exclude_idx=exclude_idx)
    elif pipeline == "august":
        return get_august_augmentations(height, width, resize=resize, exclude_idx=exclude_idx, square_roi=square_roi, **augment_kwargs)
    elif pipeline == "supervised":
        return get_supervised_augmentations(height, width, resize=resize, **augment_kwargs)
    else:
        if pipeline != "none":
            print(f"Unrecognized augmentation pipeline: {pipeline}.\n"
                            f"No augmentations will be applied.")
        return get_validation_scaling(height, width, resize=resize)


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
        resize: bool = True,
        device: str = 'cpu',
        exclude_idx: int = -1,
        square_roi: bool = True,
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
    :param device: Device on which to execute data transformations
    :param preprocess_kwargs: Keyword arguments for preprocessing
    :param shuffle: Flag indicating whether the US beam is cropped and resized to be square
    :return: A batched dataset ready for iterating over preprocessed batches
    '''

    # Construct the dataset
    augment1 = get_augmentation_transforms(
        augment_pipeline,
        height,
        width,
        resize=resize,
        exclude_idx=exclude_idx,
        square_roi=square_roi,
        #**preprocess_kwargs["augmentation"]
    )
    augment2 = get_augmentation_transforms(
        augment_pipeline,
        height,
        width,
        resize=resize,
        exclude_idx=exclude_idx,
        square_roi=square_roi
        # **preprocess_kwargs["augmentation"]
    )

    dataset = ImagePretrainDataset(
        file_df,
        img_root,
        mask_root,
        transforms1=augment1,
        transforms2=augment2,
        device=device,
    )

    persistent_workers = n_workers > 0
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_workers,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=persistent_workers
    )
    return data_loader


def prepare_train_dataloader(
        img_root: str,
        label_name: str,
        file_df: pd.DataFrame,
        batch_size: int,
        width: int,
        height: int,
        augment_pipeline: str = "august",
        shuffle: bool = False,
        n_workers: int = 0,
        drop_last: bool = False,
        resize: bool = True,
        **preprocess_kwargs
) -> DataLoader:
    '''
    Constructs a dataset for a classifier.
    :param img_root: Root directory in which all images are stored. Will be prepended to path in frames table.
    :param label_name: Name of label column
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
    augmentations = get_augmentation_transforms(
        augment_pipeline,
        height,
        width,
        resize=resize
    )

    n_classes = file_df[label_name].nunique()
    dataset = ImageClassificationDataset(
        img_root,
        file_df['filepath'].tolist(),
        file_df[label_name].to_numpy(),
        n_classes,
        transforms=augmentations
    )

    persistent_workers = n_workers > 0
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_workers,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=persistent_workers
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
        n_train_workers: int = 0,
        n_val_workers: int = 0,
        resize: bool = True,
        exclude_idx: int = -1,
        square_roi: bool = True,
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
    labelled_train_frames_path = os.path.join(splits_dir, 'train_set_frames.csv')
    labelled_train_clips_path = os.path.join(splits_dir, 'train_set_clips.csv')
    if os.path.exists(labelled_train_frames_path) and os.path.exists(labelled_train_clips_path):
        labelled_train_frames_df = pd.read_csv(labelled_train_frames_path)
        labelled_train_clips_df = pd.read_csv(labelled_train_clips_path)
    else:
        labelled_train_frames_df = pd.DataFrame()
        labelled_train_clips_df = pd.DataFrame()
    unlabelled_frames_path = os.path.join(splits_dir, 'unlabelled_frames.csv')
    unlabelled_clips_path = os.path.join(splits_dir, 'unlabelled_clips.csv')
    if os.path.exists(unlabelled_frames_path) and os.path.exists(unlabelled_clips_path):
        unlabelled_frames_df = pd.read_csv(unlabelled_frames_path)
        unlabelled_clips_df = pd.read_csv(unlabelled_clips_path)
    else:
        unlabelled_frames_df = pd.DataFrame()
        unlabelled_clips_df = pd.DataFrame()
    val_frames_path = os.path.join(splits_dir, 'val_set_frames.csv')
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
        n_workers=n_train_workers,
        drop_last=True,
        resize=resize,
        exclude_idx=exclude_idx,
        square_roi=square_roi,
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
            shuffle=True,
            channels=3,
            n_workers=n_val_workers,
            drop_last=False,
            resize=resize,
            exclude_idx=exclude_idx,
            square_roi=square_roi,
            **preprocess_kwargs
        )
    else:
        val_loader = None

    return train_loader, val_loader


def load_data_for_train(
        image_dir: str,
        label_name: str,
        width: int,
        height: int,
        splits_dir: str,
        batch_size: int,
        augment_pipeline: str = "august",
        n_train_workers: int = 0,
        n_val_workers: int = 0,
        resize: bool = True,
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
    :param preprocess_kwargs: Keyword arguments for preprocessing
    :return: dataset for pretraining
    """

    # Load data for training
    train_frames_path = os.path.join(splits_dir, f'train_set_frames.csv')
    train_clips_path = os.path.join(splits_dir, 'train_set_clips.csv')
    if os.path.exists(train_frames_path) and os.path.exists(train_clips_path):
        train_frames_df = pd.read_csv(train_frames_path)
        train_clips_df = pd.read_csv(train_clips_path)
    else:
        train_frames_df = pd.DataFrame()
        train_clips_df = pd.DataFrame()
    train_clips_df = train_clips_df.loc[train_clips_df[label_name] != -1]
    train_frames_df = train_frames_df.loc[train_frames_df[label_name] != -1]
    print("Training clips:\n", train_clips_df.describe())

    val_frames_path = os.path.join(splits_dir, f'val_set_frames.csv')
    val_clips_path = os.path.join(splits_dir, 'val_set_clips.csv')
    if os.path.exists(val_frames_path) and os.path.exists(val_clips_path):
        val_frames_df = pd.read_csv(val_frames_path)
        val_clips_df = pd.read_csv(val_clips_path)
    else:
        val_frames_df = pd.DataFrame()
        val_clips_df = pd.DataFrame()
    val_clips_df = val_clips_df.loc[val_clips_df[label_name] != -1]
    val_frames_df = val_frames_df.loc[val_frames_df[label_name] != -1]
    print("Validation clips:\n", val_clips_df.describe())

    train_loader = prepare_train_dataloader(
        image_dir,
        label_name,
        train_frames_df,
        batch_size,
        width,
        height,
        augment_pipeline=augment_pipeline,
        shuffle=True,
        channels=3,
        n_workers=n_train_workers,
        drop_last=True,
        resize=resize,
        **preprocess_kwargs
    )
    if val_frames_df.shape[0] > 0:
        val_loader = prepare_train_dataloader(
            image_dir,
            label_name,
            val_frames_df,
            batch_size,
            width,
            height,
            augment_pipeline="none",
            shuffle=True,
            channels=3,
            n_workers=n_val_workers,
            drop_last=False,
            resize=resize,
            **preprocess_kwargs
        )
    else:
        val_loader = None

    return train_loader, val_loader