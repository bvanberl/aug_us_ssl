from typing import List

import torch
import torchvision
from torchvision.transforms import ToTensor, v2

from src.augmentations import WaveletDenoise, BrightnessContrastChange, CLAHETransform, \
    ProbeTypeChange, ConvexityMutation, DepthChange, SpeckleNoise, GaussianNoise, SaltAndPepperNoise

torchvision.disable_beta_transforms_warning()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_normalize_transform(
        mean_pixel_val: List[float] = None,
        std_pixel_val: List[float] = None
) -> v2.Normalize:
    """Creates a pixel normalization transformation,

    Produces a pixel normalization transformation that
    scales pixel values to a desired mean and standard
    deviation. Defaults to ImageNet values.
    :param mean_pixel_val: Channel-wise means
    :param std_pixel_val: Channel-wise standard deviation
    :return: Normalization transform
    """
    if mean_pixel_val is None:
        mean_pixel_val = IMAGENET_MEAN
    if std_pixel_val is None:
        std_pixel_val = IMAGENET_STD
    return v2.Normalize(mean=mean_pixel_val, std=std_pixel_val)


def get_validation_scaling(
        height: int,
        width: int,
        gray_to_rgb: bool = True,
        resize_first: bool = False,
        mean_pixel_val: List[float] = None,
        std_pixel_val: List[float] = None
) -> v2.Compose:
    """Defines augmentation pipeline for supervised learning experiments.
    :param height: Image height
    :param width: Image width
    :param gray_to_rgb: If True, convert grayscale inputs to 3-channel RGB
    :param resize_first: If True, resize image to (height, width) before transforms
    :param mean_pixel_val: Channel-wise means
    :param std_pixel_val: Channel-wise standard deviation
    :return: Callable augmentation pipeline
    """
    transforms = [
        v2.Resize((height, width), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        get_normalize_transform(mean_pixel_val, std_pixel_val)
    ]
    if resize_first:
        transforms.insert(1, v2.Resize((height, width), antialias=True))
    if gray_to_rgb:
        transforms.insert(1, v2.Lambda(lambda x: x.repeat(3, 1, 1)))
    return v2.Compose(transforms)


def get_byol_augmentations(
        height: int,
        width: int,
        gray_to_rgb: bool = True,
        resize_first: bool = False,
        mean_pixel_val: List[float] = None,
        std_pixel_val: List[float] = None,
) -> v2.Compose:
    """
    Applies random data transformations according to the data augmentations
    procedure outlined in VICReg (https://arxiv.org/pdf/2105.04906.pdf),
    Appendix C.1, which is derived from BYOL.
    :param height: Image height
    :param width: Image width
    :param gray_to_rgb: If True, convert grayscale inputs to 3-channel RGB
    :param Desired input channels
    :param resize_first: If True, resize image to (height, width) before transforms
    :param mean_pixel_val: Channel-wise means
    :param std_pixel_val: Channel-wise standard deviation
    :return: Callable augmentation pipeline
    """
    transforms = [
        v2.RandomResizedCrop((height, width), scale=(0.08, 1.), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0., 0.)], p=0.8),
        v2.RandomApply([v2.GaussianBlur(23)], p=0.8),
        v2.RandomSolarize(0.5, p=0.1),
        v2.ToDtype(torch.float32, scale=True),
        get_normalize_transform(mean_pixel_val, std_pixel_val)
    ]
    if resize_first:
        transforms.insert(1, v2.Resize((height, width), antialias=True))
    if gray_to_rgb:
        transforms.insert(1, v2.Lambda(lambda x: x.repeat(3, 1, 1)))
    return v2.Compose(transforms)


def get_august_augmentations(
        gray_to_rgb: bool = True,
        wavelet_denoise_prob: float = 0.5,
        brightness_contrast_prob: float = 0.5,
        clahe_prob: float = 0.2,
        probe_type_prob: float = 0.333,
        convexity_prob: float = 0.5,
        depth_prob: float = 0.333,
        speckle_prob: float = 0.333,
        gaussian_prob: float = 0.333,
        sp_prob: float = 0.2,
        mean_pixel_val: List[float] = None,
        std_pixel_val: List[float] = None
):
    """Applies random transformations to input B-mode image.

    Possible transforms include random crop & resize, contrast
    change, Gaussian blur, and horizontal flip.
    :param gray_to_rgb: If True, convert grayscale inputs to 3-channel RGB
    :param wavelet_denoise_prob: Probability of applying wavelet denoise augmentation
    :param brightness_contrast_prob: Probability of applying brightness/contrast augmentation
    :param clahe_prob: Probability of applying CLAHE augmentation
    :param probe_type_prob: Probability of applying probe type change augmentation
    :param convexity_prob: Probability of applying convexity mutation augmentation
    :param depth_prob: Probability of applying depth change simulation augmentation
    :param speckle_prob: Probability of applying speckle noise augmentation
    :param gaussian_prob: Probability of applying Gaussian noise augmentation
    :param sp_prob: Probability of applying salt & pepper noise augmentation
    :param mean_pixel_val: Channel-wise means
    :param std_pixel_val: Channel-wise standard deviation
    :return: Callable augmentation pipeline
    """
    transforms = [
        v2.RandomApply(
            [WaveletDenoise(j_0=2)],
            p=wavelet_denoise_prob
        ),
        v2.RandomApply(
            [BrightnessContrastChange(min_brightness=0.5, max_brightness=1.5, min_contrast=0.5, max_contrast=1.5)],
            p=brightness_contrast_prob
        ),
        v2.RandomApply(
            [CLAHETransform(min_clip_limit=20., max_clip_limit=25., tile_grid_size=(6, 6))],
            p=clahe_prob
        ),
        v2.RandomApply(
            [ProbeTypeChange(square_roi=True, min_linear_width_frac=1.0, max_linear_width_frac=1.0)],
            p=probe_type_prob
        ),
        v2.RandomApply(
            [ConvexityMutation(square_roi=True, min_top_width=0., max_top_width=0.5)],
            p=convexity_prob
        ),
        v2.RandomApply(
            [DepthChange(min_depth_factor=1.0, max_depth_factor=1.5)],
            p=depth_prob
        ),
        v2.RandomApply(
            [SpeckleNoise(square_roi=True, min_lateral_res=50, max_lateral_res=100,
                                   min_axial_res=50, max_axial_res=100, min_phasors=5, max_phasors=15)],
                      p=speckle_prob
        ),
        v2.RandomApply(
            [GaussianNoise(min_sigma=0.5, max_sigma=3.0)],
            p=gaussian_prob
        ),
        v2.RandomApply(
            [SaltAndPepperNoise(min_salt_frac=0.001, max_salt_frac=0.005, min_pepper_frac=0.001,
                                                    max_pepper_frac=0.005)],
            p=sp_prob
        ),
        #v2.ToDtype(torch.float32, scale=True),
        #get_normalize_transform(mean_pixel_val, std_pixel_val)
    ]
    if gray_to_rgb:
        transforms.insert(1, v2.Lambda(lambda x: x.repeat(3, 1, 1)))
    return v2.Compose(transforms)
