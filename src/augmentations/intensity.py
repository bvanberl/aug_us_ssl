from typing import Tuple

import torch
from torch import nn, Tensor
from torchvision.transforms.v2 import functional as tvf
from kornia.enhance import equalize_clahe
import numpy as np


class CLAHETransform(nn.Module):

    def __init__(
            self,
            min_clip_limit: float = 30.0,
            max_clip_limit: float = 50.0,
            tile_grid_size: Tuple[int, int] = (8, 8)
    ):
        super(CLAHETransform, self).__init__()
        self.min_clip_limit = min_clip_limit
        self.max_clip_limit = max_clip_limit
        self.tile_grid_size = tile_grid_size

    def forward(
            self,
            image: Tensor,
            label: Tensor,
            keypoints: Tensor,
            mask: Tensor,
            probe: Tensor,
            **kwargs
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """Applies CLAHE to input image.

        Args:
            image: 3D Image tensor, with shape (c, h, w)
            label: Label for the image, which is unaltered
            keypoints: 1D tensor with shape (8,) containing beam mask keypoints
                       with format [x1, y1, x2, y2, x3, y3, x4, y4]
            mask: Beam mask, with shape (1, h, w)
            probe: Probe type of the image

        Returns:
            augmented image, label, keypoints, mask, probe type
        """

        # Sample a random clip
        clip_limit = np.random.uniform(self.min_clip_limit, self.max_clip_limit)
        new_image = equalize_clahe(image / 255., clip_limit, self.tile_grid_size)
        new_image = (new_image * 255) .to(torch.uint8)
        new_image = new_image * mask
        return new_image, label, keypoints, mask, probe


class BrightnessContrastChange(nn.Module):

    def __init__(
            self,
            min_brightness: float = 1.,
            max_brightness: float = 1.,
            min_contrast: float = 1.,
            max_contrast: float = 1.
    ):
        super(BrightnessContrastChange, self).__init__()
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast

    def forward(
            self,
            image: Tensor,
            label: Tensor,
            keypoints: Tensor,
            mask: Tensor,
            probe: Tensor,
            **kwargs
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """Applies CLAHE to input image.

        Args:
            image: 3D Image tensor, with shape (c, h, w)
            label: Label for the image, which is unaltered
            keypoints: 1D tensor with shape (8,) containing beam mask keypoints
                       with format [x1, y1, x2, y2, x3, y3, x4, y4]
            mask: Beam mask, with shape (1, h, w)
            probe: Probe type of the image

        Returns:
            augmented image, label, keypoints, mask, probe type
        """

        # Sample random brightness and contrast change
        contrast_adjust = self.min_contrast + torch.rand(()) * (self.max_contrast - self.min_contrast)
        brightness_adjust = self.min_brightness + torch.rand(()) * (self.max_brightness - self.min_brightness)

        # Apply brightness and contrast changes
        new_image = tvf.adjust_brightness(image, brightness_adjust)
        new_image = tvf.adjust_contrast(new_image, contrast_adjust)

        # Mask and return image
        new_image = new_image * mask
        return new_image, label, keypoints, mask, probe

