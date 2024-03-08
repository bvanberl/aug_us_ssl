from typing import Tuple

import torch
from torch import nn, Tensor
from kornia.enhance import equalize_clahe
import numpy as np


class CLAHETransform(nn.Module):
    def __init__(
            self,
            min_clip_limit: float = 30.0,
            max_clip_limit: float = 50.0,
            tile_grid_size: Tuple[int, int] = (8, 8)
    ):
        super().__init__()
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
