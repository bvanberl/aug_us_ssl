from typing import Any, Dict, Optional, Sequence, Tuple, Union
import warnings
import math

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms.v2 import functional as F, InterpolationMode
from torchvision.transforms.v2._utils import _setup_size


class ResizeKeypoint(nn.Module):
    """Resizes image to desired size.

    This is largely the same as torch's Resize, but its format
    is suited to the types of inputs in the AugUS augmentation pipeline.
    Modifies keypoints such that they are correct for the cropped image.
    """

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
        self.interpolation = interpolation
        self.antialias = antialias

    def _get_params(self, image: torch.Tensor) -> Dict[str, Any]:
        height = image.shape[1]
        width = image.shape[2]
        area = height * width

        log_ratio = self._log_ratio
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(
                    log_ratio[0],  # type: ignore[arg-type]
                    log_ratio[1],  # type: ignore[arg-type]
                )
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                break
        else:
            # Fallback to central crop
            in_ratio = float(width) / float(height)
            if in_ratio < min(self.ratio):
                w = width
                h = int(round(w / min(self.ratio)))
            elif in_ratio > max(self.ratio):
                h = height
                w = int(round(h * max(self.ratio)))
            else:  # whole image
                w = width
                h = height
            i = (height - h) // 2
            j = (width - w) // 2

        return dict(top=i, left=j, height=h, width=w)

    def forward(
            self,
            image: Tensor,
            label: Tensor,
            keypoints: Tensor,
            mask: Tensor,
            probe: Tensor,
            **kwargs
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """Applies the random resized crop transformation

        Takes a random crop from the image and resizes it to a fixed
        dimension. Same as torch's RandomResizedCrop, but augments
        the beam keypoints as well.

        Args:
            image: 3D Image tensor, with shape (c, h, w)
            label: Label for the image, which is unaltered
            keypoints: 1D tensor with shape (8,) containing beam mask keypoints
                       with format [x1, y1, x2, y2, x3, y3, x4, y4]
            mask: Beam mask, with shape (1, h, w)
            probe: Probe type of the image

        Returns:
            augmented image, label, transformed keypoints, transformed mask, probe type
        """

        # Get cropped and resized image and masks
        new_image = F.resize(image, size=self.size, interpolation=self.interpolation, antialias=self.antialias)
        new_mask = F.resize(mask, size=self.size, interpolation=self.interpolation, antialias=self.antialias)

        # Transform the keypoints to match the crop
        h1 = image.shape[1]
        w1 = image.shape[2]
        h2 = self.size[0]
        w2 = self.size[1]
        resize_h_scale = h2 / h1
        resize_w_scale = w2 / w1

        x1 = keypoints[0] * resize_w_scale
        y1 = keypoints[1] * resize_h_scale
        x2 = keypoints[2] * resize_w_scale
        y2 = keypoints[3] * resize_h_scale
        x3 = keypoints[4] * resize_w_scale
        y3 = keypoints[5] * resize_h_scale
        x4 = keypoints[6] * resize_w_scale
        y4 = keypoints[7] * resize_h_scale
        new_keypoints = torch.stack([x1, y1, x2, y2, x3, y3, x4, y4])

        return new_image, label, new_keypoints, new_mask, probe


class RandomResizedCropKeypoint(nn.Module):
    """Crops a random portion of the input and resize it to a given size.

    This is largely the same as torch's RandomResizedCrop, but its format
    is suited to the types of inputs in the AugUS augmentation pipeline.
    Modifies keypoints such that they are correct for the cropped image.

    Sourced mainly from https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
    Refer to torchvision docs for RandomResizedCrop for more details.
    """

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.antialias = antialias

        self._log_ratio = torch.log(torch.tensor(self.ratio))

    def _get_params(self, image: torch.Tensor) -> Dict[str, Any]:
        height = image.shape[1]
        width = image.shape[2]
        area = height * width

        log_ratio = self._log_ratio
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(
                    log_ratio[0],  # type: ignore[arg-type]
                    log_ratio[1],  # type: ignore[arg-type]
                )
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                break
        else:
            # Fallback to central crop
            in_ratio = float(width) / float(height)
            if in_ratio < min(self.ratio):
                w = width
                h = int(round(w / min(self.ratio)))
            elif in_ratio > max(self.ratio):
                h = height
                w = int(round(h * max(self.ratio)))
            else:  # whole image
                w = width
                h = height
            i = (height - h) // 2
            j = (width - w) // 2

        return dict(top=i, left=j, height=h, width=w)

    def forward(
            self,
            image: Tensor,
            label: Tensor,
            keypoints: Tensor,
            mask: Tensor,
            probe: Tensor,
            **kwargs
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """Applies the random resized crop transformation

        Takes a random crop from the image and resizes it to a fixed
        dimension. Same as torch's RandomResizedCrop, but augments
        the beam keypoints as well.

        Args:
            image: 3D Image tensor, with shape (c, h, w)
            label: Label for the image, which is unaltered
            keypoints: 1D tensor with shape (8,) containing beam mask keypoints
                       with format [x1, y1, x2, y2, x3, y3, x4, y4]
            mask: Beam mask, with shape (1, h, w)
            probe: Probe type of the image

        Returns:
            augmented image, label, transformed keypoints, transformed mask, probe type
        """

        # Get random parameters for the transform
        params = self._get_params(image)

        # Get cropped and resized image and masks
        new_image = F.resized_crop(image, **params, size=self.size,
                                   interpolation=self.interpolation, antialias=self.antialias
                                   )
        new_mask = F.resized_crop(mask, **params, size=self.size,
                                   interpolation=self.interpolation, antialias=self.antialias
                                   )

        # Transform the keypoints to match the crop
        h1 = image.shape[1]
        w1 = image.shape[2]
        h2 = self.size[0]
        w2 = self.size[1]

        resize_h_scale = h2 / params['height']
        resize_w_scale = w2 / params['width']

        x1 = (keypoints[0] - params['left']) * resize_w_scale
        y1 = (keypoints[1] - params['top']) * resize_h_scale
        x2 = (keypoints[2] - params['left']) * resize_w_scale
        y2 = (keypoints[3] - params['top']) * resize_h_scale
        x3 = (keypoints[4] - params['left']) * resize_w_scale
        y3 = (keypoints[5] - params['top']) * resize_h_scale
        x4 = (keypoints[6] - params['left']) * resize_w_scale
        y4 = (keypoints[7] - params['top']) * resize_h_scale
        new_keypoints = torch.stack([x1, y1, x2, y2, x3, y3, x4, y4])

        return new_image, label, new_keypoints, new_mask, probe
