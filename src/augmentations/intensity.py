import random
from typing import Tuple

import torch
from torch import nn, Tensor
from torchvision.transforms.v2 import functional as tvf
from kornia.enhance import equalize_clahe


class CLAHETransform(nn.Module):
    """
    This transformation applies contrast-limited adaptive histogram
    equalization to the input.
    """

    def __init__(
            self,
            min_clip_limit: float = 30.0,
            max_clip_limit: float = 50.0,
            tile_grid_size: Tuple[int, int] = (8, 8)
    ):
        """
        Initializes the CLAHETransform class
        Args:
            min_clip_limit: Minimum value for histogram clip limit
            max_clip_limit: Maximum value for histogram clip limit
            tile_grid_size: Shape of tiles into which image is divided
        """
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

        Samples a random value for the histogram clip limit and applies
        CLAHE to the image.
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

        clip_limit = random.uniform(self.min_clip_limit, self.max_clip_limit)
        new_image = equalize_clahe(image / 255., clip_limit, self.tile_grid_size)
        new_image = new_image * 255
        new_image = (new_image * mask).to(torch.uint8)
        return new_image, label, keypoints, mask, probe


class BrightnessContrastChange(nn.Module):
    """
    This transformation applies random brightness and contrast changes.
    """

    def __init__(
            self,
            min_brightness: float = 1.,
            max_brightness: float = 1.,
            min_contrast: float = 1.,
            max_contrast: float = 1.
    ):
        """
        Initializes the BrightnessContrastChange class
        Args:
            min_brightness: Minimum brightness change
            max_brightness: Maximum brightness change
            min_contrast: Minimum contrast change
            max_contrast: Maximum contrast change
        """

        assert 0. <= min_brightness <= max_brightness <= 2., "Brightness range is [0, 2]"
        assert 0. <= min_contrast <= max_contrast <= 2., "Contrast range is [0, 2]"

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
        """Applies random brightness and contrast change to image

        Samples random brightness and contrast change magnitudes and
        applies them to the image.
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
        contrast_adjust = random.uniform(self.min_contrast, self.max_contrast)
        brightness_adjust = random.uniform(self.min_brightness, self.max_brightness)

        # Apply brightness and contrast changes
        new_image = tvf.adjust_brightness(image, brightness_adjust)
        new_image = tvf.adjust_contrast(new_image, contrast_adjust)

        # Mask and return image
        new_image = (new_image * mask).to(torch.uint8)
        return new_image, label, keypoints, mask, probe


class GammaCorrection(nn.Module):
    """
    This transformation applies random gamma correction.
    """

    def __init__(
            self,
            min_gamma: float = 0.5,
            max_gamma: float = 1.75,
            gain: float = 1.
    ):
        """
        Initializes the GammaCorrection class
        Args:
            min_gamma: Minimum gamma
            max_gamma: Maximum gamma
            gain: Brightness constant
        """

        assert 0. <= min_gamma <= max_gamma, "Gamma must be non-negative"
        assert 0 < gain, "Gain must be positive"

        super(GammaCorrection, self).__init__()
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.gain = gain
    def forward(
            self,
            image: Tensor,
            label: Tensor,
            keypoints: Tensor,
            mask: Tensor,
            probe: Tensor,
            **kwargs
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """Applies gamma correction to input image

        Samples random brightness and contrast change magnitudes and
        applies them to the image.

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

        # Sample random gamma
        gamma = random.uniform(self.min_gamma, self.max_gamma)

        # Apply brightness and contrast changes
        new_image = tvf.adjust_gamma(image, gamma, self.gain)

        # Mask and return image
        new_image = (new_image * mask).to(torch.uint8)
        return new_image, label, keypoints, mask, probe
