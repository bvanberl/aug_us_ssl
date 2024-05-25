import random
import math

import torch
from torch import nn, Tensor
from torchvision.transforms.v2 import functional as tvf
from torchvision.transforms import InterpolationMode

from src.constants import Probe
from src.augmentations.aug_utils import *


class DepthChange(nn.Module):
    """Transformation that simulates an increase or decrease
    in ultrasound image depth
    """

    def __init__(
            self,
            min_depth_factor: float = 0.8,
            max_depth_factor: float = 1.2,
            preserve_beam_shape: bool = False
    ):
        """Initializes the DepthChange noise layer.

        Args:
            min_depth_factor: Minimum depth change factor
            max_depth_factor: Maximum depth change factor
            preserve_beam_shape: If True, preserves mask shape no matter what the
                depth transform is. With a depth decrease, this would crop the
                bottom of the clip.
        """
        super(DepthChange, self).__init__()
        self.min_depth_factor = min_depth_factor
        self.max_depth_factor = max_depth_factor
        self.preserve_beam_shape = preserve_beam_shape

    def forward(
            self,
            image: Tensor,
            label: Tensor,
            keypoints: Tensor,
            mask: Tensor,
            probe: Tensor,
            **kwargs
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """Simulates a change in image depth

        Samples a depth change  based on the minimum and maximum depth
        factors and zooms in or out with respect to the location
        of the ultrasound probe. A depth change < 1 or > 1 are simulated
        by zooming in and out respectively.

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

        # Determine the scale factor for the image
        scale = self.min_depth_factor + torch.rand(()) * (self.max_depth_factor - self.min_depth_factor)

        # Determine translation that would keep the origin of the beam stationary
        _, h, w = image.shape
        if probe == Probe.LINEAR.value:
            translate_x = 0.
            translate_y = (scale - 1.) * h / 2
        else:
            x_itn, y_itn = get_point_of_intersection(*keypoints)
            translate_x = (scale - 1) * (w / 2 - x_itn)
            translate_y = (scale - 1) * (h / 2 - keypoints[1])

        # Perform affine transform that scales and translates image
        new_image = tvf.affine(image, 0., [translate_x, translate_y], scale, 0., InterpolationMode.BILINEAR)

        if self.preserve_beam_shape:
            # Ensure that beam shape doesn't change
            if probe == Probe.LINEAR.value and scale < 1.:
                # Resize horizontally to fit within mask
                x_diff = ((1 - scale) * w / 2).int()
                new_image = new_image[:, :, x_diff: w - x_diff]
                new_image = tvf.resize_image(new_image, [h, w])
            new_mask = mask
        else:
            new_mask = tvf.affine(mask, 0., [translate_x, translate_y], scale, 0., InterpolationMode.BILINEAR)
        new_image = (new_mask * new_image).to(torch.uint8)
        return new_image, label, keypoints, new_mask, probe


class ShiftAndRotate(nn.Module):
    """Transformation that randomly shifts and rotates an
    ultrasound beam.
    """

    def __init__(
            self,
            max_shift: float = 0.15,
            max_rotation: float = 22.5
    ):
        """Initializes the ShiftAndRotate layer.

        Args:
            max_shift: Maximum vertical/horizontal shift
                              (as a fraction of image width)
            max_rotation: Maximum rotation (degrees)
        """
        super(ShiftAndRotate, self).__init__()
        self.max_shift = max_shift
        self.max_rotation = max_rotation

    def forward(
            self,
            image: Tensor,
            label: Tensor,
            keypoints: Tensor,
            mask: Tensor,
            probe: Tensor,
            **kwargs
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """Applies a random shift and rotation to the image

        Samples a random translation and rotation angle, then
        applies the transformation

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

        _, h, w = image.shape
        device = keypoints.get_device()
        device = 'cpu' if device == -1 else device

        # Determine the translation for the image
        translate_x = w * random.uniform(-self.max_shift, self.max_shift)
        translate_y = h * random.uniform(-self.max_shift, self.max_shift)

        # Determine the rotation angle for the image
        angle = random.uniform(-self.max_rotation, self.max_rotation)

        # Perform affine transform that scales and translates image
        new_image = tvf.affine(image, angle, [translate_x, translate_y], 1., 0., InterpolationMode.BILINEAR)
        new_mask = tvf.affine(mask, angle, [translate_x, translate_y], 1., 0., InterpolationMode.BILINEAR)

        # Get coordinates of new keypoints
        sin_angle = math.sin(angle)
        cos_angle = math.cos(angle)
        transform_matrix = torch.tensor([
            [cos_angle, -sin_angle, translate_x],
            [sin_angle, cos_angle, translate_y],
            [0., 0., 1.]
        ], device=device)
        homogenous_kp = torch.concat([
            keypoints.reshape(4, 2),
            torch.ones((4, 1), device=device)
        ], dim=-1)
        new_homogenous_kp = torch.matmul(transform_matrix, homogenous_kp.T)
        new_keypoints = new_homogenous_kp[:, :2].flatten()

        return new_image, label, new_keypoints, new_mask, probe


class HorizontalReflection(nn.Module):
    """Transformation that randomly reflects an ultrasound
    image hozizontally.
    """

    def __init__(
            self,
    ):
        """Initializes the ShiftAndRotate layer.

        Args:
            p_reflect: Probability of horizontal reflection
        """
        super(HorizontalReflection, self).__init__()

    def forward(
            self,
            image: Tensor,
            label: Tensor,
            keypoints: Tensor,
            mask: Tensor,
            probe: Tensor,
            **kwargs
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """Horizontally reflects the input image

        Reflects the image and updates the keypoints

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

        device = keypoints.get_device()
        device = 'cpu' if device == -1 else device

        # Perform transform that reflects image horizontally
        new_image = tvf.horizontal_flip(image)
        new_mask = tvf.horizontal_flip(mask)

        # Swap x1 <--> x2 and x3 <--> x4
        new_keypoints = torch.tensor([
            keypoints[2],
            keypoints[1],
            keypoints[0],
            keypoints[3],
            keypoints[6],
            keypoints[5],
            keypoints[4],
            keypoints[7]
        ], device=device)

        return new_image, label, new_keypoints, new_mask, probe