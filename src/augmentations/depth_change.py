import torch
from torch import nn, Tensor
from torchvision.transforms.v2 import functional as tvf

from src.constants import Probe
from src.augmentations.aug_utils import *


class DepthChange(nn.Module):
    """Transformation that simulates an increase or decrease
    in ultrasound image depth
    """

    def __init__(
            self,
            min_depth_factor: float = 0.8,
            max_depth_factor: float = 1.2
    ):
        """Initializes the DepthChange noise layer.

        Args:
            min_depth_factor: Minimum depth change factor
            max_depth_factor: Maximum depth change factor
        """
        super(DepthChange, self).__init__()
        self.min_depth_factor = min_depth_factor
        self.max_depth_factor = max_depth_factor

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
        new_image = tvf.affine(image, 0., [translate_x, translate_y], scale, 0.)

        if probe == Probe.LINEAR.value and scale < 1.:
            x_diff = ((1 - scale) * w / 2).int()
            new_image = new_image[:, :, x_diff: w - x_diff]
            new_image = tvf.resize_image(new_image, [h, w])
        else:
            new_image = mask * new_image
        return new_image, label, keypoints, mask, probe
