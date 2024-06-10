import random

import numpy as np
import torch
from torch import nn, Tensor
from torchvision.transforms.v2 import functional as tvf, Transform
from torchvision.transforms import InterpolationMode as im

from src.constants import BeamKeypoints, Probe
from src.augmentations.aug_utils import *
from src.data.utils import get_beam_mask

class ConvexityMutation(nn.Module):
    """Preprocessing layer that warps convex ultrasound images.

    This transformation distorts ultrasound images such that
    the angle between the left and right boundaries is mutated, mimicking
    a changed field of view. The change in angle is randomly sampled.
    A flow field is computed that translates beam pixels towards or away
    from the center depending on if the angle is increased or decreased
    respectively.
    """

    def __init__(
            self,
            square_roi: bool = False,
            min_top_width: float = 0.,
            max_top_width: float = 0.75,
            point_thresh: float = 1.,
    ):
        """Initializes the NonLinearToLinear layer.

        Args:
            square_roi: If True, images are minimal crops surrounding the beam
                that have been resized to square
            min_width_frac: The minimum possible width of the output image's
                beam width, as a fraction of original beam's width
            max_width_frac: The maximum possible width of the output image's
                beam width, as a fraction of original beam's width
            point_thresh: The maximum value for `|x2-x1|` that constitutes a
                phased clip with a pointed top
        """
        super(ConvexityMutation, self).__init__()
        self.square_roi = square_roi
        self.min_top_width = min_top_width
        self.max_top_width = max_top_width
        self.point_thresh = point_thresh

    def forward(
            self,
            image: Tensor,
            label: Tensor,
            keypoints: Tensor,
            mask: Tensor,
            probe: Tensor,
            **kwargs
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """Applies the probe type transformation to the image

        Determines a coordinate map that routes pixel locations
        in the original image to the new, distorted image, depending
        on the original probe type. Constructs the distorted beam
        and draws it onto the original image in place of the old beam.
        Applies bilinear interpolation to the new image.

        Args:
            image: 3D Image tensor, with shape (c, h, w)
            label: Label for the image, which is unaltered
            keypoints: 1D tensor with shape (8,) containing beam mask keypoints
                       with format [x1, y1, x2, y2, x3, y3, x4, y4]
            mask: Beam mask, with shape (1, h, w)
            probe: Probe type of the image
            orig_h: Original height of the image
            orig_w: Original width of the image

        Returns:
            augmented image, transformed keypoints, new probe type (linear), height, width
        """

        if probe == Probe.LINEAR.value:

            # Passthrough - linear probes are not convex
            return image, label, keypoints, mask, probe

        x1, y1, x2, y2, x3, y3, x4, y4 = keypoints
        c, h, w = image.shape
        device = image.get_device()
        device = 'cpu' if device == -1 else device

        # If beam has point at top, marginally move top keypoints laterally
        if torch.abs(x2 - x1) < self.point_thresh:
            x1 -= 0.5
            x2 += 0.5

        # Determine point and angle of intersection of lateral beam bounds
        x_itn, y_itn = get_point_of_intersection(*keypoints)
        theta = get_angle_of_intersection(x3, y3, x4, y4, x_itn, y_itn)

        # Randomly sample the width fraction of the original beam
        top_width = random.uniform(self.min_top_width, self.max_top_width)
        top_width_scale = top_width / (x2 - x1) * (x4 - x3)
        new_x1 = x_itn - (x_itn - x1) * top_width_scale
        new_x2 = x_itn + (x2 - x_itn) * top_width_scale

        y_coords = torch.linspace(0, h - 1, h).to(device)
        x_coords = torch.linspace(0, w - 1, w).to(device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Determine point and angle of intersection of new lateral beam bounds
        new_x_itn, new_y_itn = get_point_of_intersection(new_x1, y1, new_x2, y2, x3, y3, x4, y4)
        new_theta = get_angle_of_intersection(x3, y3, x4, y4, new_x_itn, new_y_itn)

        # Calculate polar coordinates with respect to new beam shape
        new_xx_c = xx - new_x_itn
        new_yy_c = yy - new_y_itn
        new_phi = torch.atan2(new_xx_c, new_yy_c)
        dists = torch.sqrt(new_xx_c**2 + new_yy_c**2)

        new_r = torch.sqrt((x3 - new_x_itn)**2 + (y3 - new_y_itn)**2)
        old_r = torch.sqrt((x3 - x_itn)**2 + (y3 - y_itn)**2)

        # Determine angular and radius ratios for old:new beam
        theta_ratio = theta / new_theta
        old_top_r = torch.sqrt((x_itn - x1)**2 + (y_itn - y1)**2)
        new_top_r = torch.sqrt((new_x_itn - new_x1) ** 2 + (new_y_itn - y1) ** 2)
        rad_ratio = (old_r - old_top_r) / (new_r - new_top_r)

        # Compute mapping from new image to old image coordinates
        rad_scaling = ((dists - new_top_r) * rad_ratio + old_top_r)
        new_xx = x_itn + rad_scaling * torch.sin(theta_ratio * new_phi)
        new_yy = y_itn + rad_scaling * torch.cos(theta_ratio * new_phi)

        # If square ROI, beam was not initially circular and bottom of beam coincided with
        # bottom of image. Adjust so that bottom of beam coincides with bottom of image.
        if self.square_roi:
            vertical_adjust = (h - 1) / new_yy[-1, x_itn.int()]
            new_yy = new_yy * vertical_adjust
            new_y3 = torch.min(y3 / vertical_adjust, torch.tensor(h - 1.))
            new_y4 = torch.min(y4 / vertical_adjust, torch.tensor(h - 1.))
            horiz_adjust = w / (new_xx[new_y3.int(), w - 1] - new_xx[new_y3.int(), new_x_itn.int()]) / 2.
            new_xx = (new_xx - new_x_itn) * horiz_adjust + new_x_itn
        else:
            new_y3 = y3
            new_y4 = y4

        # Normalize the coordinate map
        new_yy = new_yy / h * 2. - 1.
        new_xx = new_xx / w * 2. - 1.
        adjusted_grid = torch.stack([new_xx, new_yy], dim=-1)

        # Determine new keypoints
        new_keypoints = np.array([
            new_x1,
            y1,
            new_x2,
            y2,
            x3,
            new_y3,
            x4,
            new_y4
        ])

        # Construct new image using values from old image
        new_image = nn.functional.grid_sample(image.unsqueeze(0).float(), adjusted_grid.unsqueeze(0), align_corners=False).squeeze(0)
        new_mask = tvf.pil_to_tensor(get_beam_mask(h, w, new_keypoints, probe, self.square_roi, 'L'))
        new_image = (new_image * new_mask).to(torch.uint8)
        new_keypoints = torch.from_numpy(new_keypoints)


        return new_image, label, new_keypoints, new_mask, probe

