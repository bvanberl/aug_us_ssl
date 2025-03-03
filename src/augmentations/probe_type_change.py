from typing import Optional

import torch
from torch import nn, Tensor
from torchvision.transforms.v2 import functional as tvf, Transform
from torchvision.transforms import InterpolationMode as im

from src.constants import BeamKeypoints, Probe
from src.augmentations.aug_utils import *

class ProbeTypeChange(nn.Module):
    """Preprocessing layer that converts the probe type of the beam.

    This augmentation transformation distorts non-linear beam shapes
    (i.e., curved linear or phased array) to a linear beam shape, or linear
    beam shapes to curved linear beam shapes. In other words, it transforms
    the image such that it appears like it was acquired from a different
    type of ultrasound probe.
    """

    def __init__(
            self,
            square_roi: bool = False,
            min_linear_width_frac: float = 0.4,
            max_linear_width_frac: float = 0.6,
            min_convex_rad_factor: float = 1.0,
            max_convex_rad_factor: float = 2.0,
            pass_through: Optional[str] = None
    ):
        """Initializes the NonLinearToLinear layer.

        Args:
            square_roi: If True, images are minimal crops surrounding the beam
                that have been resized to square
            min_linear_width_frac: The minimum possible width of a linear output image's
                beam width, as a fraction of original beam's width
            max_linear_width_frac: The maximum possible width of a linear output image's
                beam width, as a fraction of original beam's width
            min_convex_rad_factor: The minimum possible radius for a convex beam as a multiple
                of the height of a linear beam shape
            max_convex_rad_factor: The maximum possible radius for a convex beam as a multiple
                of the height of a linear beam shape
            pass_through: Probe geometry that will be left unaltered when encountered. One of
                `linear` or `convex`

        """

        assert 0. < min_linear_width_frac <= max_linear_width_frac <= 1., "Min and max linear frac must be in [0, 1]"
        assert 1. <= min_convex_rad_factor <= max_convex_rad_factor, "Convex rad factors must be >= 1"
        if pass_through is not None:
            assert pass_through.lower() in ['linear', 'convex'], "`pass_through` must be 'linear' or 'convex'"

        super(ProbeTypeChange, self).__init__()
        self.square_roi = square_roi
        self.min_linear_width_frac = min_linear_width_frac
        self.max_linear_width_frac = max_linear_width_frac
        self.min_convex_rad_factor = min_convex_rad_factor
        self.max_convex_rad_factor = max_convex_rad_factor
        self.pass_through = pass_through

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

        Returns:
            augmented image, label, transformed keypoints, new probe type (linear)
        """

        x1, y1, x2, y2, x3, y3, x4, y4 = keypoints
        c, h, w = image.shape
        device = image.get_device()
        device = 'cpu' if device == -1 else device

        if probe == Probe.LINEAR.value:

            if self.pass_through == 'linear':
                return image, label, keypoints, mask, Probe.LINEAR.value

            # Sample random radius for curved linear beam
            rad_factor = self.min_convex_rad_factor + \
                         torch.rand(()) * (self.max_convex_rad_factor - self.min_convex_rad_factor)
            bot_r = (y3 - y1) * rad_factor

            # Calculate new keypoints
            x_itn = (x4 - x3) / 2.
            y_itn = y3 - bot_r
            new_y3 = y_itn + torch.sqrt(bot_r ** 2 - (x_itn - x1) ** 2)
            new_y4 = new_y3
            new_x1 = x_itn - (y1 - y_itn) * (x_itn - x3) / (new_y3 - y_itn)
            new_x2 = 2 * x_itn - new_x1
            top_r = torch.sqrt((x_itn - new_x1) ** 2 + (y1 - y_itn) ** 2)
            new_keypoints = torch.stack([new_x1, y1, new_x2, y2, x3, new_y3, x4, new_y4])

            # Obtain flow field mapping points in new curved linear image to old linear image
            y_coords = torch.linspace(0., h - 1., h, device=device)
            x_coords = torch.linspace(0., w - 1, w, device=device)
            new_yy, new_xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            out_xx = torch.atan2(new_xx - x_itn, new_yy - y_itn)
            x_squeeze_factor = 1. / torch.abs(out_xx[new_y3.int(), 0])
            out_xx = out_xx * x_squeeze_factor
            out_yy = (torch.sqrt((x_itn - new_xx) ** 2 + (y_itn - new_yy) ** 2) - top_r) / (bot_r - top_r)
            out_yy = out_yy * 2. - 1.

            # Warp the image and mask using the flow field
            flow_field = torch.stack([out_xx, out_yy], dim=-1)
            new_image = nn.functional.grid_sample(image.unsqueeze(0).float(), flow_field.unsqueeze(0), align_corners=False).squeeze(0)
            new_mask = nn.functional.grid_sample(mask.unsqueeze(0).float(), flow_field.unsqueeze(0), align_corners=False).squeeze(0)
            new_image = new_image.to(torch.uint8)

            return new_image, label, new_keypoints, new_mask, Probe.CURVED_LINEAR.value

        else:

            if self.pass_through == 'convex':
                return image, label, keypoints, mask, probe

            # Get point of intersection of left and right beam bounds
            x_itn, y_itn = get_point_of_intersection(*keypoints)

            # Randomly sample the width fraction of the original beam
            width_frac = self.min_linear_width_frac + \
                         torch.rand(()) * (self.max_linear_width_frac - self.min_linear_width_frac)

            # Get radius of beam
            if self.square_roi:
                # If keypoints are off-screen, shift and scale image so that they
                # are on-screen and comprise the left and right bounds.
                if x3 < 0. or x4 > w:
                    image = torch.concat([
                        torch.zeros((c, h, max(torch.abs(x3).int(), 0)), device=device),
                        image,
                        torch.zeros((c, h, max(torch.abs(x4.int() - w), 0)), device=device),
                    ], 2)
                    image = tvf.resize(image, [h, w])
                    w_delta = w / (x4 - x3)
                    x1 = (x1 - x3) * w_delta
                    x_itn = (x_itn - x3) * w_delta
                    x3 = 0.
                    x4 = w - 1.
                bot_r = h - y_itn
            else:
                bot_r = torch.sqrt((x3 - x_itn) ** 2 + (y3 - y_itn) ** 2)
            new_left = x_itn - width_frac * w / 2.
            new_right = x_itn + width_frac * w / 2.

            # Calculate new keypoints
            new_bottom = y_itn + bot_r

            new_keypoints = torch.stack([
                new_left,
                y1,
                new_right,
                y2,
                new_left,
                new_bottom,
                new_right,
                new_bottom
            ])

            if self.square_roi:
                resize_w = torch.floor(2 * torch.sqrt((h - y_itn) ** 2 - (y3 - y_itn) ** 2))
                w = resize_w.int()
                x1 = x1 * w / h
                x3 = x3 * w / h
                x4 = x4 * w / h
                x_itn = x_itn * w / h
                beam_h = h
            else:
                beam_h = y_itn + bot_r - y1  # Beam ROI height
            beam_w = x4 - x3  # Beam ROI width

            # Determine point and angle of intersection of lateral beam bounds
            theta = get_angle_of_intersection(x3, y3, x4, y4, x_itn, y_itn)

            # Calculate radius of the beam's circular sector
            y_coords = torch.linspace(0, h - 1, h, device=device)
            x_coords = torch.linspace(0, w - 1, w, device=device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

            phi = theta * ((x_coords - x3) / beam_w - 0.5)  # Angles by x coordinates
            norm_yy = (yy - y1) / beam_h  # Normalize y coordinates

            # Calculate coordinate map based on original beam type
            if probe == Probe.CURVED_LINEAR.value:
                # Length of radial line from intersection to (x1, y1)
                top_r = torch.sqrt((x1 - x_itn) ** 2 + (y1 - y_itn) ** 2)

                # Coordinate map for curved linear --> linear
                new_xx = x_itn + torch.sin(phi / width_frac) * (
                    top_r + norm_yy * (bot_r - top_r)
                )
                new_yy = y_itn + torch.cos(phi / width_frac) * (
                    top_r + norm_yy * (bot_r - top_r)
                )
            else:
                # Get lengths of all radial lines from intersection to flat top
                itn_to_top = y1 - y_itn  # Vertical distance from intersection to top of beam
                cos_phi = torch.cos(phi / width_frac)
                top_r = itn_to_top / cos_phi

                # Coordinate map for phased array --> linear
                new_xx = x_itn + (torch.sin(phi / width_frac)) * (
                    top_r + norm_yy * (bot_r - top_r)
                )
                new_yy = y_itn + cos_phi * (
                    top_r + norm_yy * (bot_r - top_r)
                )

            # Calculate bounds of the new linear beam
            half_width = (x_itn - x3) * width_frac
            new_left_bound = x_itn - half_width
            new_right_bound = x_itn + half_width
            bottom_bound = y_itn + bot_r
            top_bound = y1
            left_bound = x3
            right_bound = x4

            # Stretch horizontally so that beam is a circular sector
            if self.square_roi:
                image = tvf.resize(image, [h, w])

            # Construct new image using values from old image
            new_yy = new_yy / h * 2. - 1.
            new_xx = new_xx / w * 2. - 1.
            flow_field = torch.stack([new_xx, new_yy], dim=-1)
            mapped_image = nn.functional.grid_sample(image.unsqueeze(0).float(), flow_field.unsqueeze(0), align_corners=False).squeeze(0)

            # Update beam pixels in original image with distorted linear beam
            new_mask = (
                (yy >= top_bound)
                & (yy <= bottom_bound)
                & (xx >= new_left_bound)
                & (xx <= new_right_bound)
            ).unsqueeze(0)
            mapped_image = new_mask * mapped_image

            # Set pixels within original beam ROI but outside new ROI to black
            if not self.square_roi:
                new_image = overlay_region_on_image(
                    image,
                    mapped_image,
                    xx,
                    yy,
                    top_bound,
                    bottom_bound,
                    left_bound,
                    right_bound
                )
            else:
                new_image = tvf.resize(mapped_image, [h, h])
                new_mask = tvf.resize(new_mask, [h, h])
            new_image = new_image.to(torch.uint8)

            return new_image, label, new_keypoints, new_mask, Probe.LINEAR.value











