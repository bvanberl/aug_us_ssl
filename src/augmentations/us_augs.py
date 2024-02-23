import torch
from torch import nn
from torchvision.transforms import functional as tvf

from src.constants import BeamKeypoints, ProbeType
from src.augmentations.aug_utils import *

class NonlinearToLinear(nn.Module):
    """Preprocessing layer that converts non-linear beam shapes to linear.

    This augmentation transformation distorts non-linear beam shapes
    (i.e., curvilinear or phased array) to a linear beam shape. In other
    words, it transforms the image such that it appears like it was acquired
    using a linear probe. If the image already contained a linear beam, does
    the image is not distorted.
    """

    def __init__(
            self,
            square_roi: bool = False,
            min_width_frac: float = 0.4,
            max_width_frac: float = 0.6
    ):
        """Initializes the NonLinearToLinear layer.

        Args:
            square_roi: If True, images are minimal crops surrounding the beam
                that have been resized to square
            min_width_frac: The minimum possible width of the output image's
                beam width, as a fraction of original beam's width
            max_width_frac: The maximum possible width of the output image's
                beam width, as a fraction of original beam's width
        """
        super(NonlinearToLinear).__init__()
        self.min_width_frac = min_width_frac
        self.max_width_frac = max_width_frac
        self.square_roi = square_roi

    def forward(
            self,
            image: torch.Tensor,
            keypoints: torch.Tensor,
            probe_type: torch.Tensor,
            **kwargs
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Applies the probe type transformation to the image

        Determines a coordinate map that routes pixel locations
        in the original image to the new, distorted image, depending
        on the original probe type. Constructs the distorted beam
        and draws it onto the original image in place of the old beam.
        Applies bilinear interpolation to the new image.

        Args:
            image: 3D Image tensor, with shape (c, h, w)
            keypoints: 1D tensor with shape (8,) containing beam mask keypoints
                       with format [x1, y1, x2, y2, x3, y3, x4, y4]
            probe_type: Probe type of the image

        Returns:
            augmented image, transformed keypoints, new probe type (linear)
        """

        x1, y1, x2, y2, x3, y3, x4, y4 = keypoints
        c, h, w = image.shape

        x_itn, y_itn = get_point_of_intersection(*keypoints)

        if self.square_roi:
            width_frac = 1.
            bot_r = h - y_itn
        else:
            # Randomly sample the width fraction of the original beam
            width_frac = self.min_width_frac + \
                         torch.rand(()) * (self.max_width_frac - self.min_width_frac)
            bot_r = torch.sqrt((x3 - x_itn) ** 2 + (y3 - y_itn) ** 2)

        # Calculate new keypoints
        new_bottom = y_itn + bot_r
        new_keypoints = torch.stack([
            x3, y1, x4, y2, x3, new_bottom, x4, new_bottom
        ])

        if self.square_roi:
            resize_w = torch.floor(2 * torch.sqrt((h.float() - y_itn) ** 2 - (y3 - y_itn) ** 2))
            w = resize_w
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
        y_coords = torch.linspace(0, h - 1, h)
        x_coords = torch.linspace(0, w - 1, w)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        phi = theta * ((x_coords - x3) / beam_w - 0.5)  # Angles by x coordinates
        norm_yy = (y_coords - y1) / beam_h  # Normalize y coordinates

        # Calculate coordinate map based on original beam type
        if probe_type == ProbeType.CURVED_LINEAR.value:
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

        if self.square_roi:
            image = tvf.resize(image, [h.int(), w.int()])

        # Construct new image using values from old image
        flow_field = torch.stack([new_yy, new_xx], dim=-1)
        mapped_image = nn.functional.grid_sample(image, flow_field)

        # Update beam pixels in original image with distorted linear beam
        mask = (
            (yy >= top_bound)
            & (yy <= bottom_bound)
            & (xx >= new_left_bound)
            & (xx <= new_right_bound)
        ).float().unsqueeze(-1)
        mapped_image = mask * mapped_image

        # Set pixels within original beam ROI but outside new ROI to black
        if not self.square_roi:
            orig_mask = (
                    (yy >= top_bound)
                    & (yy <= bottom_bound)
                    & (xx >= left_bound)
                    & (xx <= right_bound)
            ).unsqueeze(-1)
            orig_mask = torch.tile(orig_mask, [1, 1, c])
            new_image = torch.where(orig_mask, mapped_image, image)
            new_image = tvf.resize(new_image, [h.int(), w.int()])
            left_bound = left_bound
        else:
            new_image = mapped_image

        return new_image, new_keypoints, ProbeType.LINEAR.value











