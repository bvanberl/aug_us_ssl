import torch
from torch import nn, Tensor
from torchvision.transforms.v2 import functional as tvf, Transform
from torchvision.transforms import InterpolationMode as im

from src.constants import BeamKeypoints, Probe
from src.augmentations.aug_utils import *

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
            max_top_width: float = 0.5,
            point_thresh: float = 1.
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

        Returns:
            augmented image, transformed keypoints, new probe type (linear)
        """

        if probe == Probe.LINEAR.value:
            return image, label, keypoints, mask, probe

        x1, y1, x2, y2, x3, y3, x4, y4 = keypoints
        c, h, w = image.shape

        # If phased array with point at top, marginally move top keypoints laterally
        if torch.abs(x2 - x1) < self.point_thresh and probe == Probe.PHASED.value:
            x1 -= 0.5
            x2 += 0.5

        # Determine point and angle of intersection of lateral beam bounds
        x_itn, y_itn = get_point_of_intersection(*keypoints)
        theta = get_angle_of_intersection(x3, y3, x4, y4, x_itn, y_itn)

        # Randomly sample the width fraction of the original beam
        top_width = self.min_top_width + \
                     torch.rand(()) * (self.max_top_width - self.min_top_width)
        top_width_scale = top_width / (x2 - x1) * (x4 - x3)
        new_x1 = x_itn - (x_itn - x1) * top_width_scale
        new_x2 = x_itn + (x2 - x_itn) * top_width_scale

        y_coords = torch.linspace(0, h - 1, h)
        x_coords = torch.linspace(0, w - 1, w)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Determine point and angle of intersection of new lateral beam bounds
        new_x_itn, new_y_itn = get_point_of_intersection(new_x1, y1, new_x2, y2, x3, y3, x4, y4)
        new_theta = get_angle_of_intersection(x3, y3, x4, y4, new_x_itn, new_y_itn)

        # Calculate polar coordinates with respect to new beam shape
        new_phi = torch.atan2(xx - new_x_itn, yy - new_y_itn)
        new_r = torch.sqrt((new_x_itn - xx)**2 + (new_y_itn - yy)**2)

        # Determine angular and radius ratios for old:new beam
        theta_ratio = theta / new_theta
        rad_ratio = (yy - y_itn) * torch.cos(new_theta / 2.) / torch.cos(theta / 2.) / (yy - new_y_itn)

        # Compute mapping from new image to old image coordinates
        new_xx = x_itn + new_r * rad_ratio * torch.sin(theta_ratio * new_phi)
        new_yy = y_itn + new_r * rad_ratio * torch.cos(theta_ratio * new_phi)

        # If square ROI, beam was not initially circular.
        # Resize so that bottom of beam coincides with bottom of image.
        if self.square_roi:
            vertical_adjust = (h - 1) / new_yy[-1, x_itn.int()]
            new_yy = new_yy * vertical_adjust
            new_y3 = y3 / vertical_adjust
            new_y4 = y4 / vertical_adjust
            horiz_adjust = w / (new_xx[new_y3.int(), w - 1] - new_xx[new_y3.int(), new_x_itn.int()]) / 2.
            new_xx = (new_xx - new_x_itn) * horiz_adjust + new_x_itn
        else:
            new_y3 = y3
            new_y4 = y4

        # Normalize the coordinate map
        new_yy = new_yy / h * 2. - 1.
        new_xx = new_xx / w * 2. - 1.
        adjusted_grid = torch.stack([new_xx, new_yy], dim=-1)

        # Construct new image using values from old image
        new_image = nn.functional.grid_sample(image.unsqueeze(0).float(), adjusted_grid.unsqueeze(0)).squeeze(0)
        new_mask = nn.functional.grid_sample(mask.unsqueeze(0).float(), adjusted_grid.unsqueeze(0)).squeeze(0)
        new_image = new_image * new_mask

        # Determine new keypoints
        new_keypoints = torch.stack([
            new_x1,
            y1,
            new_x2,
            y2,
            x3,
            new_y3,
            x4,
            new_y4
        ])

        return new_image, label, new_keypoints, new_mask, probe

