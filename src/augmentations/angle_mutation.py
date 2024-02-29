import torch
from torch import nn
from torchvision.transforms.v2 import functional as tvf, Transform
from torchvision.transforms import InterpolationMode as im

from src.constants import BeamKeypoints, Probe
from src.augmentations.aug_utils import *

class ConvexAngleMutation(nn.Module):
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
            max_top_width: float = 0.8
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
        super(ConvexAngleMutation, self).__init__()
        self.square_roi = square_roi
        self.min_top_width = min_top_width
        self.max_top_width = max_top_width

    def forward(
            self,
            image: torch.Tensor,
            label: torch.Tensor,
            keypoints: torch.Tensor,
            probe: torch.Tensor,
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
            label: Label for the image, which is unaltered
            keypoints: 1D tensor with shape (8,) containing beam mask keypoints
                       with format [x1, y1, x2, y2, x3, y3, x4, y4]
            probe: Probe type of the image

        Returns:
            augmented image, transformed keypoints, new probe type (linear)
        """

        if probe == Probe.LINEAR.value:
            return image, label, keypoints, probe

        x1, y1, x2, y2, x3, y3, x4, y4 = keypoints
        c, h, w = image.shape

        # Get point of intersection of left and right beam bounds
        x_itn, y_itn = get_point_of_intersection(*keypoints)

        # Randomly sample the width fraction of the original beam
        top_width = self.min_top_width + \
                     torch.rand(()) * (self.max_top_width - self.min_top_width)
        top_width_scale = top_width / (x2 - x1) * (x4 - x3)
        new_x1 = x_itn - (x_itn - x1) * top_width_scale
        new_x2 = x_itn + (x2 - x_itn) * top_width_scale

        # Determine new keypoints
        new_keypoints = torch.stack([
            new_x1,
            y1,
            new_x2,
            y2,
            x3,
            y3,
            x4,
            y4
        ])

        y_coords = torch.linspace(0, h - 1, h)
        x_coords = torch.linspace(0, w - 1, w)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Determine point and angle of intersection of lateral beam bounds
        theta = get_angle_of_intersection(x3, y3, x4, y4, x_itn, y_itn)

        new_x_itn, new_y_itn = get_point_of_intersection(new_x1, y1, new_x2, y2, x3, y3, x4, y4)
        new_theta = get_angle_of_intersection(x3, y3, x4, y4, new_x_itn, new_y_itn)

        phi_pr = torch.atan2(xx - new_x_itn, yy - new_y_itn)
        r_pr = torch.sqrt((new_x_itn - xx)**2 + (new_y_itn - yy)**2)
        rad_pr = torch.sqrt((new_x_itn - x3) ** 2 + (new_y_itn - y3) ** 2)
        rad = torch.sqrt((x_itn - x3) ** 2 + (y_itn - y3) ** 2)

        rad_ratio = rad / rad_pr
        theta_ratio = theta / new_theta

        itn_to_top = y1 - y_itn

        if probe == Probe.CURVILINEAR.value:
            rad_ratio = (yy - y_itn) * torch.cos(new_theta / 2.) / torch.cos(theta / 2.) / (yy - new_y_itn)
        else:
            rad_ratio = torch.concat([
                (yy[:y3.int()] - y_itn) * torch.cos(new_theta / 2.) / torch.cos(theta / 2.) / (yy[:y3.int()] - new_y_itn),
                torch.ones((h - y3.int(), w)) * rad_ratio
            ], dim=0)

        new_xx = x_itn + r_pr * rad_ratio * torch.sin(theta_ratio * phi_pr)
        new_yy = y_itn + r_pr * rad_ratio * torch.cos(theta_ratio * phi_pr)

        new_yy = new_yy / h * 2. - 1.
        new_xx = new_xx / w * 2. - 1.
        adjusted_grid = torch.stack([new_xx, new_yy], dim=-1)

        # Construct new image using values from old image
        new_image = nn.functional.grid_sample(image.unsqueeze(0).float(), adjusted_grid.unsqueeze(0)).squeeze(0)

        # Set pixels within original beam ROI but outside new ROI to black
        if not self.square_roi:
            new_image = overlay_region_on_image(image, new_image, xx, yy, y1, rad_pr, x3, x4)

        return new_image, label, new_keypoints, probe

