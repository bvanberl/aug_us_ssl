import torch
from torch import nn, Tensor
from torchvision.transforms.v2 import functional as tvf, Transform
from torchvision.transforms import InterpolationMode as im

from src.constants import BeamKeypoints, Probe
from src.augmentations.aug_utils import *

class SpeckleNoise(nn.Module):
    """Transform layer that simulates speckle noise.

    This transformation simulates the application of speckle
    noise to the input ultrasound image.
    """

    def __init__(
            self,
            square_roi: bool = False,
            min_lateral_res: int = 100,
            max_lateral_res: int = 100,
            min_axial_res: int = 100,
            max_axial_res: int = 100,
            min_phasors: int = 5,
            max_phasors: int = 15,
            sigma: float = 0.2,
    ):
        """Initializes the SpeckleNoise layer.

        Args:
        """
        super(SpeckleNoise, self).__init__()
        self.square_roi = square_roi
        self.min_lateral_res = min_lateral_res
        self.max_lateral_res = max_lateral_res
        self.min_axial_res = min_axial_res
        self.max_axial_res = max_axial_res
        self.min_phasors = min_phasors
        self.max_phasors = max_phasors
        self.sigma = sigma

    def forward(
            self,
            image: Tensor,
            label: Tensor,
            keypoints: Tensor,
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
            probe: Probe type of the image

        Returns:
            augmented image, transformed keypoints, new probe type (linear)
        """

        x1, y1, x2, y2, x3, y3, x4, y4 = keypoints
        c, h, w = image.shape

        lateral_res = (self.min_lateral_res + torch.rand(()) * (self.max_lateral_res - self.min_lateral_res)).int()
        axial_res = (self.min_axial_res + torch.rand(()) * (self.max_axial_res - self.min_axial_res)).int()

        if probe == Probe.LINEAR.value:
            y_coords = torch.linspace(y1, y3, axial_res)
            x_coords = torch.linspace(x1, x2, lateral_res)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        else:
            x_itn, y_itn = get_point_of_intersection(*keypoints)

            if self.square_roi:
                # If keypoints are off-screen, shift and scale image so that they
                # are on-screen and comprise the left and right bounds.
                resize_w = torch.floor(2 * torch.sqrt((h - y_itn) ** 2 - (y3 - y_itn) ** 2))
                w = resize_w.int()
                x3 = x3 * w / h
                x4 = x4 * w / h
                x_itn = x_itn * w / h
                image = tvf.resize(image, [h, w])
            bot_r = torch.sqrt((x3 - x_itn) ** 2 + (y3 - y_itn) ** 2)

            theta = get_angle_of_intersection(x3, y3, x4, y4, x_itn, y_itn)
            phis = torch.linspace(-theta / 2, theta / 2, lateral_res)
            rads = torch.linspace(y1, bot_r, axial_res)
            rr, pp = torch.meshgrid(rads, phis, indexing='ij')
            xx = torch.clamp(x_itn + rr * torch.sin(pp), 0, w - 1).int()
            yy = torch.clamp(y_itn + rr * torch.cos(pp), 0, h - 1).int()

        img_sampled = torch.mean(image.float(), 0)[yy.int(), xx.int()]
        amp = torch.complex(torch.sqrt(img_sampled), torch.zeros_like(img_sampled))
        m = torch.randint(self.min_phasors, self.max_phasors, ())
        for i in range(m):
            real = self.sigma * torch.randn_like(amp.real)
            imaginary = self.sigma * torch.randn_like(amp.imag)
            amp += torch.complex(real, imaginary)
        img_sampled = amp.abs() ** 2
        img_sampled = img_sampled.unsqueeze(0).repeat(c, 1, 1)

        if probe == Probe.LINEAR.value:
            y_coords = torch.linspace(-1., 1., h)
            x_coords = torch.linspace(-1., 1., w)
            out_yy, out_xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        else:
            y_coords = torch.linspace(0., h - 1., h)
            x_coords = torch.linspace(0., w - 1., w)
            new_yy, new_xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            out_xx = torch.atan2(new_xx - x_itn, new_yy - y_itn) / theta * 2.
            out_yy = torch.sqrt((x_itn - new_xx) ** 2 + (y_itn - new_yy) ** 2) / bot_r * 2. - 1.
        flow_field = torch.stack([out_xx, out_yy], dim=-1)
        new_image = nn.functional.grid_sample(img_sampled.unsqueeze(0).float(), flow_field.unsqueeze(0)).squeeze(0)

        if self.square_roi:
            new_image = tvf.resize(new_image, [h, h])

        new_image = torch.clamp(new_image, 0., 255.)
        return new_image, label, keypoints, probe

