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
            mask: Tensor,
            probe: Tensor,
            **kwargs
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """Transformation that simulates speckle noise

        Simulates the addition of speckle noise to the input image.
        Following the method from https://ieeexplore.ieee.org/document/7967056,
        images are interpolated from sampled points in the image that have been
        perturbed with multiple rounds of additive noise from a circular Gaussian,
        mimicking noise from multiple incoherent phasors. The transformation
        follows the 4-step method outlined in the paper.

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

        x1, y1, x2, y2, x3, y3, x4, y4 = keypoints
        c, h, w = image.shape

        # Determine number of x-, y-values for sample points
        lateral_res = (self.min_lateral_res + torch.rand(()) * (self.max_lateral_res - self.min_lateral_res)).int()
        axial_res = (self.min_axial_res + torch.rand(()) * (self.max_axial_res - self.min_axial_res)).int()

        # Designate sample points in the image
        if probe == Probe.LINEAR.value:

            # For linear probe types, sample points are arranged in a grid
            y_coords = torch.linspace(y1, y3, axial_res)
            x_coords = torch.linspace(x1, x2, lateral_res)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        else:

            # For non-linear probes, sample points uniformly distributed across radial lines
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
            xx = x_itn + rr * torch.sin(pp)
            yy = y_itn + rr * torch.cos(pp)

        # Obtain image intensity at sampled points (Step 1)
        xx = torch.clamp(xx, 0, w - 1).int()
        yy = torch.clamp(yy, 0, h - 1).int()
        img_sampled = torch.mean(image.float(), 0)[yy.int(), xx.int()]

        # Get sampled noisy intensity values (Step 2)
        amp = torch.complex(torch.sqrt(img_sampled), torch.zeros_like(img_sampled))
        m = torch.randint(self.min_phasors, self.max_phasors, ()) # Number of phasors
        for i in range(m):
            real = self.sigma * torch.randn_like(amp.real)
            imaginary = self.sigma * torch.randn_like(amp.imag)
            amp += torch.complex(real, imaginary)
        img_sampled = amp.abs() ** 2
        img_sampled = img_sampled.unsqueeze(0).repeat(c, 1, 1)

        # Interpolate new image based on noisy intensity at sample points (Step 3)
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

        # Clamp pixel intensities (Step 4)
        new_image = torch.clamp(new_image, 0., 255.)
        return new_image, label, keypoints, mask, probe


class GaussianNoise(nn.Module):
    """Transformation that adds Gaussian noise to ultrasound image
        """

    def __init__(
            self,
            min_sigma: float = 0.5,
            max_sigma: float = 3.0,
    ):
        """Initializes the GaussianNoise layer.

        Args:
            min_sigma: Minimum std deviation of the Gaussian
            max_sigma: Maximum std deviation of the Gaussian
        """
        super(GaussianNoise, self).__init__()
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def forward(
            self,
            image: Tensor,
            label: Tensor,
            keypoints: Tensor,
            mask: Tensor,
            probe: Tensor,
            **kwargs
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """Adds Gaussian noise to the image.

        Samples Gaussian noise and adds it to each pixel constituting
        the ultrasound beam.

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

        # Sample Gaussian noise to be added to each pixel in the ultrasound beam
        sigma = self.min_sigma + torch.rand(()) * (self.max_sigma - self.min_sigma)
        noise = sigma * torch.randn_like(mask)
        noise = noise * mask

        # Add the Gaussian noise to the image
        new_image = image + noise
        new_image = torch.clamp(new_image, 0., 255.)
        return new_image, label, keypoints, mask, probe


class SaltAndPepperNoise(nn.Module):
    """Transformation that adds salt & pepper noise to ultrasound image
        """

    def __init__(
            self,
            min_salt_frac: float = 0.005,
            max_salt_frac: float = 0.01,
            min_pepper_frac: float = 0.005,
            max_pepper_frac: float = 0.01,
    ):
        """Initializes the SaltAndPepper noise layer.

        Args:
            min_salt_frac: Minimum fraction of the image with salt noise
            max_salt_frac: Maximum fraction of the image with salt noise
            min_pepper_frac: Minimum fraction of the image with pepper noise
            max_pepper_frac: Maximum fraction of the image with pepper noise
        """
        super(SaltAndPepperNoise, self).__init__()
        self.min_salt_frac = min_salt_frac
        self.max_salt_frac = max_salt_frac
        self.min_pepper_frac = min_pepper_frac
        self.max_pepper_frac = max_pepper_frac

    def forward(
            self,
            image: Tensor,
            label: Tensor,
            keypoints: Tensor,
            mask: Tensor,
            probe: Tensor,
            **kwargs
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """Samples and applies salt & pepper noise

        Applies salt and pepper noise at random locations in the image.
        Salt noise consists of setting the pixel intensity to white
        and pepper noise consists of setting the intensity to black.

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

        # Determine the number of pixels to shade white and black
        c, h, w = image.shape
        n_salt = int((self.min_salt_frac + torch.rand(()) * (self.max_salt_frac - self.min_salt_frac)) * h * w)
        n_pepper = int((self.min_pepper_frac + torch.rand(()) * (self.max_pepper_frac - self.min_pepper_frac)) * h * w)

        # Add salt noise
        salt_locs = torch.randint(0, w * h - 1, size=(n_salt,))
        image[:, salt_locs // w, salt_locs % w] = 255.

        # Add pepper noise
        pepper_locs = torch.randint(0, w * h - 1, size=(n_pepper,))
        image[:, pepper_locs // w, pepper_locs % w] = 0.

        new_image = image * mask    # Only apply noise to the ultrasound beam
        return new_image, label, keypoints, mask, probe

