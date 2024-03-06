from typing import Tuple

import torch

from src.constants import Probe

def get_point_of_intersection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
) -> Tuple[float, float]:
    """Determines the point of intersection of two lines.

    Calculates point of intersection of two lines. The first line
    passes through (x1, y1) and (x3, y3). The second line passes
    through (x2, y2) and (x4, y4).
    Args:
        x1: x-coordinate of point 1
        y1: y-coordinate of point 1
        x2: x-coordinate of point 2
        y2: y-coordinate of point 2
        x3: x-coordinate of point 3
        y3: y-coordinate of point 3
        x4: x-coordinate of point 4
        y4: y-coordinate of point 4

    Returns:
        (x, y) coordinates for point of intersection
    """
    # Left line: a1*x + b1*y + c1
    a1 = y3 - y1
    b1 = x1 - x3
    c1 = a1 * x1 + b1 * y1

    # Right line: a2*x + b2*y + c2
    a2 = y4 - y2
    b2 = x2 - x4
    c2 = a2 * x2 + b2 * y2

    # If determinant of the system is 0, lines are parallel
    det = a1 * b2 - a2 * b1
    if det == 0.0:
        raise ValueError("The lines are parallel.")

    # Calculate coordinates of the intersection
    x_itn = (b2 * c1 - b1 * c2) / det
    y_itn = (a1 * c2 - a2 * c1) / det
    return x_itn, y_itn


def get_angle_of_intersection(
    x_a: torch.Tensor,
    y_a: torch.Tensor,
    x_b: torch.Tensor,
    y_b: torch.Tensor,
    x_itn: torch.Tensor,
    y_itn: torch.Tensor,
) -> torch.Tensor:
    """Determine the angle of intersection of 2 lines.

    Given the point of intersection of 2 lines and an
    additional point on each line, determine the angle
    of intersection between the lines.

    Args:
        x_a: x-coordinate of point solely on line a
        y_a: y-coordinate of point solely on line a
        x_b: x-coordinate of point solely on line b
        y_b: y-coordinate of point solely on line b
        x_itn: x-coordinate of point of intersection
        y_itn: y-coordinate of point of intersection

    Returns:
        angle of intersection, in radians
    """
    v1 = torch.stack([x_a - x_itn, y_a - y_itn])
    v2 = torch.stack([x_b - x_itn, y_b - y_itn])

    cos_sim = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
    return torch.acos(torch.clamp(cos_sim, min=-1.0, max=1.0))


def overlay_region_on_image(image, new_image, xx, yy, top, bottom, left, right):
    """Pastes a region in `new_image` onto `image`

    Args:
        image: Image onto which to paste (c, h, w)
        new_image: Image containing region to paste (c, h, w)
        xx: x-coordinate map (h, w)
        yy: y-coordinate map (h, w)
        top: Top bound of region
        bottom: Bottom bound of region
        left: Left bound of region
        right: Right bound of region

    Returns: Original image with region from `new_image` pasted onto it

    """
    orig_mask = (
            (yy >= top)
            & (yy <= bottom)
            & (xx >= left)
            & (xx <= right)
    ).unsqueeze(0)
    orig_mask = orig_mask.repeat(3, 1, 1)
    return torch.where(orig_mask, new_image, image)

def overlay_region_on_image(image, new_image, xx, yy, top, bottom, left, right):
    """Pastes a region in `new_image` onto `image`

    Args:
        image: Image onto which to paste (c, h, w)
        new_image: Image containing region to paste (c, h, w)
        xx: x-coordinate map (h, w)
        yy: y-coordinate map (h, w)
        top: Top bound of region
        bottom: Bottom bound of region
        left: Left bound of region
        right: Right bound of region

    Returns: Original image with region from `new_image` pasted onto it

    """
    orig_mask = (
            (yy >= top)
            & (yy <= bottom)
            & (xx >= left)
            & (xx <= right)
    ).unsqueeze(0)
    orig_mask = orig_mask.repeat(3, 1, 1)
    return torch.where(orig_mask, new_image, image)

