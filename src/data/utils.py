from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import cv2

from src.constants import Probe


def get_beam_mask(h, w, keypoints, probe):

    mask = np.zeros((h, w, 1))

    if probe == Probe.LINEAR.value:
        polygon = get_linear_beam_shape(keypoints)
    elif probe == Probe.CURVILINEAR.value:
        polygon = get_curved_linear_beam_shape(keypoints)
    elif probe == Probe.PHASED.value:
        polygon = get_phased_array_beam_shape(keypoints)
    else:
        raise Exception("Probe type {} does not exist".format(probe))

    for i in range(len(polygon) - 1):
        mask = cv2.line(
            mask,
            (polygon[i][0], polygon[i][1]),
            (polygon[i + 1][0], polygon[i + 1][1]),
            (0, 0, 255),
        )

    return mask

    return

def get_point_of_intersection(keypoints) -> Tuple[float, float]:
    """Determines the origina point of the ultrasound beam.

    Calculates point of intersection of the two linear lateral
    bounds of the ultrasound beam. The first line
    passes through (x1, y1) and (x3, y3). The second line passes
    through (x2, y2) and (x4, y4).
    Args:
        keypoints: Beam keypoints, in format [x1, y1, x2, y2, x3, y3, x4, y4]

    Returns:
        (x, y) coordinates for point of intersection
    """

    x1, y1, x2, y2, x3, y3, x4, y4 = keypoints

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


def get_points_on_arc(
    x_a: float,
    y_a: float,
    x_b: float,
    x_c: float,
    y_c: float,
    bottom_flush: bool = False,
    n_points: int = 200,
) -> List[Tuple[int, int]]:
    """Sample points on the arc of a circle.

    Returns a list of points on the arc of a circle
    where (x_c, y_c) is the centre of the circle and
    (x_a, y_a) and (x_b, y_b) are points on the circle
    at each end of the arc.
    Note: Assumes that the coordinates are in image space.
          If the images are resized, the arc is no longer
          on a circle.

    Args:
        x_a: x-coordinate of point a
        y_a: y-coordinate of point a
        x_b: x-coordinate of point b
        x_c: x-coordinate of centre of circle
        y_c: y-coordinate of centre of circle
        n_points: Number of points to sample on the arc

    Returns:
        List of (x, y) coordinates
    """

    xs = np.linspace(x_a, x_b, n_points)
    if bottom_flush:
        coeff = np.array([
            [(x_a - x_c) ** 2, (y_a - y_c) ** 2],
            [(x_b - x_c) ** 2, (y_a - y_c) ** 2]
        ])
        dep = np.array([1, 1])
        a, b = np.linalg.solve(coeff, dep)
        ys = np.sqrt((1 - (xs - x_c) * a) / b) + y_c
    else:
        radius = np.linalg.norm([x_c - x_a, y_c - y_a])
        ys = np.sqrt(radius**2 - (xs - x_c) ** 2) + y_c
    arc = [(int(xs[i]), int(ys[i])) for i in range(len(xs))]
    return arc


def get_phased_array_beam_shape(
    keypoints: npt.NDArray[np.float32],
) -> List[Tuple[float, float]]:
    """Returns polygon representing outline of phased array probe.

    Determines points on a polygon that give the bounds of a phased
    array probe. The bottom circular bound is approximated by edges
    on a polygon.
    Args:
        keypoints: Beam keypoints, in format [x1, y1, x2, y2, x3, y3, x4, y4]

    Returns:
        List of point coordinates defining the shape of the mask
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = keypoints

    x_itn, y_itn = get_point_of_intersection(keypoints)

    arc = get_points_on_arc(x3, y3, x4, x_itn, y_itn)

    polygon = [(x1, y1), (x3, y3)]
    for p in arc:
        polygon.append(p)
    polygon.append((x2, y2))
    polygon.append((x1, y1))
    return polygon


def get_curved_linear_beam_shape(
    keypoints: npt.NDArray[np.float32],
) -> List[Tuple[float, float]]:
    """Returns polygon representing outline of phased array probe.

    Determines points on a polygon that give the bounds of a curved
    linear probe. The bottom and top circular bounds are
    approximated by edges on a polygon.
    Args:
        keypoints: Beam keypoints, in format [x1, y1, x2, y2, x3, y3, x4, y4]

    Returns:
        List of point coordinates defining the shape of the mask
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = keypoints

    x_itn, y_itn = get_point_of_intersection(keypoints)

    # Sample points on the top and bottom arcs
    arc_bot = get_points_on_arc(x3, y3, x4, x_itn, y_itn)
    arc_top = get_points_on_arc(x1, y1, x2, x_itn, y_itn)

    # Get a list of points on the polygon in the order in which the
    # edges will be drawn. We start from the top left corner of the
    # beam and proceed counterclockwise.
    polygon = [(x1, y1), (x3, y3)]  # Left line
    for p in arc_bot:  # Bottom circle
        polygon.append(p)
    polygon.append((x2, y2))  # Right line
    for p in reversed(arc_top):  # Top circle
        polygon.append(p)
    return polygon


def get_linear_beam_shape(
    keypoints: npt.NDArray[np.float32],
) -> List[Tuple[float, float]]:
    """Return polygon for linear probe given keypoints.

    Args:
        keypoints: Beam keypoints, in format [x1, y1, x2, y2, x3, y3, x4, y4]

    Returns:
        list of point coordinates representing shape of mask polygon. For
        linear probes, only the top left and bottom right corners are used.
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = keypoints

    # The polygon defining the mask is a rectangle
    polygon = [
        (x1, y1),
        (x2, y2),
        (x4, y4),
        (x3, y3),
        (x1, y1),
    ]

    return polygon


def get_linear_beam_shape(
    keypoints,
) -> List[Tuple[float, float]]:
    """Return polygon for linear probe given keypoints.

    Args:
        keypoints: keypoints for phased array
            `[[x1, y1], [x2, y2], ..., [xn, yn,]]`

    Returns:
        list of point coordinates representing shape of mask polygon. For
        linear probes, only the top left and bottom right corners are used.
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = keypoints

    # The polygon defining the mask is a rectangle
    polygon = [
        (x1, y1),
        (x2, y2),
        (x3, y3),
        (x4, y4),
        (x1, y1),
    ]

    return polygon