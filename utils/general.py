import numpy as np


def integral_intensity_projection(image):
    """
    This function takes a 2D image and returns the integral projection of the intensity in the vertical and
    horizontal directions.

    Args:
        image: A 2D array representing an image

    Returns:
        hor: A 1-D array containing the integral projection of intensity in the horizontal direction
        ver: A 1-D array containing the integral projection of intensity in the vertical direction
    """
    ver = np.sum(image, axis=0).astype('int32')
    hor = np.sum(image, axis=1).astype('int32')

    return hor, ver


def distance(x0, y0, x1, y1, x2, y2):
    """
    This function calculates the Euclidean distance between a point and a line.
    Args:
        x0: The x-coordinate of the point
        y0: The y-coordinate of the point
        x1: The x-coordinate of the first point on the line
        y1: The y-coordinate of the first point on the line
        x2: The x-coordinate of the second point on the line
        y2: The y-coordinate of the second point on the line

    Returns: The Euclidean distance between the point and the line
    """
    return np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def intersection(p1, p2, p3, p4):
    """
    This function finds the intersection point between two lines given by four points.
    Args:
        p1, p2, p3, p4: Four points that define two lines: (p1, p2) and (p3, p4)

    Returns: A tuple containing the x and y coordinates of the intersection point
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return px, py
