import numpy as np


def filter_zero_points(points, carry):
    indices = ~np.all(points == 0, axis=1)
    return points[indices], carry[indices]


def filter_by_depth(points, carry, zmax=0.95, zmin=0):
    indices = (points[:, 2] < zmax) & (points[:, 2] > zmin)
    return points[indices], carry[indices]