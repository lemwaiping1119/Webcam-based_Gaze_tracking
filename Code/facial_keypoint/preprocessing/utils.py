import os
import numpy as np


def load_pts(path):
    """Reads a .pts file and returns a NumPy array of points."""
    with open(path) as f:
        rows = [row.strip() for row in f]
    head = rows.index('{') + 1
    tail = rows.index('}')
    raw_points = rows[head:tail]
    coords_set = [point.split() for point in raw_points]
    points = np.array([[float(num) for num in coords] for coords in coords_set])
    return points
