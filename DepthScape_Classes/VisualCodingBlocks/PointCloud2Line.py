from ..Geometry import Line
from ..Geometry import PointCloud
import pyransac3d as pyrsc
import random
import numpy as np
def PointCloud2Line(pointCloud):
    selected_points = pointCloud.points
    np.random.seed(42)
    if selected_points.shape[0] > 10000:
        random_indices = np.random.choice(selected_points.shape[0], 10000, replace=False)
        random_points = selected_points[random_indices]
    else:
        random_points = selected_points[:]
    line1 = pyrsc.Line()
    A, B, inliners = line1.fit(random_points, 0.01)
    #A is slope, B is point
    return Line(B, A, inliners)