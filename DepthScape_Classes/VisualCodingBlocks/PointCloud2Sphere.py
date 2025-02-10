from ..Geometry import Sphere
from ..Geometry import PointCloud
import pyransac3d as pyrsc
import random
import numpy as np
def PointCloud2Sphere(pointCloud):
    selected_points = pointCloud.points
    np.random.seed(42)
    if selected_points.shape[0] > 10000:
        random_indices = np.random.choice(selected_points.shape[0], 10000, replace=False)
        random_points = selected_points[random_indices]
    else:
        random_points = selected_points[:]
    #TODO: check if this fitting works. We may need a containing fitter instead.
    sphere1 = pyrsc.Sphere()
    center, radius, best_inliers = sphere1.fit(random_points, 0.01)
    center_np=np.array(center)
    return Sphere(center_np, radius, best_inliers)