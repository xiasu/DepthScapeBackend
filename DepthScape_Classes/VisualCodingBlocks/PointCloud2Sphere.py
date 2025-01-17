from ..Geometry import Sphere
from ..Geometry import PointCloud
import pyransac3d as pyrsc
def PointCloud2Sphere(pointCloud):
    selected_points = pointCloud.points
    sphere1 = pyrsc.Sphere()
    center, radius, best_inliers = sphere1.fit(selected_points, 0.01)
    return Sphere(center, radius, best_inliers)