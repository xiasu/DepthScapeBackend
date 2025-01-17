import pyransac3d as pyrsc
from ..Geometry import PointCloud
from ..Geometry import Plane

def PointCloud2Plane(pointCloud):
    selected_points = pointCloud.points
    plane1 = pyrsc.Plane()
    best_eq, best_inliers = plane1.fit(selected_points, 0.01)
    return Plane(best_eq[0], best_eq[1], best_eq[2], best_eq[3], best_inliers)