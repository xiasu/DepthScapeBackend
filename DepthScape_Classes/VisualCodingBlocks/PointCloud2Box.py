from ..Geometry import Box
from ..Geometry import PointCloud
import pyransac3d as pyrsc

def PointCloud2Box(pointCloud):
    selected_points = pointCloud.points
    box1 = pyrsc.Box()
    best_eq, best_inliers = box1.fit(selected_points, 0.01)
    return Box(best_eq, best_inliers)