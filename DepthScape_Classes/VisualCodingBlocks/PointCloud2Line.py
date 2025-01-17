from ..Geometry import Line
from ..Geometry import PointCloud
import pyransac3d as pyrsc

def PointCloud2Line(pointCloud):
    selected_points = pointCloud.points
    line1 = pyrsc.Line()
    A, B, inliners = line1.fit(selected_points, 0.01)
    #A is slope, B is point
    return Line(B, A, inliers)