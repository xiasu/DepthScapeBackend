import pyransac3d as pyrsc
from ..Geometry import PointCloud
from ..Geometry import Plane
import random
import numpy as np
def PointCloud2Plane(pointCloud):
    selected_points = pointCloud.points
    np.random.seed(42)
    if selected_points.shape[0] > 10000:
        random_indices = np.random.choice(selected_points.shape[0], 10000, replace=False)
        random_points = selected_points[random_indices]
    else:
        random_points = selected_points[:]
    #Use iterative plane detection instead!
    count_points=len(random_points)
    found_planes = []
    #Then iteratively apply plane detection, with a maximum of 3 planes, to the selected points. Each time a plane is detected, remove the inliers from the point cloud and repeat the process until no more planes can be detected.
    for i in range(3): 
        plane = pyrsc.Plane()
        best_eq, best_inliers = plane.fit(random_points,thresh=0.05)
        pl=Plane(best_eq[0],best_eq[1],best_eq[2],best_eq[3],random_points[best_inliers])
        # Remove inliers from the point cloud
        random_points = np.delete(random_points, best_inliers, axis=0)
        #deduplicate the planes
        is_new=True
        for existing_plane in found_planes:
            if np.linalg.norm(existing_plane.normal - pl.normal) < 0.1:
                is_new=False
                break
        if is_new:
            found_planes.append(pl)
        if len(random_points) / count_points < 0.03:
            break
    return found_planes
        