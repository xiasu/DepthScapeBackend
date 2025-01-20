from ..Geometry import Cylinder
from ..Geometry import PointCloud
import numpy as np
from scipy.spatial import ConvexHull

def PointCloud2Cylinder(pointCloud,direction):
    if direction is not None:
        return PointCloud2CylinderWithAxis(pointCloud, direction)
    selected_points = pointCloud.points
    np.random.seed(42)
    if selected_points.shape[0] > 10000:
        random_indices = np.random.choice(selected_points.shape[0], 10000, replace=False)
        random_points = selected_points[random_indices]
    else:
        random_points = selected_points[:]
    points = np.array(random_points)
    # Compute the convex hull of the point cloud
    hull = ConvexHull(points)

    # Extract the vertices of the convex hull
    hull_points = points[hull.vertices]

    # Compute the centroid of the convex hull
    centroid = np.mean(hull_points, axis=0)

    # Compute the covariance matrix of the hull points
    cov_matrix = np.cov(hull_points.T)

    # Perform eigen decomposition to find the principal axes
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # The axis of the cylinder is the eigenvector corresponding to the largest eigenvalue
    axis = eigenvectors[:, np.argmax(eigenvalues)]

    # Project the hull points onto the axis to find the height of the cylinder
    projections = np.dot(hull_points - centroid, axis)
    min_proj, max_proj = np.min(projections), np.max(projections)
    height = max_proj - min_proj

    # Compute the radius of the cylinder
    distances = np.linalg.norm(np.cross(hull_points - centroid, axis), axis=1)
    radius = np.max(distances)

    # Create and return the cylinder
    cylinder = Cylinder(centroid, axis, radius, height)
    return cylinder

def PointCloud2CylinderWithAxis(pointCloud, axis):
    selected_points = pointCloud.points
    np.random.seed(42)
    if selected_points.shape[0] > 10000:
        random_indices = np.random.choice(selected_points.shape[0], 10000, replace=False)
        random_points = selected_points[random_indices]
    else:
        random_points = selected_points[:]
    # Import necessary libraries
    # Convert point cloud to numpy array
    points = np.array(random_points)

    # Normalize the given axis
    axis = axis / np.linalg.norm(axis)

    # Project the points onto the plane orthogonal to the given axis
    projections = points - np.outer(np.dot(points, axis), axis)

    # Compute the centroid of the projected points
    centroid_proj = np.mean(projections, axis=0)

    # Compute the distances from the projected points to the centroid
    distances = np.linalg.norm(projections - centroid_proj, axis=1)

    # The radius of the cylinder is the maximum distance
    radius = np.max(distances)

    # Project the points onto the axis to find the height of the cylinder
    projections_on_axis = np.dot(points, axis)
    min_proj, max_proj = np.min(projections_on_axis), np.max(projections_on_axis)
    height = max_proj - min_proj

    # Compute the centroid of the original points
    centroid = np.mean(points, axis=0)

    # Create and return the cylinder
    cylinder = Cylinder(centroid, axis, radius, height)
    return cylinder