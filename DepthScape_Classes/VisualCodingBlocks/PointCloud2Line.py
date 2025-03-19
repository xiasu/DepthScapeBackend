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
    # Convert to numpy array
    points = np.array(random_points)
    
    # Calculate centroid
    centroid = np.mean(points, axis=0)
    
    # Center the points
    centered_points = points - centroid
    
    # Calculate covariance matrix
    cov_matrix = np.dot(centered_points.T, centered_points)
    
    # Get eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Direction is eigenvector with largest eigenvalue
    A = eigenvectors[:, np.argmax(eigenvalues)]
    A = A / np.linalg.norm(A)
    
    # Point on line is the centroid
    B = centroid
    
    # Calculate distances from points to line
    v = points - B
    distances = np.linalg.norm(np.cross(v, A), axis=1)
    
    # Get inliers (points within threshold distance)
    threshold = 0.05  # Adjust this threshold as needed
    inliners = np.where(distances < threshold)[0]
    
    # Project inlier points onto line
    inlier_points = points[inliners]
    v = inlier_points - B
    dots = np.dot(v, A)
    projections = B + np.outer(dots, A)
    
    # Find the two farthest projected points
    proj_dists = np.linalg.norm(projections - projections[0], axis=1)
    farthest_idx = np.argmax(proj_dists)
    
    # Update line point B to be the midpoint between the two ends
    line_start = projections[0]
    line_end = projections[farthest_idx]
    B = (line_start + line_end) / 2
    # Calculate length as distance between start and end points
    length = np.linalg.norm(line_end - line_start)
    #A is slope, B is point
    return Line(B, A,length, inliners)