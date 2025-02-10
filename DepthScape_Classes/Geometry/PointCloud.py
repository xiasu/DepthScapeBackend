import numpy as np
import random
class PointCloud:
    def __init__(self, points):
        self.points = points
        self.calculate_parameters()
    def calculate_parameters(self):
        self.center = np.mean(self.points, axis=0)
        self.bounding_box = np.array([np.min(self.points, axis=0), np.max(self.points, axis=0)])
        distances = np.linalg.norm(self.points - self.center, axis=1)
        # Determine the 99th percentile distance to include 99% of points
        self.radius = np.percentile(distances, 99)
    def get_downsampled_points(self, count=10000):
        if self.points.shape[0] > count:
            random_indices = np.random.choice(self.points.shape[0], count, replace=False)
            return self.points[random_indices]
        else:
            return self.points