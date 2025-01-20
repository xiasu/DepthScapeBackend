import numpy as np
class Line:
    def __init__(self, point, slope, inliners):
        self.point = point
        #The slope is a 3 dimensional vector representing the slope of the line in 3D space
        self.slope = slope
        self.inliners = inliners
        #Normalizing the slope to get the direction
        self.direction = self.slope / np.linalg.norm(self.slope)