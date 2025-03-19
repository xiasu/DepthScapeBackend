import numpy as np
class Line:
    def __init__(self, center, slope,length, inliners):
        self.center = center
        #The slope is a 3 dimensional vector representing the slope of the line in 3D space
        self.slope = slope
        self.length = length
        self.inliners = inliners
        #Normalizing the slope to get the direction
        self.direction = self.slope / np.linalg.norm(self.slope)
        self.endPoint = self.center + self.direction * self.length/2
        self.startPoint = self.center - self.direction * self.length/2
