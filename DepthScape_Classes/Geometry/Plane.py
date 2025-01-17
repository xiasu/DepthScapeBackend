class Plane:
    #This class represents a 3D plane in the depth space. It should have four parameters: a,b,c,d
    # Plane equation coefficients: ax + by + cz + d = 0
    def __init__(self,a,b,c,d, inliers):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.inliers = inliers
    def get_boundary(self):
        #This method should return the boundary of the plane
        pass