class Plane:
    #This class represents a 3D plane in the depth space. It should have four parameters: a,b,c,d
    # Plane equation coefficients: ax + by + cz + d = 0
    def __init__(self,a,b,c,d, inliers):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.inliers = inliers
        self.normal = np.array([a, b, c])
        norm = np.linalg.norm(self.normal)
        if norm != 0:
            self.normal = self.normal / norm
        self.primary = self.get_primary()
        
    def get_boundary(self):
        #This method should return the boundary of the plane
        pass
    def get_primary(self):
        #This method should return the primary axis of the plane. It should be one or two vectors that are orthogonal to the normal vector of the plane.
        if len(self.inliers) < 3:
            return None

        # Center the inliers
        inliers_centered = self.inliers - np.mean(self.inliers, axis=0)

        # Perform SVD
        _, s, vh = np.linalg.svd(inliers_centered)

        # The primary directions are the first two right-singular vectors
        primary_directions = vh[:2]

        # Compare the ratio of the two most primary directions
        if s[1] / s[0] < 0.1:
            primary_directions = primary_directions[:1]

        return primary_directions
