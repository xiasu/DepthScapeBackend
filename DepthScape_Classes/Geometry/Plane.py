import numpy as np
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
        if len(inliers) > 0:
            self.get_primary_and_boundary()
            self.extruded = self.get_extruded()
        #self.boundary = self.get_boundary()
        
    def get_primary_and_boundary(self):
        #This method should return the primary axis of the plane. It should be one or two vectors that are orthogonal to the normal vector of the plane.
        if len(self.inliers) < 3:
            return None

        # Center the inliers
        #inliers_centered = self.inliers - np.mean(self.inliers, axis=0)

        #Cast all the inliners inside the plane
        inliers_centered = self.inliers - np.mean(self.inliers, axis=0)
        inliers_centered -= np.outer(np.dot(inliers_centered, self.normal), self.normal)

        # Perform SVD
        _, s, vh = np.linalg.svd(inliers_centered)

        # The primary directions are the first two right-singular vectors
        primary_directions = vh[:2]
        #Use these two directions to define the boundary of the plane
        # Project the inliers onto the primary directions
        projections = np.dot(inliers_centered, primary_directions.T)

        # Find the min and max projections to define the boundary
        min_proj = np.min(projections, axis=0)
        max_proj = np.max(projections, axis=0)

        # Define the boundary points in the plane
        boundary_points = []
        for i in range(2):
            for j in range(2):
                point = min_proj + (max_proj - min_proj) * np.array([i, j])
                boundary_points.append(point)

        self.boundary = np.dot(boundary_points, primary_directions) + np.mean(self.inliers, axis=0)
        #calculate the span of the plane , with respect to the primary directions
        # Calculate span along primary directions
        self.span = np.array([max_proj[i] - min_proj[i] for i in range(2)])

        # Compare the ratio of the two most primary directions
        self.is_rectangular = True
        if s[1] / s[0] < 0.1:
            #primary_directions = primary_directions[:1]
            self.is_rectangular = False

        self.primary = primary_directions
        # Calculate the center of the boundary
        self.center = np.mean(self.boundary, axis=0)

        #Also use the primary directions and normal to calculate the rotation euler angles of the plane
        # Calculate rotation matrix from primary directions and normal
        rotation_matrix = np.vstack([self.primary[0], self.primary[1], self.normal])
        
        # Convert rotation matrix to euler angles (in radians)
        # Using intrinsic rotations (xyz order)
        sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
            y = np.arctan2(-rotation_matrix[2,0], sy)
            z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        else:
            x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
            y = np.arctan2(-rotation_matrix[2,0], sy)
            z = 0

        self.euler_angles = np.array([x, y, z])
    def set_primary_center_span(self,primary,center,span):
        #This method should set the primary and boundary of the plane. Specifically, it's used for the facial and skeletal cases, where the plane is defined by the face or the skeleton.
        self.primary = primary
        self.center = center
        self.span = span
        self.boundary = self.get_boundary_with_span()

    def get_boundary_with_span(self):
        #This method should return the boundary of the plane based on the primary directions and the span.
        boundary_points = []
        for i in range(2):
            for j in range(2):
                # Calculate offset from center using primary directions scaled by span
                offset = (i - 0.5) * self.span[0] * self.primary[0] + (j - 0.5) * self.span[1] * self.primary[1]
                point = self.center + offset
                boundary_points.append(point)
        return np.array(boundary_points)
    def get_extruded(self):
        #This method should return the extruded plane.
        #The extruded planes are two planes that are perpendicular to the original plane and are of the same size.
        #The extruded plane should use one of the primary direction as its new primary direction, and the other direction as its normal vector.
        #The extruded plane should be of the same size as the original plane.
        
        extruded_planes = []
        
        # Create two extruded planes
        for i in range(2):
            # For the first extruded plane, use primary[0] as primary and normal as second direction
            # For the second extruded plane, use primary[1] as primary and normal as second direction
            primary_dir = self.primary[i]
            normal_dir = self.normal
            
            # The other primary direction becomes the normal of the extruded plane
            other_primary = self.primary[(i+1)%2]
            
            # Calculate the plane equation coefficients (ax + by + cz + d = 0)
            a, b, c = other_primary
            # Calculate d using a point on the plane (the center)
            d = -np.dot(other_primary, self.center)
            
            # Create the new plane
            extruded_plane = Plane(a, b, c, d, [])
            
            # Set the primary directions, center, and span for the extruded plane
            # Primary directions are the selected primary direction and the normal of the original plane
            new_primary = [primary_dir, normal_dir]
            
            # Use the same center as the original plane
            new_center = self.center.copy()
            
            # The span should match the original plane's corresponding dimension
            # and use the height/thickness for the dimension along the normal
            new_span = np.array([self.span[i], self.span[(i+1)%2]])  # Using the same span for simplicity
            
            # Set the properties of the extruded plane
            extruded_plane.set_primary_center_span(new_primary, new_center, new_span)
            
            extruded_planes.append(extruded_plane)
            
        return extruded_planes
        
        
    


