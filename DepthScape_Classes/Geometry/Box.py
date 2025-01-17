class Box:
    def __init__(self, eq, inliers):
        #We use a (3,4) numpy array to represent the box. Each of the three 4 dimensional vectors represent a face of the box
        self.eq=eq
        self.inliners=inliers 