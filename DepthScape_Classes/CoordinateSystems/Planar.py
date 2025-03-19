import numpy as np
class Planar:
    def __init__(self, plane=None, visual_coding=None,variables=None):
        self.plane = plane
        #self.direction_1 = direction_1
        #self.direction_2 = direction_2
        self.visual_coding = visual_coding
        self.variables=variables
        self.calculate_parameters()
        self.create_json()
    def calculate_parameters(self):
        if self.plane:
            self.normal = np.array([self.plane.a, self.plane.b, self.plane.c])
            norm = np.linalg.norm(self.normal)
            if norm != 0:
                self.normal = self.normal / norm
            self.primary = self.plane.primary
            self.boundary = self.plane.boundary
            # if self.direction_1 is not None:
            #     self.primary = self.direction_1
            if hasattr(self.plane, 'center'):
                self.center = self.plane.center
            else:
                self.center = np.zeros(3)
            self.span = self.plane.span
            self.euler_angles = self.plane.euler_angles
        else:
            pass
            #Use the two directions to define the plane
            #The plane only have a normal vector, and the primary directions are the two directions provided.
            #self.normal = np.cross(self.direction_1, self.direction_2)
            #self.primary = np.array([self.direction_1, self.direction_2])
    def create_json(self):
        #Create a JSON representation of the planar coordinate system
        self.json = {
            "name":self.visual_coding.name if self.visual_coding else "Planar Coordinate System",
            "description":self.visual_coding.description if self.visual_coding else "Planar Coordinate System",
            "type": "Planar",
            "normal": self.normal.tolist(),
            "center": self.center.tolist(),
            "plane_parameters": [float(self.plane.a), float(self.plane.b), float(self.plane.c), float(self.plane.d)] if self.plane else None,
            "primary": [dir.tolist() for dir in self.primary],
            "boundary": self.span.tolist() if hasattr(self, 'span') else None,
            "euler_angles": self.euler_angles.tolist() if hasattr(self, 'euler_angles') else None,
        }
        #"boundary": self.boundary.tolist() if hasattr(self, 'boundary') else None,