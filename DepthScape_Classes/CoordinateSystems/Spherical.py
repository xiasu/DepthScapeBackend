class Spherical:
    def __init__(self, base_sphere, visual_coding,variables):
        self.base_sphere = base_sphere
        #self.base_direction = base_direction
        self.visual_coding = visual_coding
        self.variables = variables
        self.calculate_parameters()
        self.create_json()
    def calculate_parameters(self):
        #Calculate the parameters of the spherical coordinate system
        self.center = self.base_sphere.center
        self.radius = self.base_sphere.radius
    def create_json(self):
        #Create a JSON representation of the spherical coordinate system
        self.json = {
            "name":self.visual_coding.name if self.visual_coding else "Spherical Coordinate System",
            "description":self.visual_coding.description if self.visual_coding else "Cylindrical Coordinate System",
            "type": "Spherical",
            "center": self.center.tolist(),
            "radius": self.radius,
            #"primary": self.base_direction.tolist() if hasattr(self, 'primary') else None,
        }