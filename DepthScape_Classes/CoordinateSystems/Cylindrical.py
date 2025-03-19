class Cylindrical:
    def __init__(self, base_cylinder, visual_coding,variables):
        self.base_cylinder = base_cylinder
        #self.base_direction = base_direction
        self.visual_coding = visual_coding
        self.variables = variables
        self.calculate_parameters()
        self.create_json()
    def calculate_parameters(self):
        #Calculate the parameters of the cylindrical coordinate system
        self.center = self.base_cylinder.center
        self.axis = self.base_cylinder.axis
        self.radius = self.base_cylinder.radius
        self.height = self.base_cylinder.height
    def create_json(self):
        #Create a JSON representation of the cylindrical coordinate system
        self.json = {
            "name":self.visual_coding.name if self.visual_coding else "Cylindrical Coordinate System",
            "description":self.visual_coding.description if self.visual_coding else "Cylindrical Coordinate System",
            "name":self.visual_coding.name if self.visual_coding else "Cylindrical",
            "type": "Cylindrical",
            "center": self.center.tolist(),
            "axis": self.axis.tolist(),
            "radius": self.radius,
            "height": self.height,
            #"primary": self.base_direction.tolist() if hasattr(self, 'primary') else None,
        }