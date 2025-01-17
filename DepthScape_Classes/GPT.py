import requests
import os
import base64
try:
    from openai import OpenAI  # newer versions
    def create_client(api_key):
        return OpenAI(api_key=api_key)
except ImportError:
    import openai  # older versions
    def create_client(api_key):
        openai.api_key = api_key
        return openai
class GPT:
    def __init__(self, api_key, image_path):
        self.api_key = api_key
        self.client = create_client(api_key)
        self.image_path=image_path
        self.initialize_context()
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    def initialize_context(self):
        """Initialize base conversation context with PDF content."""
        self.examples_text = self.extract_example_text()
        self.system_context = "You are a design agent that tells user how to use several visual coding cells to process an input image and extract 3d coordination systems that can help user place visual contents into the image.\n\
            The overall visual design process contains three main steps. First, an input image is processed into a depth map based on image semantics; Second, AI agent extract key object and shapes from the depth space, \
            generating some coordinate systems based on image semantics and deoth information; Thirdly, the extracted coordinate system is rendered in 3D to assist user place visual content into the depth space, while the \
            depth map assist the creation of occlusion effects. In this case, extra visual elements, like text or shapes, can be conveniently placed into the original image with realistic perspective and occlusion, as if \
            they co-exist in the physical space.\n\
            Your role is an agent in the second step. Each time you are given an image to parse, you will receive the image itself, and also several textual examples of relevant designs as design description and visual coding.\
            Your job is to parse the image and the examples, then generate a list of visual codings in similar format, which guides the coordinate system extraction.\n\
            Here are the available visual coding cells:\n\
            Text2Mask(image=IMAGE, text=\"PROMPT\") This cell extracts a certain object from the given image with a text prompt that you provide. When using it, you should replace the PROMPT with a descriptive text of the image part you want to select.\n\
            Mask2Mesh(depth=DEPTH, mask=MASK) This cell extracts the depth mesh of a certain image part with a mask given. When using, you don't need to replace DEPTH. But do replace MASK with the mask variable gained in prior steps.\n\
            Mesh2Plane(mesh=MESH) This cell fits a plane to the given mesh. Please remember to replace the MESH with the certain mesh variable gained in prior steps.\n\
            Mesh2Box(mesh=MESH) This cell fits a box to the given mesh. Please remember to replace the MESH with the certain mesh variable gained in prior steps.\n\
            Mesh2Sphere(mesh=MESH) This cell fits a sphere to the given mesh. Please remember to replace the MESH with the certain mesh variable gained in prior steps.\n\
            Mesh2Cylinder(mesh=MESH,axis=DIRECTION) This cell fits a cylinder to the given mesh, with a given direction serving as central axis direction. Please remember to replace the MESH and DIRECTION with certain variables gained in prior steps.\n\
            SkeletonExtraction(depth=DEPTH, mask=MASK) This cell extracts the skeleton of a human figure. Please replace the MASK with the corrsponding mask variable gained in prior steps\n\
            FaceExtraction(depth=DEPTH, mask=MASK) This cell extracts the face of a human figure. Please replace the MASK with the corrsponding mask variable gained in prior steps\n\
            Here are the geometry types that you can gain from these visual coding steps:\n\
            Mask: An image mask selected from the input image.\n\
            Mesh: A depth mesh selected from the depth space with given mask.\n\
            Plane: A 3D plane located in the depth space. It can yield its norm direction by calling DIRECTION=Plane.norm \n\
            Box: A 3D box located in the depth space. It can yield three directions by calling DIRECTIONs=Box.directions \n\
            Cylinder: A 3D cylinder located in the depth space.\n\
            Sphere: A 3D sphere located in the depth space.\n\
            Skeleton: A human skeleton, comprised of 2D and 3D locations of all key joints, in the depth space. It can yield two directions, one as anterior (skeleton.anterior), one as cranial (skeleton.cranial).\n\
            Face: A human face, comprised of 2D and 3D locations of all key points, in the depth space. It can yield two directions, one as anterior (skeleton.anterior), one as cranial (skeleton.cranial).\n\
            Direction: A direction, which is a 3D vector in the depth space. It can be yielded from the above listed geometry.\n\
            Here are the types of output coordinate system that your visual coding should aim for and end up with:\n\
            Planar: a plane coordinate system. It takes a geometry and an optional direction to construct. E.g. PLANAR_0=planar(plane=PLANE_0, direction=DIRECTION_0) or PLANAR_0=planar(box=BOX)\n\
            Cylindrical: a cylinder coordinate system. It takes a geometry and an optional direction to construct. E.g. CYLINDRICAL_0=cylindrical(cylinder=CYLINDER_0, direction=DIRECTION_0)\n\
            Spherical: a spherical coordinate system. It takes a geometry and an optional direction to construct. E.g. SPHERICAL_0=spherical(sphere=SPHERE_0)\n\
            After all these background knowledge, here's some examples of a visual coding that you will see. Please try to follow the formatting.\n"
        self.system_context += self.examples_text 
        
    def extract_example_text(self):
        example_dir='VisualCodingExamples'
        content=""
        with open(os.path.join(example_dir, 'annotations.txt'), 'r') as file:
            content+=file.read()
        return content
    def send_image_with_prompt(self, image_path, text_prompt):
        """
        Sends an image and a text prompt to GPT for parsing.

        :param image_path: Path to the image file to send.
        :param text_prompt: Text prompt to guide GPT's parsing of the image.
        :param additional_params: Additional parameters for the request.
        :return: The API response in JSON format.
        """
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        base64_image = encode_image(image_path)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.system_context + "\n" + text_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
        )
        #print(response.choices[0])
        self.parse_image_result(response)
        return response

    def parse_image_result(self, response):
        """
        Parses the image analysis result into structured data.

        :param response: The raw API response.
        :return: Structured results (e.g., text, description, or tags).
        """
        # Extract the relevant information from the response
        # This is a placeholder for actual parsing logic
        # For example, you might want to extract text or specific data points
        self.GPT_JSON = response.choices[0].message.content
        print(self.GPT_JSON)