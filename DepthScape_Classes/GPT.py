import requests

class GPT:
    def __init__(self, api_key, base_url="https://api.openai.com/v1/"):
        self.api_key = api_key
        self.base_url = base_url

    def send_image_with_prompt(self, image_path, additional_params=None):
        """
        Sends an image and a text prompt to GPT for parsing.

        :param image_path: Path to the image file to send.
        :param text_prompt: Text prompt to guide GPT's parsing of the image.
        :param additional_params: Additional parameters for the request.
        :return: The API response in JSON format.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "multipart/form-data",
        }

        files = {
            "image": open(image_path, "rb"),
        }

        # Prepare data payload
        data = {
            "prompt": "You are a design agent that tells user how to use several visual coding cells to process an input image and extract 3d coordination systems that can help user place visual contents into the image.\n\
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
            After all these background knowledge, here's an example of a visual coding that you will see. Please try to follow the formatting.\n\
            Description of design: This poster design features a cityscape at night, viewed from an elevated perspective. The image showcases towering buildings with glowing windows, illuminated streets, and a dynamic urban atmosphere. Superimposed over the city scene is bold white text that reads \"NO OTHER NAME.\" The text appears to integrate with the perspective of the buildings, giving it a three-dimensional effect as if it were floating within the cityscape. The overall design combines urban aesthetics with a striking typographical element, creating a visually engaging and dramatic composition.\n\
            Visual coding:\n\
            MASK_0=Text2Mask(image=IMAGE, prompt=”the front building in the input image”)\n\
            MESH_0=Mask2Mesh(depth=DEPTH, mask=MASK_0)\n\
            BOX_0=Mesh2Box(mesh=MESH_0)\n\
            PLANAR_0=Planar(Box=BOX_0)\n\
            " 
        }

        # Include additional parameters if provided
        if additional_params:  
            data.update(additional_params)

        # API endpoint for image+text analysis (hypothetical endpoint)
        url = self.base_url + "images/analyze_with_prompt"

        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()

    def parse_image_result(self, response):
        """
        Parses the image analysis result into structured data.

        :param response: The raw API response.
        :return: Structured results (e.g., text, description, or tags).
        """
        