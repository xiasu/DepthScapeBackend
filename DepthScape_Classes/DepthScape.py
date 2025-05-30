from moge.model import MoGeModel
from moge.utils.vis import colorize_depth
import utils3d
import cv2
import trimesh
import trimesh.visual
from PIL import Image
import torch
import numpy as np
from .GPT import GPT
from .VisualCoding import VisualCoding
from .VisualCodingBlocks import Text2Mask, Mask2PointCloud, PointCloud2Line, PointCloud2Plane, PointCloud2Cylinder, PointCloud2Sphere, FaceExtraction, SkeletonExtraction
from .CoordinateSystems.Spherical import Spherical
from .CoordinateSystems.Cylindrical import Cylindrical
from .CoordinateSystems.Planar import Planar
from .Geometry.PointCloud import PointCloud
import dill
import os
import json
import traceback
import time
class DepthScape:
    #This class implements all crucial image info of a given image and turn it into a DepthScape
    #Taking an image as input, this class implements most essential parsing functions to create proposal element placements
    def __init__(self, id ,image_dir, MOGE_model, device, save_mesh_directory) -> None:
        #Some fast and simple image processing
        self.start_time = time.time()
        self.id = id
        self.MOGE_model = MOGE_model
        self.device = device
        self.save_mesh_directory=save_mesh_directory
        self.image_dir = image_dir
        self.image = cv2.cvtColor(cv2.imread(str(image_dir)), cv2.COLOR_BGR2RGB)
        original_height, original_width = self.image.shape[:2]
        self.resize_img()
        resized_image_path = self.image_dir.replace('.jpg', '_resized.jpg')
        cv2.imwrite(resized_image_path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        self.resized_image_path = resized_image_path
        height, width = self.image.shape[:2]
        self.height = height
        self.width = width
        self.original_height = original_height
        self.original_width = original_width
        self.scale = height/original_height
        self.glb_directory=None
        self.finished=False
        self.GPT_JSON_results=[]
        self.output_coordinate_systems=[]
        self.processing_results = []  # Store processing results including errors
        self.timing = {
            "total_time": 0,
            "moge_time": 0,
            "gpt_time": 0,
            "masking_times": {},
            "geometry_processing_times": {}
        }
        print('DepthScape initialized')
    def resize_img(self):
        max_size=1080
        height, width = self.image.shape[:2]
        if max(height, width) <= max_size:
            return
        if height > width:
            new_height = max_size
            new_width = int(width * max_size / height)
        else:
            new_width = max_size
            new_height = int(height * max_size / width)
        self.image = cv2.resize(self.image, (new_width, new_height))
    def process_image(self):
        #The time consuming part of image processing
        print('Processing image')
        self.conduct_MoGe()
        #self.ask_GPT()
    def conduct_MoGe(self,threshold = 0.02):
        moge_start_time = time.time()
        image_tensor = torch.tensor(self.image / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        # Inference 
        output = self.MOGE_model.infer(image_tensor)
        points, depth, mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()
        self.points=points
        #self.pointCloud=PointCloud(points)
        self.depth=depth
        self.mask=mask
        self.intrinsics=intrinsics                                                                                                  
        faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                points,
                self.image.astype(np.float32) / 255,
                utils3d.numpy.image_uv(width=self.width, height=self.height),
                mask=mask & ~utils3d.numpy.depth_edge(depth, rtol=threshold, mask=mask),
                tri=True
            )
                # When exporting the model, follow the OpenGL coordinate conventions:
                # - world coordinate system: x right, y up, z backward.
                # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
        vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
        self.mesh=trimesh.Trimesh(
                    vertices=vertices,
                    vertex_colors=vertex_colors,
                    faces=faces, 
                    process=False
                ) #This is the mesh used to conduct geometry extraction
        # trimesh.Trimesh(
        #     vertices=vertices, 
        #     faces=faces, 
        #     visual = trimesh.visual.texture.TextureVisuals(
        #         uv=vertex_uvs, 
        #         material=trimesh.visual.material.PBRMaterial(
        #             baseColorTexture=Image.fromarray(self.image),
        #             metallicFactor=0.5,
        #             roughnessFactor=1.0
        #         )
        #     ),
        #     process=False
        # ).export(self.save_mesh_directory + '/' + self.id + '.glb')
        # self.glb_directory=self.save_mesh_directory + '/' + self.id + '.glb'
        # print('MoGe conducted. GLB saved at ' + self.glb_directory)

        faces_full, vertices_full, vertex_colors_full, vertex_uvs_full = utils3d.numpy.image_mesh(
                points,
                self.image.astype(np.float32) / 255,
                utils3d.numpy.image_uv(width=self.width, height=self.height),
                mask=mask & ~utils3d.numpy.depth_edge(depth, rtol= float('inf'), mask=mask),
                tri=True
            )
                # When exporting the model, follow the OpenGL coordinate conventions:
                # - world coordinate system: x right, y up, z backward.
                # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
        vertices_full, vertex_uvs_full = vertices_full * [1, -1, -1],  vertex_uvs_full * [1, -1] + [0, 1]
        # self.mesh_full=trimesh.Trimesh(
        #             vertices=vertices_full,
        #             vertex_colors=vertex_colors_full,
        #             faces=faces_full, 
        #             process=False
        #         ) 
        trimesh.Trimesh(
            vertices=vertices_full, 
            faces=faces_full, 
            visual = trimesh.visual.texture.TextureVisuals(
                uv=vertex_uvs_full, 
                material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=Image.fromarray(self.image),
                    metallicFactor=0.5,
                    roughnessFactor=1.0
                )
            ),
            process=False
        ).export(self.save_mesh_directory + '/' + self.id + '.glb') #This is the mesh being sent to server
        self.glb_directory=self.save_mesh_directory + '/' + self.id + '.glb'
        print('MoGe conducted. GLB saved at ' + self.glb_directory)
        #Use the points to calculate the FOV
        # Compute FOV for each point
        max_fov_x, max_fov_y = 0, 0
        key_points=[points[0][0],points[0][-1],points[-1][0],points[-1][-1]]
        for point in key_points:
            x, y, z = point
            # Ignore points behind the camera (negative z)
            if z <= 0:
                continue
            # Calculate vertical and horizontal angles
            fov_y = 2 * np.arctan(np.abs(y) / z)
            fov_x = 2 * np.arctan(np.abs(x) / z)
            # Update max FOV values
            max_fov_y = max(max_fov_y, fov_y)
            max_fov_x = max(max_fov_x, fov_x)
        print("FOV x:", max_fov_x)
        print("FOV y:", max_fov_y)
        if max_fov_y == 0:
            #This means the image has plenty of sky so the corners are not reconstructed in the mesh. In this case, we can iterate through the image points to find the maximum FOV
            max_fov_y = 0
            max_y_2d = 0.5
            max_point = None
            height, width = self.image.shape[:2]
            print("Height:", height, "Width:", width)
            for i, point in enumerate(self.points.reshape(-1, 3)):
                x, y, z = point
                if z <= 0:
                    continue
                fov_y = 2 * np.arctan(np.abs(y) / z)
                y_2d =np.abs(0.5-(i // width) / height)
                if fov_y > max_fov_y:
                    max_fov_y = fov_y
                    max_y_2d = y_2d
                    max_point = point
            if max_y_2d < 0.5:
                scaling_factor = 0.5 / max_y_2d
                max_fov_y = 2 * np.arctan(np.abs(max_point[1]) * scaling_factor / max_point[2])
                print("Adjusting fov y by a factor of", scaling_factor)
            print("FOV y adjusted:", max_fov_y)
        self.fov=max_fov_y
        self.timing["moge_time"] = time.time() - moge_start_time

    def ask_GPT(self):
        #Ask GPT to parse the image context and generate some visual coding proposals. 
        #self.GPT=GPT(self.image_dir)
        #self.visual_coding_texts=self.GPT.ask()
        #for visual_coding in self.visual_coding_texts:
        #    self.parse_visual_coding(visual_coding)
        gpt_start_time = time.time()
        with open('openAI-key.txt', 'r') as file:
            api_key = file.read().strip()
        self.GPT = GPT(api_key, self.image_dir)
        prompt="Please parse the image and generate all potentially relevant visual coding proposal. Make sure your reply is in json format, which contains a list of visual coding proposal. \
            Each proposal include a name, a description, and also the visual_code, which is a list of code lines. Give me the visual coding for all key objects you can find in the entire image.\
                 You should also suggest all relevant types of visual codings for one object. For example, for a human figure, you can suggest a cylindrical visual coding to surround the body, or the skeletal or facial planes to track their body.\
                    For the skeletal and facial planes, you should suggest both the median and the frontal planes, but in different visual codings, and differentiate them in the name.\
                         Remember that even though you may be suggesting facial planes, you should still provide text prompt in the Text2Mask cell for the entire human figure, or the extraction may fail. \
                    For a square object, you may suggest a planar visual coding, and also an extruded plane visual coding. \
                        Remember to ensure that each visual coding only have one coordinate system as output."
        with open('VisualCodingExamples/GPT_Response_Example.txt', 'r') as file:
            prompt+=file.read()
        self.GPT_JSON_results=self.GPT.send_image_with_prompt(self.image_dir, prompt)
        self.timing["gpt_time"] = time.time() - gpt_start_time
        #Then execute all the collected visual coding
        for vc in self.GPT_JSON_results:
            
            try:
                result = self.conduct_visual_coding(vc)
                processing_result = {
                    "visual_coding": vc,
                    "status": "success",
                    "result": result
                }
            except Exception as e:
                error_traceback = traceback.format_exc()
                print(f"Error conducting visual coding: {e}")
                print(error_traceback)
                processing_result = {
                    "visual_coding": vc,
                    "status": "error",
                    "error_message": str(e),
                    "traceback": error_traceback,
                    "result": None
                }
                result = None
                
            self.processing_results.append(processing_result)
                
            if isinstance(result, list):
                for item in result:
                    self.output_coordinate_systems.append(item)
            elif result is not None:
                self.output_coordinate_systems.append(result)
            else:
                pass
        
        # Save the processing results to a JSON file
        self.save_processing_results()
            #self.output_coordinate_systems.append(self.conduct_visual_coding(vc))
            
    def save_processing_results(self):
        """Save all visual coding and processing results to a JSON file"""
        self.timing["total_time"] = time.time() - self.start_time
        results_path = os.path.join(self.save_mesh_directory, f'{self.id}_processing_results.json')
        
        # Convert processing results to a serializable format
        serializable_results = []
        for result in self.processing_results:
            serializable_result = {
                "visual_coding": result["visual_coding"].__dict__,
                "status": result["status"],
            }
            
            if result["status"] == "error":
                serializable_result["error_message"] = result["error_message"]
                serializable_result["traceback"] = result["traceback"]
            
            # Add timing information for this visual coding
            vc_name = result["visual_coding"].name
            if vc_name in self.timing["geometry_processing_times"]:
                serializable_result["processing_time"] = self.timing["geometry_processing_times"][vc_name]
            
            # Add masking time if available
            if vc_name in self.timing["masking_times"]:
                serializable_result["masking_time"] = self.timing["masking_times"][vc_name]
            
            serializable_results.append(serializable_result)
            
        # Create the output dictionary
        output_data = {
            "id": self.id,
            "image_path": self.image_dir,
            "timing": {
                "total_time": self.timing["total_time"],
                "moge_time": self.timing["moge_time"],
                "gpt_time": self.timing["gpt_time"]
            },
            "gpt_json_results": [result.__dict__ for result in self.GPT_JSON_results],
            "processing_results": serializable_results
        }
        
        # Save to file
        with open(results_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"Processing results saved to {results_path}")
        
    def conduct_visual_coding(self,visual_coding):
        #This function parses the visual coding and execute it
        vc_start_time = time.time()
        variables = {}
        def parse_visual_code_line(line_text):
            # Parse the line to extract output name, function name, and parameters
            output_name = line_text.split('=')[0].strip()
            func_and_params = line_text.split('=', 1)[1].strip()
            func_name = func_and_params.split('(')[0].strip()
            params_str = func_and_params[func_and_params.index('(') + 1:func_and_params.index(')')]
            parameters = [param.strip() for param in params_str.split(',')]
            return output_name, func_name, parameters
        def parse_variable(variable_text):
            #Parse this variable and see if it's a compound type and need to be specially addressed
            # a compound type will be marked with a dot
            if '.' in variable_text:
                # Split the variable into its components
                parts = variable_text.split('.')
                # Retrieve the value from the variables dictionary
                try:
                    base = variables[parts[0]]
                except Exception as e:
                    print(f"Error getting variable {parts[0]}: {e}")
                    return None
                suffix = parts[1]
                # Traverse the nested attributes
                base_type=parts[0].split('_')[0] #Get the type of the base variable
                if base_type == 'FACE':
                    if suffix == 'median':
                        return base.get_median()
                    elif suffix == 'frontal':
                        return base.get_frontal()
                    elif suffix == 'anterior':
                        return base.get_anterior()
                    elif suffix == 'cranial':
                        return base.get_cranial()
                    else:
                        raise ValueError(f"Unexpected compound variable encountered! {base_type} doesn't support {suffix} as suffix")
                elif base_type == 'SKELETON':
                    if suffix == 'median':
                        return base.get_median()
                    elif suffix == 'frontal':
                        return base.get_frontal()
                    elif suffix == 'anterior':
                        return base.get_anterior()
                    elif suffix == 'cranial':
                        return base.get_cranial()
                    else:
                        raise ValueError(f"Unexpected compound variable encountered! {base_type} doesn't support {suffix} as suffix")
                elif base_type == 'PLANE':
                #In this case, the base object can be a list. To simplify the situation, just return the first base
                    if isinstance(base, list):
                        base = base[0]
                    if suffix == 'normal':
                        return base.normal
                    elif suffix == 'primary':
                        return base.primary
                    elif suffix == 'extruded':
                        return base.extruded
                    else:
                        raise ValueError(f"Unexpected compound variable encountered! {base_type} doesn't support {suffix} as suffix")
                # elif base_type == 'LINE':
                #     if suffix == 'direction':
                #         return base.direction
                #     else:
                #         raise ValueError(f"Unexpected compound variable encountered! {base_type} doesn't support {suffix} as suffix")
                elif base_type == 'CYLINDER':
                    raise ValueError(f"Unexpected compound variable encountered! The {base_type} type is not supported yet.")
                elif base_type == 'SPHERE':
                    raise ValueError(f"Unexpected compound variable encountered! The {base_type} type is not supported yet.")
                else:
                    raise ValueError(f"Unexpected compound variable encountered! {base_type} is not supported yet.")
            else:
                return variables[variable_text]
        
        for line in visual_coding.visual_code:
            output_name, func_name, parameters = parse_visual_code_line(line)
            if func_name == 'Text2Mask':
                mask_start_time = time.time()
                text_prompt = parameters[0].split('=')[1].strip().strip('"')
                mask = Text2Mask.Text2Mask(self,text_prompt)
                if mask is None:
                    return None
                mask_image = (mask.mask).astype(np.uint8)
                mask_image_path =self.image_dir.replace('.jpg', f'_masked_{text_prompt}.jpg')
                cv2.imwrite(mask_image_path, mask_image)
                print("Mask saved at " + mask_image_path)
                variables[output_name] = mask
                mask_time = time.time() - mask_start_time
                if visual_coding.name not in self.timing["masking_times"]:
                    self.timing["masking_times"][visual_coding.name] = mask_time
                else:
                    self.timing["masking_times"][visual_coding.name] += mask_time
            elif func_name == 'Mask2Mesh':
                input_mask = parse_variable(parameters[0].split('=')[1].strip())
                pc= Mask2PointCloud.Mask2PointCloud(self,input_mask)
                variables[output_name] = pc
            elif func_name == 'Mesh2Plane':
                input_pc =  parse_variable(parameters[0].split('=')[1].strip())
                plane = PointCloud2Plane.PointCloud2Plane(input_pc)
                variables[output_name] = plane
            elif func_name == 'Mesh2Line':
                input_pc =  parse_variable(parameters[0].split('=')[1].strip())
                line = PointCloud2Line.PointCloud2Line(input_pc)
                variables[output_name] = line
            elif func_name == 'Mesh2Sphere':
                input_pc =  parse_variable(parameters[0].split('=')[1].strip())
                sphere = PointCloud2Sphere.PointCloud2Sphere(input_pc)
                variables[output_name] = sphere
            elif func_name == 'Mesh2Cylinder':
                # Note that this line have two params. CYLINDER_0=Mesh2Cylinder(mesh=MESH_0, direction=NULL)
                input_pc =  parse_variable(parameters[0].split('=')[1].strip())
                input_direction = None if parameters[1].split('=')[1].strip() == 'NULL' else parse_variable(parameters[1].split('=')[1].strip())
                cylinder = PointCloud2Cylinder.PointCloud2Cylinder(input_pc,input_direction)
                variables[output_name] = cylinder
            elif func_name == 'SkeletonExtraction':
                input_mask =  parse_variable(parameters[0].split('=')[1].strip())
                skeleton = SkeletonExtraction.SkeletonExtraction(self,input_mask)
                variables[output_name] = skeleton
            elif func_name == 'FaceExtraction':
                input_mask =  parse_variable(parameters[0].split('=')[1].strip())
                face = FaceExtraction.FaceExtraction(self,input_mask)
                variables[output_name] = face
            elif func_name == 'Planar':
                #There are two cases. When given a plane and a direction, and when given two directions. Planar(plane = FACE_0.median, direction = NULL) || PLANAR=Planar(direction_1 = PLANE_0.normal, direction_2 = LINE_0.direction)
                if 'plane' in parameters[0]: #and 'direction' in parameters[1]:
                    base_plane= parse_variable(parameters[0].split('=')[1].strip())
                    #base_direction_1= None if parameters[1].split('=')[1].strip() == 'NULL' else parse_variable(parameters[1].split('=')[1].strip())
                    #base_direction_2 = None
                    if isinstance(base_plane, list):
                        planars=[]
                        for plane in base_plane:
                            #planars.append(Planar(plane, base_direction_1 , base_direction_2,visual_coding,variables))
                            planars.append(Planar(plane, visual_coding,variables))
                        return planars
                    else:
                        #return Planar(base_plane, base_direction_1 , base_direction_2,visual_coding,variables)
                        return Planar(base_plane, visual_coding,variables)
                # elif 'direction_1' in parameters[0] and 'direction_2' in parameters[1]:
                #     base_plane = None
                #     base_direction_1= parse_variable(parameters[0].split('=')[1].strip())
                #     base_direction_2= parse_variable(parameters[1].split('=')[1].strip())
                #     return Planar(base_plane, base_direction_1, base_direction_2,visual_coding,variables)
                else:
                    raise ValueError(f"Unexpected parameters for Planar function: {parameters}")
            elif func_name == 'Cylindrical':
                #CYLINDRICAL_0=Cylindrical(cylinder=CYLINDER_0, direction=NULL)
                base_cylinder= parse_variable(parameters[0].split('=')[1].strip())
                #base_direction= None if parameters[1].split('=')[1].strip() == 'NULL' else parse_variable(parameters[1].split('=')[1].strip())
                return Cylindrical(base_cylinder, visual_coding,variables)
            elif func_name == 'Spherical':
                #SPHERICAL_0=Spherical(sphere=SPHERE_0, direction=NULL)
                base_sphere= parse_variable(parameters[0].split('=')[1].strip())
                #base_direction= None if parameters[1].split('=')[1].strip() == 'NULL' else parse_variable(parameters[1].split('=')[1].strip())
                return Spherical(base_sphere, visual_coding,variables)
            else:
                raise ValueError(f"Unknown function name: {func_name}")
        
        # Record the total geometry processing time for this visual coding
        geometry_time = time.time() - vc_start_time
        # Subtract masking time if any
        if visual_coding.name in self.timing["masking_times"]:
            geometry_time -= self.timing["masking_times"][visual_coding.name]
        self.timing["geometry_processing_times"][visual_coding.name] = geometry_time
        
        return None
    
    def get_result_JSON(self):
        #This function returns the JSON result of the GPT parsing and geometric extraction
        result = {
            "coordinate_systems": [coordinate_system.json for coordinate_system in self.output_coordinate_systems]
        }
        # Save coordinate systems as pickle files
        
        # Create directory if it doesn't exist
        # pickle_dir = os.path.join(self.save_mesh_directory, 'coordinate_systems')
        # os.makedirs(pickle_dir, exist_ok=True)
        
        # # Save each coordinate system
        # for i, coord_sys in enumerate(self.output_coordinate_systems):
        #     pickle_path = os.path.join(pickle_dir, f'coordinate_system_{i}.pkl')
        #     with open(pickle_path, 'wb') as f:
        #         dill.dump(coord_sys, f)
        return result