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
class DepthScape:
    #This class implements all crucial image info of a given image and turn it into a DepthScape
    #Taking an image as input, this class implements most essential parsing functions to create proposal element placements
    def __init__(self, id ,image_dir, MOGE_model, device, save_mesh_directory) -> None:
        #Some fast and simple image processing
        self.id = id
        self.MOGE_model = MOGE_model
        self.device = device
        self.save_mesh_directory=save_mesh_directory
        self.image_dir = image_dir
        self.image = cv2.cvtColor(cv2.imread(str(image_dir)), cv2.COLOR_BGR2RGB)
        original_height, original_width = self.image.shape[:2]
        self.resize_img()
        height, width = self.image.shape[:2]
        self.height = height
        self.width = width
        self.original_height = original_height
        self.original_width = original_width
        self.scale = height/original_height
        self.glb_directory=None
        self.finished=False
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
        image_tensor = torch.tensor(self.image / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        # Inference 
        output = self.MOGE_model.infer(image_tensor)
        points, depth, mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()
        self.points=points
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
        self.fov=max_fov_y

    def ask_GPT(self):
        #Ask GPT to parse the image context and generate some visual coding proposals. 
        #self.GPT=GPT(self.image_dir)
        #self.visual_coding_texts=self.GPT.ask()
        #for visual_coding in self.visual_coding_texts:
        #    self.parse_visual_coding(visual_coding)
        api_key = "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"
        self.GPT=GPT(api_key,self.image_dir)
        self.GPT.send_image_with_prompt(self.image_dir, "Please parse the image and generate one visual coding proposal. Make sure you only send the visual coding proposal and nothing else.")
        pass
    def parse_visual_coding(self,visual_coding):
        pass
    def get_GPT_JSON(self):
        #for JSON_results in GPT.JSON_results:
        #    if not JSON_results.sent:
        #        JSON_results.sent=True
        #        return JSON_results
        return None