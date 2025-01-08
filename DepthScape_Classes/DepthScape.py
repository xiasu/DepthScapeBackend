from moge.model import MoGeModel
from moge.utils.vis import colorize_depth
import utils3d
import cv2
import trimesh
import trimesh.visual
from PIL import Image
import torch

class DepthScape:
    #This class implements all crucial image info of a given image and turn it into a DepthScape
    #Taking an image as input, this class implements most essential parsing functions to create proposal element placements
    def __init__(self, id ,image_dir, MOGE_model, device, save_mesh_directory) -> None:
        self.id = id
        self.MOGE_model = MOGE_model
        self.device = device
        self.save_mesh_directory=save_mesh_directory
        self.image_dir = image_dir
        self.image = cv2.cvtColor(cv2.imread(str(image_dir)), cv2.COLOR_BGR2RGB)
        height, width = self.image.shape[:2]
        self.height = height
        self.width = width
        #Use MoGE to parse image depth
        self.conduct_MoGe()
        #Let GPT parse the image context
                                                                                   
        #Receive and parse the visual coding blocks

        pass

    def conduct_MoGe(self,threshold = 0.02):
        image_tensor = torch.tensor(self.image / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        # Inference 
        output = self.model.infer(image_tensor)
        points, depth, mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()
        self.points=points
        self.depth=depth
        self.mask=mask
        self.intrinsics=intrinsics
         
                # When exporting the model, follow the OpenGL coordinate conventions:
                # - world coordinate system: x right, y up, z backward.
                # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
        vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
        self.mesh=trimesh.Trimesh(
                    vertices=vertices,
                    vertex_colors=vertex_colors,
                    faces=faces, 
                    process=False
                ) 
        trimesh.Trimesh(
            vertices=vertices, 
            faces=faces, 
            visual = trimesh.visual.texture.TextureVisuals(
                uv=vertex_uvs, 
                material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=Image.fromarray(self.image),
                    metallicFactor=0.5,
                    roughnessFactor=1.0
                )
            ),
            process=False
        ).export(self.save_mesh_directory + '/' + self.id + '.glb')
        self.glb_directory=self.save_mesh_directory + '/' + self.id + '.glb'