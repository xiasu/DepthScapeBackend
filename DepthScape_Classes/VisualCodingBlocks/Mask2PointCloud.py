from ..Geometry import PointCloud
from ..Geometry import Mask
import utils3d
import numpy as np
def Mask2PointCloud(depthScape,mask):
    mask_boolean = mask > 0
    depth_mask=np.logical_not(utils3d.numpy.depth_edge(depthScape.depth, rtol=0.02, mask=depthScape.mask))
    combined_mask=np.logical_and(mask_boolean, depth_mask)
    combined_mask_indices = np.argwhere(combined_mask > 0)
    selected_points = depthScape.points[combined_mask_indices[:, 0], combined_mask_indices[:, 1]]
    return PointCloud(selected_points)
