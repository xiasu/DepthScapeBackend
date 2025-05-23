#This class leverage openpose to extract skeleton from a given image
import cv2
import mediapipe as mp
from ..Geometry.Skeleton import Skeleton
def SkeletonExtraction(depthScape,mask):
    # Load the image
    image = depthScape.image
    #Apply the mask to the image, so that only the masked area is processed
    image = cv2.bitwise_and(image, image, mask=mask.mask)
    # Save the masked image
    mask_image_path = depthScape.image_dir.replace('.jpg', f'_masked_skeleton.jpg')
    cv2.imwrite(mask_image_path, image)
    print("Masked skeleton image saved at " + mask_image_path)
    mp_pose = mp.solutions.pose
    # Detect the pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        results = pose.process(image)
    skeleton = Skeleton(depthScape,results)
    
    return skeleton