#This class leverage openpose to extract skeleton from a given image
import cv2
import mediapipe as mp
from ..Geometry.Face import Face
def FaceExtraction(depthScape,mask):
    # Load the image
    image = depthScape.image
    #Apply the mask to the image, so that only the masked area is processed
    image = cv2.bitwise_and(image, image, mask=mask)
    # Detect the pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        results = pose.process(image)
    face = Face(depthScape,results)
    return results