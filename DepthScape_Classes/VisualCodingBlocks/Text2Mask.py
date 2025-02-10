import numpy as np
import requests
from ..Geometry.Mask import Mask
def Text2Mask(depthScape,text):
    #This function takes a depth scape, which include an image, and also a text, and return a mask of the text
    #The mask is a binary image with the same size as the image, where the text is white and the rest is black
    def decode_mask(mask, height, width):
        return np.array(mask, dtype=np.uint8).reshape((height, width))
    def get_mask(response_json):
        height=depthScape.height
        width=depthScape.width
        # Create a blank canvas for visualization
        mask_canvas = np.zeros((height, width), dtype=np.uint8)

        # Decode RLE masks and overlay on canvas
        for annotation in response_json["annotations"]:
            mask = annotation["segmentation"]
            decoded_mask = decode_mask(mask, height, width)
            mask_canvas = np.maximum(mask_canvas, decoded_mask * 255)
            
        return  Mask(depthScape, text,mask_canvas)
    print(f"Text2Mask:{text}")
    SERVER_URL = "http://127.0.0.1:5000/segment"
    # Send the image to the server
    with open(depthScape.resized_image_path, 'rb') as img_file:
        files = {'image': img_file}
        data = {'prompt': text}

        # Send POST request
        response = requests.post(SERVER_URL, files=files, data=data)

        if response.status_code == 200:
            return get_mask(response.json())
        else:
            print(f"Error occured when parsing mask. Usually it's because the text prompt is not existent, or the segmentation model cannot be that specific.Skipping the next steps in this visual coding.")
            return None
    
    

