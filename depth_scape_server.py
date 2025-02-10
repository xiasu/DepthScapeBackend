from flask import Flask, request, jsonify, send_from_directory
import os
import uuid
from PIL import Image  # Example image processing library
import json
#from celery import Celery
from DepthScape_Classes import *
import torch
from moge.model import MoGeModel
from moge.utils.vis import colorize_depth
import utils3d
from flask_cors import CORS
#import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
app = Flask(__name__)
CORS(app)
# Configure upload folder
UPLOAD_FOLDER = 'Server_Data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
executor = ThreadPoolExecutor()
device = torch.device('cuda')
pretrained_model_name_or_path='Ruicheng/moge-vitl'
model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure Celery
#app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
#app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
#celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
#celery.conf.update(app.config)

# In-memory status tracking (you can use a database for persistence)
open_spaces = {}
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        print("No image provided")
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    unique_id = str(uuid.uuid4())
    print(unique_id)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_id)
    os.makedirs(save_path, exist_ok=True)    
    
    # Save the original image
    image_path = os.path.join(save_path, file.filename)
    file.save(image_path)
    #depthScape=DepthScape(unique_id ,image_path, model, device, save_path)
    # Use a dict to save this image's processing status
    open_spaces[unique_id] = None

    # Start processing in the background
    #process_image.delay(unique_id ,image_path,save_path)
    #asyncio.create_task(process_image(unique_id, image_path, save_path))
    #executor.submit(process_image, unique_id, image_path, save_path)
    process_image(unique_id, image_path, save_path)
    #executor.submit(ask_gpt, unique_id)
    ask_gpt(unique_id)
    return jsonify({"id": unique_id,"fov": open_spaces[unique_id].fov,"results": open_spaces[unique_id].get_result_JSON()}), 200

#@celery.task
def process_image(unique_id ,image_path,save_path):
    
    # image processing as DepthScape
    depthScape=DepthScape(unique_id ,image_path, model, device, save_path)
    # Use a dict to save this image's processing status
    open_spaces[unique_id] = depthScape
    print("Processing image "+depthScape.id)
    depthScape.process_image()
    print(depthScape.intrinsics)
    fy=depthScape.intrinsics[1,1]
    fov=2*np.arctan(depthScape.height/(2*fy))
    print("FOV y:",fov)
    

def ask_gpt(unique_id):
    print("Asking GPT")
    depthScape = open_spaces[unique_id]
    depthScape.ask_GPT()

# @app.route('/stream/<image_id>', methods=['GET'])
# def stream_image_processing_results(image_id):
#     if image_id not in open_spaces:
#         return jsonify({"error": "Invalid ID"}), 404
#     #depthScape= open_spaces[image_id]
#     #In a total of 60 seconds, check if the corresponding glb file is generated every second. If so, send the url to the glb file with yield.
#     glb_info_sent=False
#     for i in range(60):
#         depthScape= open_spaces[image_id]
#         if depthScape:
#             if depthScape.glb_directory and not glb_info_sent:
#                 yield f'data: {json.dumps({"type": "glb", "message": "GLB ready"})}\n\n'
#                 print("GLB ready info sent")
#                 glb_info_sent=True
#             GPT_JSON = depthScape.get_GPT_JSON()
#             if GPT_JSON:
#                 yield f"data: {json.dumps({'type': 'json', 'data': GPT_JSON})}\n\n"
#             if depthScape.finished:
#                 yield f'data: {json.dumps({"type": "status", "message": "Processing finished"})}\n\n'
#                 break
#         time.sleep(1)

@app.route('/status/<image_id>', methods=['GET'])
def check_status(image_id):
    if image_id not in statuses:
        return jsonify({"error": "Invalid ID"}), 404
    return jsonify({"id": image_id, "status": statuses[image_id]}), 200
 
@app.route('/glb/<image_id>', methods=['GET'])
def get_result(image_id):
    glb_dir = f"{UPLOAD_FOLDER}/{image_id}"
    return send_from_directory(glb_dir, f"{image_id}.glb")

@app.route('/json/<image_id>', methods=['GET'])
def get_GPT_result(image_id):
    #Wait until the GPT result is ready
    # depthScape= open_spaces[image_id]
    # max_wait=60
    # while not depthScape.GPT_JSON:
    #     time.sleep(1)
    #     max_wait-=1
    #     if max_wait==0:
    #         return jsonify({"error": "GPT result expired"}), 404
    # return jsonify(depthScape.GPT_JSON)
    return jsonify({"error": "Not implemented yet"}), 404


if __name__ == '__main__':
    app.run(debug=True, port=5001)
