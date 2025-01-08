from flask import Flask, request, jsonify, send_from_directory
import os
import uuid
from PIL import Image  # Example image processing library
import json
from celery import Celery
from DepthScape_Classes import *

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'Server_Data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device('cuda')
pretrained_model_name_or_path='Ruicheng/moge-vitl'
model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure Celery
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# In-memory status tracking (you can use a database for persistence)
statuses = {}
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    unique_id = str(uuid.uuid4())
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_id)
    os.makedirs(save_path, exist_ok=True)
    
    # Save the original image
    image_path = os.path.join(save_path, file.filename)
    file.save(image_path)
    
    # Initialize status
    statuses[unique_id] = "pending"

    # Start processing in the background
    process_image.delay(unique_id, image_path, save_path)

    return jsonify({"id": unique_id}), 200

@celery.task
def process_image(unique_id, image_path, save_path):
    try:
        statuses[unique_id] = "processing"
        # image processing as DepthScape
        depthScape=DepthScape(unique_id ,image_path, model, device, save_path)
        statuses[unique_id] = "completed"
    except:
        pass

@app.route('/status/<image_id>', methods=['GET'])
def check_status(image_id):
    if image_id not in statuses:
        return jsonify({"error": "Invalid ID"}), 404
    return jsonify({"id": image_id, "status": statuses[image_id]}), 200
 
@app.route('/glb/<image_id>', methods=['GET'])
def get_result(image_id):
    if statuses.get(image_id) != "completed":
        return jsonify({"error": "Processing not completed or invalid ID"}), 400
    glb_dir = f"{UPLOAD_FOLDER}/{image_id}"
    return send_from_directory(glb_dir, f"{image_id}.glb")

@app.route('/result/<image_id>/<filename>', methods=['GET'])
def download_file(image_id, filename):
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], image_id)
    return send_from_directory(save_path, filename)

if __name__ == '__main__':
    app.run(debug=True)
