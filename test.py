from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import shutil
from pathlib import Path
import requests

RPI_IP = "192.168.4.4"
URL = f"http://{RPI_IP}:8000/upload"

app = Flask(__name__)

# Load YOLOv5 model
model_path = Path("C:\\Users\\muhdi\\OneDrive\\Documents\\wk8.pt")
model = torch.hub.load("ultralytics/yolov5:master", "custom", path = model_path)

# Endpoint to receive file uploads
@app.route('/upload', methods=['POST'])
def upload_image():
    image_data = request.files['image'].read()
    img = Image.open(io.BytesIO(image_data))
    
    # Run inference
    results = model(img, size=640)
    results.print()
    results.save()
    
    detected_image_path = "C:\\Users\\muhdi\\OneDrive\\Documents\\GitHub\\yolov5-flask-object-detection\\runs\\detect\\exp"
    
    server_url = "http://192.168.4.4:5000/upload"
    
    with open(detected_image_path, "r+") as file:
        files = {"file": file}
        response = requests.post(server_url, files=files)
    
    if response.status_code == 200:
        print("Detected image uploaded successfully.")
    else:
        print("Failed to upload detected image. Status code:", response.status_code) 
        
    # Return inference results
    return jsonify(results.pandas().xyxy[0].to_dict(orient='records'))
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)