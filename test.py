from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
from pathlib import Path

app = Flask(__name__)

# Load YOLOv5 model
model_path = Path("C:\\Users\\muhdi\\OneDrive\\Documents\\wk8.pt")
model = torch.hub.load("ultralytics/yolov5:master", "custom", path = model_path)

# Endpoint to receive file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Run inference
        results = model(img, size=640)
        results.print()
        results.save()
        
        # Return inference results
        return jsonify(results.pandas().xyxy[0].to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)