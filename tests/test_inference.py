import io
import torch
import os 
from PIL import Image
from pathlib import Path
# Model
model_path = Path("C:\\Users\\muhdi\\OneDrive\\Documents\\wk8.pt")
model = torch.hub.load("ultralytics/yolov5:master", "custom", path = model_path)

img = Image.open("20240313_163902.jpg")  # PIL image direct open
# Read from bytes as we do in app
with open("20240313_163902.jpg", "rb") as file:
    img_bytes = file.read()
img = Image.open(io.BytesIO(img_bytes))

results = model(img, size=640)  # includes NMS

print(results.pandas().xyxy[0])
