import io
import torch
import torchvision
import torchvision.transforms as T
import os 
from PIL import Image
from pathlib import Path
from torchvision.io import read_image

# Model
model_path = Path("C:\\Users\\muhdi\\OneDrive\\Documents\\wk8.pt")
model = torch.hub.load("ultralytics/yolov5:master", "custom", path = model_path)

image_path = "C:/Users/muhdi/OneDrive/Desktop/20240313_163902.jpg"
img = Image.open(image_path)  # PIL image direct open
# Read from bytes as we do in app
with open(image_path, "rb") as file:
    img_bytes = file.read()
img = Image.open(io.BytesIO(img_bytes))

results = model(img, size=640)  # includes NMS
results.print()
results.save()
print(results.pandas().xyxy[0])
image_id = results.pandas().xyxy[0].iloc[0]['name']
print("Image ID: ", image_id)


