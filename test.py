from flask import Flask, request, jsonify
from PIL import Image
import io
import csv
import torch
import socket
import pandas as pd
import shutil
from pathlib import Path
import requests

max_bbsize = 0
max_image_id = None
final_xmin = 0
final_xmax = 0
final_ymin = 0
final_ymax = 0

RPI_IP = "192.168.4.4"
URL = f"http://{RPI_IP}:8000/upload"

HOST = "192.168.4.30"
PORT = 8000

app = Flask(__name__)

# Load YOLOv5 model
model_path = Path("C:\\Users\\muhdi\\OneDrive\\Documents\\wk8.pt")
model = torch.hub.load("ultralytics/yolov5:master", "custom", path = model_path)

# Endpoint to receive file uploads
@app.route('/upload', methods=['POST'])
def upload_image():
    image_data = request.files['image'].read()
    img = Image.open(io.BytesIO(image_data))
    
    global max_bbsize
    global max_image_id
    global final_xmin
    global final_xmax 
    global final_ymin 
    global final_ymax 

    # Run inference
    results = model(img, size=640)
    results.print()
    results.save()
    print(results.pandas().xyxy[0])
    preds = results.pandas().xyxy[0]
    cols = ['xmin','ymin','xmax','ymax','confidence','class','name']
    df: pd.DataFrame = pd.DataFrame(preds, columns=cols)
    df.to_csv('yolopreds.csv')
    
    with open('yolopreds.csv', mode = 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            xmin_value = row['xmin']
            ymin_value = row['ymin']
            xmax_value = row['xmax']
            ymax_value = row['ymax']
            image_id = row['name']
        
            width = float(xmax_value) - float(xmin_value)
            height = float(ymax_value) - float(ymin_value)
            
            bbox_size = width * height
            
            if(bbox_size > max_bbsize):
                final_xmin = xmin_value
                final_xmax = xmax_value
                final_ymin = ymin_value
                final_ymax = ymax_value
                max_bbsize = bbox_size
                max_image_id = image_id
                
                print(final_xmin)
                
            if image_id == "bullseye":
                continue
        
            #print(bbox_size)
            #print(image_id)
            
    print()
    print(max_bbsize)
    print(max_image_id)
    print(final_xmin)
    print(final_xmax)
    print(final_ymin)
    print(final_ymax)
    
    msg_str: str = ";".join([max_image_id,
    final_xmin,
    final_xmax,
    final_ymin,
    final_ymax,])
    
    msg_str.removeprefix(";")
    data = msg_str.encode("utf-8")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        
        server_socket.listen()
        print("Server is listening")
        
        conn, addr = server_socket.accept()
        print(f"Connected to {addr}")
        
        with conn:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                print("Received:", data.decode())
                
                conn.sendall(data)    
    
    
    #detected_image_path = r"C:\Users\muhdi\OneDrive\Documents\GitHub\yolov5-flask-object-detection\runs\detect"
    
    #server_url = "http://192.168.4.4:8000/upload"
    # fileNames = ["exp"].extend(["exp"+str(i) for i in range(2,9)])
    # print(fileNames)
    # print(type(fileNames))
    #fileNames = ["exp", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8"]
    
    #with open(f"{detected_image_path}\\{fileNames[0]}\\image0.jpg", "rb+") as file:
        #files = {"file": file}
        #response = requests.post(server_url, files=files)
    # except:
    #     print(fileNames)
    #     print(type(fileNames))
    
    #print(response)
    #if response.status_code == 200:
        #print("Detected image uploaded successfully.")
    #else:
        #print("Failed to upload detected image. Status code:", response.status_code) 
        
    # Return inference results
    # return jsonify(results.pandas().xyxy[0].to_dict(orient='records'))
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)