from ultralytics import YOLO
import torch
# Load a model
# pip install -U ultralytics 
model = YOLO("yolo11s-seg.pt")  
model.train(data="data.yaml",cos_lr=True, epochs=500,imgsz=512,resume=False,pretrained=True,augment=True,
            batch=56,overlap_mask=True,patience=15,cache=True,optimizer='auto',plots=True,exist_ok=True,scale=0.5,
            name='Carton-seg-s')

