from ultralytics import YOLO
import cv2
import numpy as np
import random
import os

# load model 
model = YOLO('training/Carton-seg-s/weights/best.pt')

load_img = cv2.imread('training/test/net (9125).jpg')

os.makedirs('visualization', exist_ok=True)
# 
results = model.predict(source=load_img,conf = 0.6)
# Make a copy of the original image to blend overlays later
overlay = load_img.copy()

# Draw filled random color per instance
for i in range(len(results[0].boxes.cls.numpy())):
    segment = (results[0].masks.xy)[i]
    arr = np.array(segment, dtype=np.int32)
    arr = arr.reshape((-1, 1, 2))
    
    # Generate a random color (in BGR)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # Fill polygon on overlay image
    cv2.fillPoly(overlay, [arr], color)

# Blend the overlay with the original image
alpha = 0.5  
output = cv2.addWeighted(overlay, alpha, load_img, 1 - alpha, 0)

# Save the final image
cv2.imwrite("visualization/image_3.jpg", output)