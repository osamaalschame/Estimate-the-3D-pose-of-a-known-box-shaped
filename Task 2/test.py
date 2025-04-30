import argparse
import os
import cv2
import numpy as np
import random
from ultralytics import YOLO

def visualize_prediction(model_path, image_path):
    # Load model
    model = YOLO(model_path)

    # Load image
    load_img = cv2.imread(image_path)
    if load_img is None:
        raise ValueError(f"Image not found or invalid: {image_path}")

    # Ensure output directory exists
    os.makedirs('visualization', exist_ok=True)

    # Run prediction
    results = model.predict(source=load_img, conf=0.6)
    overlay = load_img.copy()

    # Draw masks
    for i in range(len(results[0].boxes.cls.numpy())):
        segment = (results[0].masks.xy)[i]
        arr = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.fillPoly(overlay, [arr], color)

    # Blend overlays
    alpha = 0.5
    output = cv2.addWeighted(overlay, alpha, load_img, 1 - alpha, 0)

    # Generate output path
    image_name = os.path.basename(image_path)
    save_path = os.path.join("visualization", image_name)

    # Save image
    cv2.imwrite(save_path, output)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Segmentation Visualizer")
    parser.add_argument("--model", required=True, help="Path to YOLO model")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    visualize_prediction(args.model, args.image)
