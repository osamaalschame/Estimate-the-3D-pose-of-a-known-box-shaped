# 3D Pose Estimation & Object Segmentation Tasks

## Introduction

This repository contains two tasks related to computer vision and deep learning:

- **Task 1**: Estimate the **3D pose** of a known box-shaped object using a provided **depth map**.
- **Task 2**: **Train and evaluate a model** for **segmenting box-shaped objects** in RGB scenes.

Both tasks aim to advance capabilities in spatial understanding and object localization using real-world imaging data.

---

## 🛠️ Environment Setup

To get started, create a virtual environment and install the required dependencies.

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Linux/macOS
venv\Scripts\activate      # On Windows
```

### 2. Install required libraries
```bash
pip install -r requirements.txt
```
---

## 📁 Repository Structure

```
.
├── Task1/
│   ├── data/                  # Provided depth maps and related data
│   └── translate.py           # Code to estimate 3D pose from depth map
|   └── README.md              # Approach implementation
│
├── Task2/
│   ├── train.py               # Training script for segmentation model
│   ├── test.py                # test the model
│   ├── visualization/         # Folder for saving visual outputs
│   ├── training/              # Saved model weights, checkpoints
|   ├── dataset/               # put the dataset in this folder OSCD dataset  
|   ├── data.yaml/             # contains the directory for train and val images,num_classes
│   └── performance_report.pdf # Final performance report of the model
│   └── README.md.             # Report training setup: model architecture, loss function,lr
│
├── README.md                  # This readme file
└── requirements.txt           # List of required Python libraries
```

---

## 📌 Task Details

### Task 1: 3D Pose Estimation
- **Input**: Extrinsics, intrinsics, one-box-color and depth map of a scene, .
- **Output**: 3D position and orientation (pose) of a known box-shaped object.
- **Script**: `translate.py` reads depth maps, processes them, and estimates the pose.

### Task 2: Box-shaped Object Segmentation
- **Input**: RGB scenes with box-shaped objects.
- **Model**: Deep learning model trained to predict segmentation masks.
- **Scripts**:
  - `train.py` for training
  - `test.py` for evaluation
- **Outputs**: Visualized segmentation masks, trained model weights, and a performance report.

---

## 🚀 Quick Start

### For Task 1:
```bash
cd Task1
python translate.py
```

### For Task 2:
```bash
cd Task2
python train.py --data data.yaml --model yolo11s-seg.pt --epochs 200 --imgsz 512 --batch 16 --name Carton-seg-s --patience 15
   # To train the model
python test.py --model training/Carton-seg-s/weights/best.pt --image dataset/test/net\ \(9125\).jpg
    # To evaluate the model
```

---

