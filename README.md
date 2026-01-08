# YOLO-Style Object Detection from Scratch (PyTorch)

This repository implements a **complete object detection pipeline from scratch** using a **YOLO-style convolutional neural network** built in PyTorch.

The project covers:
- Manual dataset handling
- Annotation conversion (VOC → YOLO)
- Model design and training from scratch (no pretrained weights)
- Evaluation using mAP@0.5 and inference speed (FPS)
- Real-time detection using webcam input

---

## 🚀 Key Highlights

- ✅ Trained **entirely from scratch** (no pretrained backbone)
- ✅ Uses **PASCAL VOC 2007** dataset
- ✅ Manual dataset download (offline-safe)
- ✅ Custom YOLO-style CNN
- ✅ End-to-end reproducible pipeline
---

## 🧱 Environment Setup (RUN FIRST)

### 1️⃣ Create and activate a virtual environment (recommended)

```bash
conda create -n yolo_scratch python=3.10
conda activate yolo_scratch
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
📥 Dataset Download (Manual)
Dataset Used

PASCAL VOC 2007 (Train + Validation)

Source (Kaggle)

https://www.kaggle.com/datasets/stpeteishii/pascal-voc-2007-dataset

Steps

Download the dataset ZIP from Kaggle.

Extract it inside the project as:
### 🔄 Dataset Conversion (VOC → YOLO)

Convert XML annotations to YOLO format:
```bash
python voc_to_yolo.py
```
This creates a YOLO-style dataset:
```bash
dataset/
├── images/train/
├── images/val/
├── labels/train/
├── labels/val/
└── classes.txt
```
### 🧠 Model Architecture

YOLO-style convolutional neural network

Single-scale prediction (13×13 grid)

Input resolution: 416×416

No anchor boxes

No pretrained weights

Lightweight CNN backbone

Each grid cell predicts:

Bounding box (x, y, w, h)

Objectness score

Class probabilities

```bash
python train.py
```
Training Configuration

Optimizer: Adam

Learning rate: 1e-3

Batch size: 16

Epochs: 100

Pretrained weights: ❌ None
### Evaluation
```bash
python eval_fps.py
python eval_map.py
```
### 🎥 Real-Time Detection (Webcam)

Run real-time inference:

```bash
python detect.py
```
