# 🏭 Manufacturing Defect Detection System

> **Production-ready visual defect detection system** that automatically classifies product images as **Normal** or **Defective** using deep learning with transfer learning.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.30%25-brightgreen.svg)](#evaluation-results)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Dataset Strategy](#dataset-strategy)
- [Model Selection & Reasoning](#model-selection--reasoning)
- [Evaluation Results](#evaluation-results)
- [Error Analysis](#error-analysis)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Docker Deployment](#docker-deployment)
- [Demo Script](#demo-script)
- [Known Limitations & Trade-offs](#known-limitations--trade-offs)
- [Project Structure](#project-structure)

---

## Project Overview

This system provides an end-to-end pipeline for detecting manufacturing defects in casting product images:

1. **Data Pipeline** – Automated download, stratified splitting, and augmentation
2. **Model Training** – Transfer learning with MobileNetV3-Small backbone
3. **Evaluation** – Comprehensive metrics, confusion matrix, ROC curve, and error analysis
4. **Production API** – FastAPI REST service with health checks and structured logging
5. **Containerization** – Multi-stage Docker build for portable deployment

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MANUFACTURING DEFECT DETECTION SYSTEM                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────┐    │
│  │   Raw Data   │───▶│  Data Pipeline   │───▶│   Train / Val /   │    │
│  │  (Kaggle)    │    │  - Resize 224px  │    │    Test Splits    │    │
│  │  6,633 imgs  │    │  - Augmentation  │    │  70% / 15% / 15% │    │
│  └──────────────┘    │  - Normalize     │    └────────┬───────────┘    │
│                      └──────────────────┘             │                │
│                                                       ▼                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     TRAINING PIPELINE                            │  │
│  │  ┌─────────────────┐  ┌──────────────┐  ┌───────────────────┐   │  │
│  │  │  MobileNetV3    │  │   Weighted   │  │  Cosine Annealing │   │  │
│  │  │  (Pre-trained)  │──│   CrossEnt   │──│   LR + AdamW      │   │  │
│  │  │  + Custom Head  │  │   Loss       │  │  + Early Stopping │   │  │
│  │  └─────────────────┘  └──────────────┘  └───────────────────┘   │  │
│  └──────────────────────────────────────┬───────────────────────────┘  │
│                                         │                              │
│                                         ▼                              │
│  ┌─────────────────────┐    ┌─────────────────────────────────────┐   │
│  │  Saved Model (.pth) │    │          EVALUATION                 │   │
│  │  + ONNX Export      │    │  Accuracy: 99.30%  |  F1: 99.38%   │   │
│  └──────────┬──────────┘    │  ROC-AUC: 99.96%   |  0 FPs, 7 FNs│   │
│             │               └─────────────────────────────────────┘   │
│             ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    FASTAPI REST SERVICE                          │  │
│  │  POST /predict  ─  Upload image → {class, confidence, latency} │  │
│  │  GET  /health   ─  System readiness check                      │  │
│  │  Avg Latency: 24.6ms  |  CORS enabled  |  Structured logging  │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                     DOCKER CONTAINER                            │  │
│  │  Multi-stage build  |  Uvicorn  |  Health checks               │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Dataset Strategy

### Source
**Casting Defects in Submersible Pump Impellers** – a real-life industrial dataset from Kaggle containing grayscale images of cast metal impellers classified as normal (`ok_front`) or defective (`def_front`).

### Data Distribution

| Split   | Total | Defective | Normal | Ratio     |
|---------|-------|-----------|--------|-----------|
| Train   | 4,643 | 2,631     | 2,012  | 70%       |
| Val     | 995   | 563       | 432    | 15%       |
| Test    | 995   | 564       | 431    | 15%       |
| **Total** | **6,633** | **3,758** | **2,875** | **100%** |

### Preprocessing & Augmentation

- **Resize**: All images resized to 224×224 pixels
- **Normalization**: ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`
- **Training augmentations**:
  - Random horizontal flip (p=0.5) and vertical flip (p=0.3)
  - Random rotation (±15°)
  - Random affine translation (±5%) and scale (95–105%)
  - Color jitter (brightness ±20%, contrast ±20%, saturation ±10%)
  - Random grayscale (p=0.1)

### Class Imbalance Handling

The dataset has a mild imbalance (56.7% defective vs 43.3% normal). We use **inverse-frequency weighted cross-entropy loss**:

| Class     | Weight |
|-----------|--------|
| Defective | 0.882  |
| Normal    | 1.154  |

---

## Model Selection & Reasoning

### Architecture: MobileNetV3-Small (via `timm`)

| Criterion            | MobileNetV3-Small | ResNet-18        | EfficientNet-B0  |
|----------------------|-------------------|------------------|------------------|
| Parameters           | **1.67M**         | 11.7M            | 5.3M             |
| Inference (CPU)      | **~25ms**         | ~45ms            | ~35ms            |
| ImageNet Top-1       | 67.7%             | 69.8%            | 77.1%            |
| Model Size           | **~7MB (ONNX)**   | ~45MB            | ~20MB            |
| Edge Deployability   | ✅ Excellent       | ⚠️ Moderate      | ✅ Good           |

**Why MobileNetV3-Small?**

1. **Inference speed**: At ~25ms per image on CPU, it meets real-time manufacturing line requirements where decisions must be made in < 100ms.
2. **Compact size**: Only 1.67M parameters (7MB ONNX) enables deployment on edge devices directly on the production line.
3. **Transfer learning power**: Despite being lightweight, ImageNet pre-training provides robust low-level feature extraction (edges, textures, patterns) that transfer well to defect detection.
4. **Sufficient accuracy**: Achieves **99.30% accuracy** on this task — the simpler decision boundary of binary defect classification doesn't require a heavier model.

### Custom Classification Head

```
Backbone Output (1024-d) → Dropout(0.3) → Linear(1024, 256) → ReLU
                         → Dropout(0.15) → Linear(256, 2) → Softmax
```

### Training Configuration

| Hyperparameter       | Value              |
|----------------------|--------------------|
| Optimizer            | AdamW              |
| Learning Rate        | 1e-3               |
| Weight Decay         | 1e-4               |
| LR Scheduler         | Cosine Annealing   |
| Batch Size           | 32                 |
| Epochs               | 20                 |
| Early Stopping       | Patience = 5       |
| Best Checkpoint      | Epoch 12           |

---

## Evaluation Results

### Test Set Metrics

| Metric        | Score     |
|---------------|-----------|
| **Accuracy**  | **99.30%** |
| **Precision** | **100.00%** |
| **Recall**    | **98.76%** |
| **F1-Score**  | **99.38%** |
| **ROC-AUC**   | **99.96%** |

### Classification Report

```
              precision    recall  f1-score   support

   defective       1.00      0.99      0.99       564
      normal       0.98      1.00      0.99       431

    accuracy                           0.99       995
   macro avg       0.99      0.99      0.99       995
weighted avg       0.99      0.99      0.99       995
```

### Confusion Matrix

|                    | Predicted Defective | Predicted Normal |
|--------------------|:-------------------:|:----------------:|
| **Actual Defective** | 557 (TP)           | 7 (FN)           |
| **Actual Normal**    | 0 (FP)             | 431 (TN)         |

### Inference Latency

| Metric        | Value     |
|---------------|-----------|
| Average       | 24.6 ms   |
| P95           | 30.1 ms   |

---

## Error Analysis

### Summary
- **False Positives: 0** – No normal products were incorrectly flagged as defective (zero waste)
- **False Negatives: 7** – Seven defective products were missed (1.24% miss rate)

### False Negative Analysis

| Image File                | Confidence | Analysis                                           |
|---------------------------|:----------:|----------------------------------------------------|
| `cast_def_0_1053.jpeg`    | 99.4%      | High-confidence misclassification – subtle defect  |
| `cast_def_0_2535.jpeg`    | 97.7%      | Defect may be near image border / partially visible |
| `cast_def_0_2949.jpeg`    | 98.6%      | Subtle surface defect hard to distinguish           |
| `cast_def_0_3464.jpeg`    | 98.6%      | Defect pattern similar to normal surface texture    |
| `cast_def_0_4073.jpeg`    | 97.1%      | Low-contrast defect region                          |
| `cast_def_0_6254.jpeg`    | 98.8%      | Defect at edge of casting                           |
| `cast_def_0_7578.jpeg`    | 77.9%      | Lowest confidence – borderline case                 |

### Key Observations

1. **No FPs is ideal for manufacturing** – prevents unnecessary rejection of good products, saving cost.
2. **7 FNs (1.24% miss rate)** – in safety-critical applications, this could be improved by:
   - Lowering the confidence threshold (trading off some FPs for fewer FNs)
   - Ensembling multiple models
   - Adding a secondary inspection stage for low-confidence predictions
3. **Most FNs are high-confidence** – suggesting these are genuinely ambiguous samples where defects are subtle or border-line. Image-level annotation quality could be a contributing factor.

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- pip (or conda)
- Docker (optional, for containerized deployment)

### Local Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd assessment2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (download → split → train → evaluate → export)
python -m src.main

# 4. Run evaluation only (requires pre-trained model in models/)
python -m src.main --skip-train
```

### Kaggle API Setup (for dataset download)

The pipeline uses `kagglehub` for automatic dataset download. If prompted:

```bash
# Option A: Set environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Option B: Place kaggle.json in ~/.kaggle/kaggle.json
```

---

## Usage

### Run Inference on a Single Image

```bash
python predict.py --image path/to/image.jpg
```

### Run Inference on a Directory

```bash
python predict.py --image-dir path/to/images/
```

### Start the API Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/image.jpg"
```

---

## API Reference

### `POST /predict`

Upload an image for defect classification.

**Request**: `multipart/form-data` with an `file` field containing the image.

**Response** (200 OK):
```json
{
  "predicted_class": "defective",
  "confidence_score": 0.9953,
  "processing_time_ms": 24.6,
  "all_probabilities": {
    "defective": 0.9953,
    "normal": 0.0047
  }
}
```

### `GET /health`

Check system health and model readiness.

**Response** (200 OK):
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "model_path": "models/best_model.pth"
}
```

### Interactive API Docs

Once the server is running, visit:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Docker Deployment

### Build & Run (Docker Compose)

```bash
# Build and start
docker-compose up --build -d

# Check health
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict -F "file=@test_image.jpg"

# Stop
docker-compose down
```

### Build & Run (Docker CLI)

```bash
# Build
docker build -t defect-detection .

# Run
docker run -p 8000:8000 -v ./models:/app/models:ro defect-detection
```

---

## Demo Script

### 2-3 Minute Technical Demo Outline

1. **Introduction** (20s)
   - Problem: Manual inspection is slow and error-prone
   - Solution: Automated CNN-based defect detection

2. **Architecture Overview** (30s)
   - Walk through the system architecture diagram
   - Explain MobileNetV3 choice and transfer learning approach

3. **Live API Demo** (60s)
   - Start the API server: `uvicorn src.api.main:app`
   - Open Swagger UI at `/docs`
   - Upload a defective image → show prediction response
   - Upload a normal image → show correct classification
   - Show health endpoint

4. **Results Summary** (30s)
   - Show confusion matrix and key metrics
   - Highlight: 99.30% accuracy, 0 false positives
   - Discuss inference latency (24.6ms avg)

5. **Production Readiness** (20s)
   - Docker deployment: `docker-compose up`
   - ONNX export for edge deployment
   - Structured logging and health checks

---

## Known Limitations & Trade-offs

### Limitations

1. **Binary classification only** – Classifies as defective/normal but does not localize or categorize defect types (e.g., cracks, holes, surface roughness).
2. **Single product type** – Trained specifically on submersible pump impeller castings. Would need retraining/fine-tuning for other product types.
3. **CPU inference** – Tested on CPU (24.6ms avg). GPU deployment would further reduce latency for batch processing.
4. **No defect localization** – Does not provide bounding boxes or heatmaps showing where defects are located (would require Grad-CAM or object detection approaches).
5. **7 missed defects** – 1.24% false negative rate may be unacceptable in safety-critical applications.

### Production Trade-offs

| Decision                   | Trade-off                                                |
|----------------------------|----------------------------------------------------------|
| MobileNetV3 over ResNet    | Smaller model, faster inference, slightly lower capacity |
| Binary over multi-class    | Simpler but loses defect categorization information      |
| 0.5 confidence threshold   | Balanced precision/recall; can be adjusted per use case  |
| Single model vs ensemble   | Faster inference but less robust on edge cases           |
| CPU deployment             | More portable but slower than GPU for batch processing   |

### Future Improvements

- **Grad-CAM visualization** for defect localization
- **Multi-class defect classification** (crack types, surface defects, etc.)
- **Model ensembling** to reduce false negatives
- **Active learning** pipeline for continuous improvement with new production data
- **A/B testing framework** for comparing model versions in production

---

## Project Structure

```
assessment2/
├── src/
│   ├── __init__.py          # Package init
│   ├── config.py            # Central configuration & hyperparameters
│   ├── dataset.py           # Data download, splitting, augmentation, loaders
│   ├── model.py             # MobileNetV3 classifier architecture
│   ├── train.py             # Training loop with early stopping & checkpointing
│   ├── evaluate.py          # Metrics, confusion matrix, ROC, error analysis
│   ├── main.py              # Pipeline orchestration entry point
│   └── api/
│       ├── __init__.py
│       ├── main.py          # FastAPI application with /predict and /health
│       └── schemas.py       # Pydantic request/response models
├── tests/
│   └── test_api.py          # Unit & integration tests
├── models/
│   ├── best_model.pth       # Trained model checkpoint (~21MB)
│   └── model.onnx           # ONNX exported model (~7MB)
├── outputs/
│   ├── evaluation_report.json   # Full metrics & error analysis
│   ├── confusion_matrix.png     # Confusion matrix visualization
│   ├── roc_curve.png            # ROC curve plot
│   ├── training_curves.png      # Loss & accuracy curves
│   └── training_history.json    # Epoch-by-epoch metrics
├── data/
│   └── processed/           # Train/val/test splits (auto-generated)
├── predict.py               # Standalone prediction script
├── requirements.txt         # Python dependencies
├── Dockerfile               # Multi-stage container build
├── docker-compose.yml       # One-command deployment
├── .gitignore               # Git exclusions
└── README.md                # This file
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src
```

---

## Reproducing Training

```bash
# Full pipeline from scratch
python -m src.main

# With forced data re-split
python -m src.main --force-split

# Evaluate existing model only
python -m src.main --skip-train
```

Training takes approximately **90 minutes on CPU** (20 epochs, batch size 32) or **~10 minutes on GPU**.

---

## License

This project is developed as part of an academic assessment.
