"""
FastAPI production REST API for defect classification.

Endpoints:
    POST /predict  – upload an image and receive defect classification
    GET  /health   – system health / readiness check
"""

import io
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.schemas import PredictionResponse, HealthResponse
from src.config import MODEL_PATH, CLASS_NAMES, CONFIDENCE_THRESHOLD, IMG_SIZE
from src.dataset import get_eval_transforms
from src.model import load_model_for_inference

# ──────────────────────────────────────────────────────────────────────────────
# App initialisation
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Manufacturing Defect Detection API",
    description=(
        "Production REST API that classifies manufacturing product images "
        "as **normal** or **defective** using a MobileNetV3-based CNN."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model & transforms (loaded once at startup) ──────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
transform = get_eval_transforms()


@app.on_event("startup")
async def load_model():
    """Load the trained model into memory at API startup."""
    global model
    model_path = MODEL_PATH
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"No model checkpoint at {model_path}")
    model = load_model_for_inference(str(model_path), device)
    logger.info(f"✅ Model loaded from {model_path} on {device}")


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Return API health and model readiness status."""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        device=str(device),
        model_path=str(MODEL_PATH),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="Image file to classify")):
    """
    Accept an image upload and return defect classification.

    - **predicted_class**: `"normal"` or `"defective"`
    - **confidence_score**: probability of the predicted class
    - **processing_time_ms**: end-to-end inference time in milliseconds
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type '{file.content_type}'. Please upload an image.",
        )

    start_time = time.perf_counter()

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probs[predicted_idx])

    processing_time_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        f"Prediction: {predicted_class} ({confidence:.3f}) "
        f"| latency={processing_time_ms:.1f}ms | file={file.filename}"
    )

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence_score=round(confidence, 4),
        processing_time_ms=round(processing_time_ms, 2),
        all_probabilities={
            CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))
        },
    )
