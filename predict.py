"""
Standalone prediction script – demonstrates how to use the trained model
for inference on a single image or directory of images.

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --image-dir path/to/images/
"""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from loguru import logger

from src.config import MODEL_DIR, BEST_MODEL_FILENAME, CLASS_NAMES
from src.dataset import get_eval_transforms
from src.model import load_model_for_inference


def predict_single(image_path: str, model, transform, device) -> dict:
    """Run inference on a single image."""
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    latency_ms = (time.perf_counter() - start) * 1000

    pred_idx = int(probs.argmax())
    return {
        "file": str(image_path),
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))},
        "latency_ms": round(latency_ms, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Defect Detection Inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image")
    group.add_argument("--image-dir", type=str, help="Path to directory of images")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = MODEL_DIR / BEST_MODEL_FILENAME
    model = load_model_for_inference(str(model_path), device)
    transform = get_eval_transforms()

    if args.image:
        result = predict_single(args.image, model, transform, device)
        logger.info(f"Result: {result}")
    else:
        img_dir = Path(args.image_dir)
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            for img_path in sorted(img_dir.glob(ext)):
                result = predict_single(img_path, model, transform, device)
                logger.info(
                    f"{result['file']} → {result['predicted_class']} "
                    f"({result['confidence']:.3f}) [{result['latency_ms']:.1f}ms]"
                )


if __name__ == "__main__":
    main()
