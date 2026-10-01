"""
Main entry point – orchestrates the full pipeline:
  1. Download dataset
  2. Prepare train/val/test splits
  3. Train the model
  4. Evaluate on test set
  5. Export model to ONNX

Usage:
    python -m src.main                # run full pipeline
    python -m src.main --skip-train   # evaluate only (requires existing model)
"""

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger

# Configure loguru
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)

from src.config import MODEL_DIR, BEST_MODEL_FILENAME, OUTPUT_DIR, RANDOM_SEED
from src.dataset import download_dataset, prepare_splits, get_dataloaders
from src.model import build_model, load_model_for_inference
from src.train import train
from src.evaluate import evaluate


def export_onnx(model, device, save_path: Path) -> None:
    """Export trained model to ONNX format for portable deployment."""
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        str(save_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    logger.info(f"✅ ONNX model exported → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Manufacturing Defect Detection Pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, evaluate only")
    parser.add_argument("--force-split", action="store_true", help="Force re-split of data")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Step 1: Download dataset ──
    logger.info("=" * 60)
    logger.info("STEP 1: Dataset acquisition")
    logger.info("=" * 60)
    raw_root = download_dataset()

    # ── Step 2: Prepare splits ──
    logger.info("=" * 60)
    logger.info("STEP 2: Preparing train/val/test splits")
    logger.info("=" * 60)
    prepare_splits(raw_root, force=args.force_split)

    # ── Step 3: Load data ──
    logger.info("=" * 60)
    logger.info("STEP 3: Creating data loaders")
    logger.info("=" * 60)
    train_loader, val_loader, test_loader, class_weights = get_dataloaders()

    if not args.skip_train:
        # ── Step 4: Train ──
        logger.info("=" * 60)
        logger.info("STEP 4: Training model")
        logger.info("=" * 60)
        model = build_model(device)
        history = train(model, train_loader, val_loader, class_weights, device)

    # ── Step 5: Evaluate ──
    logger.info("=" * 60)
    logger.info("STEP 5: Evaluating on test set")
    logger.info("=" * 60)
    model_path = MODEL_DIR / BEST_MODEL_FILENAME
    model = load_model_for_inference(str(model_path), device)
    report = evaluate(model, test_loader, device)

    # ── Step 6: Export ONNX ──
    logger.info("=" * 60)
    logger.info("STEP 6: Exporting to ONNX")
    logger.info("=" * 60)
    onnx_path = MODEL_DIR / "model.onnx"
    export_onnx(model, device, onnx_path)

    logger.info("=" * 60)
    logger.info("🎉  PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
