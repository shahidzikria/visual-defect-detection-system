"""
Dataset module – handles downloading, splitting, augmentation, and loading.

Uses the Kaggle Casting Defects dataset (casting_data) which contains
grayscale images of submersible pump impeller castings classified as
'ok_front' (normal) and 'def_front' (defective).
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from loguru import logger
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    CLASS_NAMES,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Download the raw dataset via kagglehub
# ──────────────────────────────────────────────────────────────────────────────

def download_dataset() -> Path:
    """
    Download the casting defect dataset using kagglehub.
    Returns the path to the downloaded dataset root.
    """
    try:
        import kagglehub
        logger.info("Downloading casting defect dataset via kagglehub …")
        path = kagglehub.dataset_download("ravirajsinh45/real-life-industrial-dataset-of-casting-product")
        logger.info(f"Dataset downloaded to: {path}")
        return Path(path)
    except Exception as e:
        logger.warning(f"kagglehub download failed: {e}. Attempting manual data setup …")
        # Fallback: user should place data manually
        if RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.iterdir()):
            logger.info(f"Using existing raw data at {RAW_DATA_DIR}")
            return RAW_DATA_DIR
        raise FileNotFoundError(
            f"Could not download dataset and no raw data found at {RAW_DATA_DIR}. "
            "Please download the dataset manually from Kaggle and place it in data/raw/."
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Locate image folders (handles nested structure)
# ──────────────────────────────────────────────────────────────────────────────

def _find_class_dirs(root: Path) -> Tuple[Path, Path]:
    """
    Walk *root* and return (defective_dir, normal_dir).
    The Kaggle dataset nests images under various directory layouts;
    this function handles the common ones.
    """
    def_dir = None
    ok_dir = None

    for dirpath, dirnames, filenames in os.walk(root):
        dirname_lower = Path(dirpath).name.lower()
        if "def" in dirname_lower and filenames:
            def_dir = Path(dirpath)
        elif ("ok" in dirname_lower or "normal" in dirname_lower) and filenames:
            ok_dir = Path(dirpath)

    if def_dir is None or ok_dir is None:
        # Try looking for direct subdirs named casting_data/casting_data/...
        for child in root.rglob("*"):
            if child.is_dir():
                cname = child.name.lower()
                if "def" in cname:
                    def_dir = child
                elif "ok" in cname or "normal" in cname:
                    ok_dir = child
            if def_dir and ok_dir:
                break

    if def_dir is None or ok_dir is None:
        raise FileNotFoundError(
            f"Could not locate defective / normal subdirectories under {root}. "
            "Expected folder names containing 'def' and 'ok' / 'normal'."
        )

    logger.info(f"Found defective images: {def_dir}  ({len(list(def_dir.glob('*')))} files)")
    logger.info(f"Found normal images   : {ok_dir}  ({len(list(ok_dir.glob('*')))} files)")
    return def_dir, ok_dir


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Create stratified train / val / test split
# ──────────────────────────────────────────────────────────────────────────────

def prepare_splits(raw_root: Path, force: bool = False) -> None:
    """
    Create train / val / test directories with class sub-folders
    by performing a stratified split on the raw images.
    """
    if TRAIN_DIR.exists() and not force:
        logger.info("Splits already exist; skipping. Use force=True to rebuild.")
        return

    def_dir, ok_dir = _find_class_dirs(raw_root)

    # Collect all file paths with labels
    all_images = []
    all_labels = []
    for img_path in sorted(def_dir.glob("*")):
        if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            all_images.append(img_path)
            all_labels.append(0)  # defective
    for img_path in sorted(ok_dir.glob("*")):
        if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            all_images.append(img_path)
            all_labels.append(1)  # normal

    logger.info(f"Total images found: {len(all_images)}  "
                f"(defective={all_labels.count(0)}, normal={all_labels.count(1)})")

    # First split: train vs (val+test)
    train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(
        all_images, all_labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=all_labels,
        random_state=RANDOM_SEED,
    )

    # Second split: val vs test
    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(
        temp_imgs, temp_lbls,
        test_size=relative_test,
        stratify=temp_lbls,
        random_state=RANDOM_SEED,
    )

    logger.info(f"Split sizes → train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

    # Copy files into the folder structure expected by ImageFolder
    for split_name, imgs, lbls in [
        ("train", train_imgs, train_lbls),
        ("val", val_imgs, val_lbls),
        ("test", test_imgs, test_lbls),
    ]:
        for img_path, label in zip(imgs, lbls):
            cls_name = CLASS_NAMES[label]
            dest_dir = PROCESSED_DATA_DIR / split_name / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest_dir / img_path.name)

    logger.info("✅ Data splits created successfully.")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Augmentation transforms
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """Training augmentations: realistic manufacturing variations."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Deterministic transforms for validation / test / inference."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Data loaders
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced data."""
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    total = len(targets)
    weights = total / (len(class_counts) * class_counts.astype(float))
    logger.info(f"Class weights: {dict(zip(CLASS_NAMES, weights.round(3)))}")
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Return (train_loader, val_loader, test_loader, class_weights).
    """
    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=get_train_transforms())
    val_ds = datasets.ImageFolder(str(VAL_DIR), transform=get_eval_transforms())
    test_ds = datasets.ImageFolder(str(TEST_DIR), transform=get_eval_transforms())

    logger.info(f"Class-to-index mapping: {train_ds.class_to_idx}")

    class_weights = compute_class_weights(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights
