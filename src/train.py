"""
Training pipeline for the defect detection model.

Features:
- Weighted cross-entropy loss for class imbalance
- Cosine annealing learning rate schedule
- Early stopping with patience
- Best model checkpointing
- Training history logging
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from src.config import (
    MODEL_DIR,
    OUTPUT_DIR,
    BEST_MODEL_FILENAME,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LR_SCHEDULER,
    STEP_SIZE,
    GAMMA,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    RANDOM_SEED,
)
from src.model import DefectClassifier


# ──────────────────────────────────────────────────────────────────────────────
# Early stopping helper
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE,
                 min_delta: float = EARLY_STOPPING_MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"  ⏳ EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


# ──────────────────────────────────────────────────────────────────────────────
# Single epoch runners
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: DefectClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: DefectClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the model. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ──────────────────────────────────────────────────────────────────────────────
# Full training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    model: DefectClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
) -> Dict[str, List[float]]:
    """
    Full training loop with:
    - Class-weighted cross-entropy
    - Learning rate scheduling
    - Early stopping
    - Best checkpoint saving
    """
    torch.manual_seed(RANDOM_SEED)

    # Weighted loss for class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # LR scheduler
    if LR_SCHEDULER == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=STEP_SIZE, gamma=GAMMA
        )

    early_stopping = EarlyStopping()

    # History tracking
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODEL_DIR / BEST_MODEL_FILENAME

    logger.info(f"Starting training for {NUM_EPOCHS} epochs on {device}")
    total_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log metrics
        logger.info(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} │ "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} │ "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} │ "
            f"lr={current_lr:.2e} │ {epoch_time:.1f}s"
        )

        # Track history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                },
                best_model_path,
            )
            logger.info(f"  💾 Saved best model (val_acc={val_acc:.4f})")

        # Early stopping check
        if early_stopping(val_loss):
            logger.info(f"  🛑 Early stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - total_start
    logger.info(f"Training complete in {total_time:.1f}s  |  Best val_acc = {best_val_acc:.4f}")

    # Save training history
    history_path = OUTPUT_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    return history
