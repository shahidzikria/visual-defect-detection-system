"""
Model module – defines the defect classifier using transfer learning.

Architecture: MobileNetV3-Small (via timm)
- Lightweight (~2.5 M params) for fast inference
- Pre-trained on ImageNet for strong feature extraction
- Custom classification head for binary defect detection
"""

import torch
import torch.nn as nn
import timm
from loguru import logger

from src.config import BACKBONE, PRETRAINED, NUM_CLASSES


class DefectClassifier(nn.Module):
    """
    Binary defect classifier built on a pre-trained MobileNetV3-Small backbone.

    Architecture choices rationale:
    - MobileNetV3-Small balances accuracy and inference speed – critical for
      real-time manufacturing line deployment.
    - Transfer learning from ImageNet provides robust low-level feature
      extraction even with limited domain-specific data.
    - A custom head with dropout prevents overfitting on small datasets.
    """

    def __init__(
        self,
        backbone_name: str = BACKBONE,
        pretrained: bool = PRETRAINED,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name

        # Load pre-trained backbone; reset classifier to Identity
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,          # removes the original head
        )

        # Determine feature dimension dynamically via dummy forward pass
        # (num_features can be inaccurate for some timm models)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            dummy_out = self.backbone(dummy)
            in_features = dummy_out.shape[-1]
        logger.info(
            f"Backbone '{backbone_name}' loaded  |  "
            f"feature dim = {in_features}  |  "
            f"pretrained = {pretrained}"
        )

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # (B, in_features)
        logits = self.classifier(features)   # (B, num_classes)
        return logits

    def count_parameters(self) -> dict:
        """Return a summary of trainable vs frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": total - trainable,
        }


def build_model(device: torch.device) -> DefectClassifier:
    """Instantiate the model and move to the target device."""
    model = DefectClassifier()
    model = model.to(device)
    param_info = model.count_parameters()
    logger.info(
        f"Model params → total: {param_info['total_params']:,}  "
        f"trainable: {param_info['trainable_params']:,}"
    )
    return model


def load_model_for_inference(model_path: str, device: torch.device) -> DefectClassifier:
    """Load a trained model from a checkpoint file for inference."""
    model = DefectClassifier(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle both full checkpoint dict and raw state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
            f"with val_acc={checkpoint.get('val_acc', '?')}"
        )
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model
