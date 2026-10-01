"""
Pydantic schemas for the FastAPI request/response models.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response schema for the /predict endpoint."""

    predicted_class: str = Field(
        ...,
        description="Predicted label: 'normal' or 'defective'",
        examples=["defective"],
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (probability) of the predicted class",
        examples=[0.9532],
    )
    processing_time_ms: float = Field(
        ...,
        description="End-to-end processing time in milliseconds",
        examples=[23.45],
    )
    all_probabilities: Dict[str, float] = Field(
        ...,
        description="Probabilities for each class",
        examples=[{"defective": 0.9532, "normal": 0.0468}],
    )


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    status: str = Field(
        ...,
        description="API status: 'healthy' or 'model_not_loaded'",
        examples=["healthy"],
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded and ready for inference",
    )
    device: str = Field(
        ...,
        description="Compute device (e.g., 'cpu', 'cuda')",
        examples=["cpu"],
    )
    model_path: str = Field(
        ...,
        description="Path to the loaded model checkpoint",
    )
