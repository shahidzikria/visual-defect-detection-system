"""
Tests for the FastAPI prediction API.

Covers:
- Health check endpoint
- Prediction endpoint with valid images
- Prediction endpoint with invalid input
"""

import io
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import torch
from PIL import Image
from fastapi.testclient import TestClient


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _create_test_image(size=(224, 224), color=(128, 128, 128)) -> bytes:
    """Create a dummy RGB image as bytes."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_model():
    """Create a mock model that returns fixed predictions."""
    model = MagicMock()
    # Return logits where defective (class 0) has higher score
    model.return_value = torch.tensor([[2.0, -1.0]])
    model.eval = MagicMock()
    return model


@pytest.fixture
def client(mock_model):
    """Create a test client with a mocked model."""
    # We need to patch the model before importing the app
    with patch("src.api.main.model", mock_model):
        with patch("src.api.main.device", torch.device("cpu")):
            from src.api.main import app
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_valid_image(self, client):
        image_bytes = _create_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert data["predicted_class"] in ["normal", "defective"]
        assert "confidence_score" in data
        assert 0.0 <= data["confidence_score"] <= 1.0
        assert "processing_time_ms" in data
        assert "all_probabilities" in data

    def test_predict_returns_probabilities(self, client):
        image_bytes = _create_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        data = response.json()
        probs = data["all_probabilities"]
        assert "defective" in probs
        assert "normal" in probs

    def test_predict_no_file_returns_422(self, client):
        response = client.post("/predict")
        assert response.status_code == 422


class TestModelInference:
    """Tests for model correctness."""

    def test_model_output_shape(self):
        """Ensure the model produces correct output shape."""
        from src.model import DefectClassifier
        model = DefectClassifier(pretrained=False)
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy)
        assert output.shape == (1, 2), f"Expected (1, 2), got {output.shape}"

    def test_model_probabilities_sum_to_one(self):
        """Softmax probabilities should sum to ~1."""
        from src.model import DefectClassifier
        model = DefectClassifier(pretrained=False)
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy)
            probs = torch.softmax(output, dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5
