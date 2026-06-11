"""FastAPI endpoint tests — no real model, no network, no GPU."""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api import app

_COUNTS_RAW = {
    "total_cells": 10,
    "mitotic_cells": 2,
    "non_mitotic_cells": 8,
    "mitotic_index": 0.2,
}
_COUNTS_CI = {
    "mi": 0.21,
    "var_mi": 0.001,
    "sigma_mi": 0.031,
    "ci_lower": 0.148,
    "ci_upper": 0.272,
    "n_cel": 10.0,
    "n_mit": 2.1,
    "var_cel": 0.5,
    "var_mit": 0.3,
}
_DETECTIONS = pd.DataFrame(
    {
        "x_min": [10, 50],
        "y_min": [10, 50],
        "x_max": [40, 80],
        "y_max": [40, 80],
        "confidence": [0.9, 0.8],
        "class_name": ["cell", "cell"],
        "mitosis": [True, False],
        "p_hat": [0.9, 0.8],
        "q_interphase": [0.2, 0.8],
        "q_mitosis": [0.8, 0.2],
        "class_id": [0, 0],
        "image": ["test.png", "test.png"],
        "mitosis_score": [0.8, 0.2],
    }
)


def _make_model_stub() -> MagicMock:
    result = MagicMock()
    result.get_counts.return_value = _COUNTS_RAW.copy()
    result.get_counts_with_ci.return_value = _COUNTS_CI.copy()
    result.detections = _DETECTIONS
    model = MagicMock()
    model.predict.return_value = result
    return model


@pytest.fixture()
def client():
    model_stub = _make_model_stub()
    with (
        patch("app.api.ensure_production_weights", return_value=MagicMock()),
        patch("app.api.AlliumCepaModel", return_value=model_stub),
    ):
        with TestClient(app) as c:
            yield c


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color=(128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


class TestHealth:
    def test_returns_ok_when_model_loaded(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "model_loaded": True}


class TestPredict:
    def test_returns_counts_and_detections_keys(self, client):
        resp = client.post("/predict", files={"file": ("t.png", _png_bytes(), "image/png")})
        assert resp.status_code == 200
        payload = resp.json()
        assert "counts" in payload
        assert "detections" in payload

    def test_counts_contains_merged_raw_and_ci_keys(self, client):
        resp = client.post("/predict", files={"file": ("t.png", _png_bytes(), "image/png")})
        counts = resp.json()["counts"]
        for key in ("total_cells", "mitotic_cells", "non_mitotic_cells", "mitotic_index"):
            assert key in counts, f"missing raw key: {key}"
        for key in ("mi", "ci_lower", "ci_upper", "sigma_mi"):
            assert key in counts, f"missing CI key: {key}"

    def test_detections_is_list_of_records(self, client):
        resp = client.post("/predict", files={"file": ("t.png", _png_bytes(), "image/png")})
        detections = resp.json()["detections"]
        assert isinstance(detections, list)
        assert len(detections) == 2
        assert "x_min" in detections[0]
        assert "mitosis" in detections[0]
