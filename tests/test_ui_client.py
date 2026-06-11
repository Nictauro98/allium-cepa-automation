"""UI HTTP client tests — no network, no Streamlit runtime."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests as requests_lib

from ui.client import run_prediction

_CANNED_COUNTS = {
    "total_cells": 5,
    "mitotic_cells": 1,
    "non_mitotic_cells": 4,
    "mitotic_index": 0.2,
    "mi": 0.21,
    "ci_lower": 0.10,
    "ci_upper": 0.32,
    "sigma_mi": 0.06,
    "n_cel": 5.0,
    "n_mit": 1.05,
    "var_mi": 0.003,
    "var_cel": 0.4,
    "var_mit": 0.2,
}
_CANNED_DETECTIONS = [
    {
        "x_min": 10, "y_min": 10, "x_max": 40, "y_max": 40,
        "confidence": 0.9, "class_name": "cell", "mitosis": True, "p_hat": 0.9,
    },
    {
        "x_min": 50, "y_min": 50, "x_max": 80, "y_max": 80,
        "confidence": 0.8, "class_name": "cell", "mitosis": False, "p_hat": 0.8,
    },
]


def _fake_response() -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {"counts": _CANNED_COUNTS, "detections": _CANNED_DETECTIONS}
    return mock


def _uploaded(name: str = "test.png", data: bytes = b"imgdata") -> MagicMock:
    stub = MagicMock()
    stub.name = name
    stub.getvalue.return_value = data
    return stub


class TestRunPrediction:
    def test_returns_dict_and_dataframe(self):
        with patch("ui.client.requests.post", return_value=_fake_response()):
            counts, detections = run_prediction(_uploaded())
        assert isinstance(counts, dict)
        assert isinstance(detections, pd.DataFrame)

    def test_counts_contains_all_expected_keys(self):
        with patch("ui.client.requests.post", return_value=_fake_response()):
            counts, _ = run_prediction(_uploaded())
        for key in ("mi", "ci_lower", "ci_upper", "total_cells", "mitotic_cells", "non_mitotic_cells"):
            assert key in counts, f"missing key: {key}"

    def test_detections_has_expected_columns(self):
        with patch("ui.client.requests.post", return_value=_fake_response()):
            _, detections = run_prediction(_uploaded())
        for col in ("x_min", "y_min", "x_max", "y_max", "confidence", "mitosis"):
            assert col in detections.columns, f"missing column: {col}"

    def test_detections_row_count_matches_payload(self):
        with patch("ui.client.requests.post", return_value=_fake_response()):
            _, detections = run_prediction(_uploaded())
        assert len(detections) == len(_CANNED_DETECTIONS)

    def test_posts_to_predict_endpoint(self):
        mock_post = MagicMock(return_value=_fake_response())
        with patch("ui.client.requests.post", mock_post):
            run_prediction(_uploaded(name="sample.tif"))
        url = mock_post.call_args[0][0]
        assert url.endswith("/predict")

    def test_raises_on_http_error(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests_lib.HTTPError("500 Server Error")
        with patch("ui.client.requests.post", return_value=mock_resp):
            with pytest.raises(requests_lib.HTTPError):
                run_prediction(_uploaded())
