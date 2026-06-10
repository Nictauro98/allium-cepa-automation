"""Shared pytest fixtures."""
from __future__ import annotations

import pytest


@pytest.fixture()
def sample_eval_report() -> dict:
    return {
        "macro_f1": 0.85,
        "f1_per_class": {"mitosis": 0.80, "no_mitosis": 0.90},
        "accuracy": 0.87,
        "ece": 0.05,
        "n_samples": 200,
    }


@pytest.fixture()
def sample_prod_metrics() -> dict:
    return {
        "macro_f1": 0.80,
        "f1_per_class": {"mitosis": 0.76, "no_mitosis": 0.84},
        "accuracy": 0.82,
        "ece": 0.06,
        "n_samples": 200,
    }
