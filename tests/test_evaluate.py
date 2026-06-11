"""Tests for evaluate.py metric computation — no model weights needed."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate import _ece


class TestEce:
    def test_perfect_calibration_gives_zero(self):
        # ECE is 0 when within every bin, mean confidence == fraction correct.
        # Simplest case: all 100 samples fall in the same bin (conf=0.8),
        # and exactly 80% are predicted correctly → accuracy = confidence = 0.8.
        n = 100
        conf = 0.8
        probs = np.array([[conf, 1 - conf]] * n)
        labels = np.array([0] * 80 + [1] * 20)  # argmax=0 for all; 80 correct, 20 wrong
        ece = _ece(probs, labels)
        assert ece == pytest.approx(0.0, abs=1e-6)

    def test_worst_calibration_gives_high_ece(self):
        # Confident but always wrong → ECE should be close to 1.
        probs = np.array([[0.99, 0.01]] * 100)
        labels = np.ones(100, dtype=int)  # always wrong
        ece = _ece(probs, labels)
        assert ece > 0.8

    def test_ece_in_valid_range(self):
        rng = np.random.default_rng(42)
        probs_raw = rng.dirichlet([1, 1], size=200)
        labels = rng.integers(0, 2, size=200)
        ece = _ece(probs_raw, labels)
        assert 0.0 <= ece <= 1.0

    def test_ece_returns_float(self):
        probs = np.array([[0.6, 0.4], [0.3, 0.7]])
        labels = np.array([0, 1])
        result = _ece(probs, labels)
        assert isinstance(result, float)

    def test_single_sample(self):
        probs = np.array([[0.8, 0.2]])
        labels = np.array([0])
        ece = _ece(probs, labels)
        assert 0.0 <= ece <= 1.0
