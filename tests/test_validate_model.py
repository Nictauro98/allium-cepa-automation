"""Tests for the validate_model gate logic — fully isolated, no network."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from allium_cepa_classifier.config.validation_config import ValidationConfig
from allium_cepa_classifier.validation.validate_model import ValidationResult, _check, run_validation


class TestCheckFunction:
    """Unit tests for the _check decision logic."""

    def test_approves_when_all_criteria_pass(self, sample_eval_report, sample_prod_metrics):
        cfg = ValidationConfig()
        reasons = _check(sample_eval_report, sample_prod_metrics, cfg)
        assert reasons == []

    def test_rejects_when_macro_f1_delta_insufficient(self, sample_prod_metrics):
        new = {**sample_prod_metrics, "macro_f1": sample_prod_metrics["macro_f1"] + 0.005}
        cfg = ValidationConfig(min_f1_delta=0.01)
        reasons = _check(new, sample_prod_metrics, cfg)
        assert any("Macro F1" in r for r in reasons)

    def test_rejects_when_macro_f1_regresses(self, sample_prod_metrics):
        new = {**sample_prod_metrics, "macro_f1": sample_prod_metrics["macro_f1"] - 0.01}
        cfg = ValidationConfig(min_f1_delta=0.01)
        reasons = _check(new, sample_prod_metrics, cfg)
        assert any("Macro F1" in r for r in reasons)

    def test_rejects_when_per_class_f1_drops_too_much(self, sample_prod_metrics):
        new_per = dict(sample_prod_metrics["f1_per_class"])
        new_per["mitosis"] -= 0.05
        new = {**sample_prod_metrics, "macro_f1": sample_prod_metrics["macro_f1"] + 0.02, "f1_per_class": new_per}
        cfg = ValidationConfig(min_f1_delta=0.01, per_class_tolerance=0.03)
        reasons = _check(new, sample_prod_metrics, cfg)
        assert any("mitosis" in r for r in reasons)

    def test_rejects_when_ece_worsens_too_much(self, sample_prod_metrics):
        new = {
            **sample_prod_metrics,
            "macro_f1": sample_prod_metrics["macro_f1"] + 0.02,
            "ece": sample_prod_metrics["ece"] + 0.05,
        }
        cfg = ValidationConfig(min_f1_delta=0.01, ece_tolerance=0.02)
        reasons = _check(new, sample_prod_metrics, cfg)
        assert any("ECE" in r for r in reasons)

    def test_allows_ece_increase_within_tolerance(self, sample_prod_metrics):
        new = {
            **sample_prod_metrics,
            "macro_f1": sample_prod_metrics["macro_f1"] + 0.02,
            "ece": sample_prod_metrics["ece"] + 0.01,
        }
        cfg = ValidationConfig(min_f1_delta=0.01, ece_tolerance=0.02)
        reasons = _check(new, sample_prod_metrics, cfg)
        assert reasons == []

    def test_multiple_failures_reported(self, sample_prod_metrics):
        new = {
            **sample_prod_metrics,
            "macro_f1": sample_prod_metrics["macro_f1"] - 0.05,
            "ece": sample_prod_metrics["ece"] + 0.10,
        }
        cfg = ValidationConfig(min_f1_delta=0.01, ece_tolerance=0.02)
        reasons = _check(new, sample_prod_metrics, cfg)
        assert len(reasons) >= 2


class TestRunValidation:
    """Integration tests for run_validation — mocks the storage provider."""

    def test_auto_approves_when_no_baseline(self, tmp_path, sample_eval_report):
        report_path = tmp_path / "evaluation_report.json"
        report_path.write_text(json.dumps(sample_eval_report))

        mock_storage = MagicMock()
        mock_storage.exists.return_value = False

        with patch("allium_cepa_classifier.validation.validate_model.get_storage", return_value=mock_storage):
            result = run_validation(report_path)

        assert result.approved is True
        assert result.current_metrics is None
        assert any("first model" in r.lower() for r in result.reasons)

    def test_approves_when_metrics_improve(self, tmp_path, sample_eval_report, sample_prod_metrics):
        report_path = tmp_path / "evaluation_report.json"
        report_path.write_text(json.dumps(sample_eval_report))

        mock_storage = MagicMock()
        mock_storage.exists.return_value = True
        mock_storage.read_text.return_value = json.dumps(sample_prod_metrics)

        with patch("allium_cepa_classifier.validation.validate_model.get_storage", return_value=mock_storage):
            result = run_validation(report_path)

        assert result.approved is True
        assert result.reasons == []

    def test_rejects_when_metrics_worse(self, tmp_path, sample_prod_metrics):
        worse = {**sample_prod_metrics, "macro_f1": sample_prod_metrics["macro_f1"] - 0.05}
        report_path = tmp_path / "evaluation_report.json"
        report_path.write_text(json.dumps(worse))

        mock_storage = MagicMock()
        mock_storage.exists.return_value = True
        mock_storage.read_text.return_value = json.dumps(sample_prod_metrics)

        with patch("allium_cepa_classifier.validation.validate_model.get_storage", return_value=mock_storage):
            result = run_validation(report_path)

        assert result.approved is False
        assert len(result.reasons) > 0

    def test_result_is_dataclass(self, tmp_path, sample_eval_report):
        report_path = tmp_path / "evaluation_report.json"
        report_path.write_text(json.dumps(sample_eval_report))

        mock_storage = MagicMock()
        mock_storage.exists.return_value = False

        with patch("allium_cepa_classifier.validation.validate_model.get_storage", return_value=mock_storage):
            result = run_validation(report_path)

        assert isinstance(result, ValidationResult)
        assert isinstance(result.approved, bool)
        assert isinstance(result.reasons, list)
