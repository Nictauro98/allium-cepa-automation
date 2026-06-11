"""Weight-loader tests — no network, no MinIO, fake storage stub writes dummy files."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from allium_cepa_classifier.config import AlliumCepaConfig, ProductionConfig
from allium_cepa_classifier.serving.weights import ensure_production_weights


def _storage_stub() -> MagicMock:
    stub = MagicMock()
    stub.get_file.side_effect = lambda key, dest: Path(dest).write_bytes(b"dummy")
    return stub


class TestEnsureProductionWeights:
    def test_creates_cache_dir_if_missing(self, tmp_path):
        cache = tmp_path / "new_cache"
        cfg = ProductionConfig(weights_cache_dir=cache)
        with patch("allium_cepa_classifier.serving.weights.get_storage", return_value=_storage_stub()):
            ensure_production_weights(cfg)
        assert cache.is_dir()

    def test_pulls_all_three_weight_files(self, tmp_path):
        cfg = ProductionConfig(weights_cache_dir=tmp_path)
        stub = _storage_stub()
        with patch("allium_cepa_classifier.serving.weights.get_storage", return_value=stub):
            ensure_production_weights(cfg)
        assert stub.get_file.call_count == 3
        assert (tmp_path / "object_detection.pt").exists()
        assert (tmp_path / "classifier_calibrated.pt").exists()
        assert (tmp_path / "yolo_isotonic_calibrator.pkl").exists()

    def test_skips_files_that_already_exist(self, tmp_path):
        cfg = ProductionConfig(weights_cache_dir=tmp_path)
        (tmp_path / "object_detection.pt").write_bytes(b"cached")
        stub = _storage_stub()
        with patch("allium_cepa_classifier.serving.weights.get_storage", return_value=stub):
            ensure_production_weights(cfg)
        assert stub.get_file.call_count == 2

    def test_returns_allium_cepa_config_with_cache_paths(self, tmp_path):
        cfg = ProductionConfig(weights_cache_dir=tmp_path)
        with patch("allium_cepa_classifier.serving.weights.get_storage", return_value=_storage_stub()):
            result = ensure_production_weights(cfg)
        assert isinstance(result, AlliumCepaConfig)
        assert result.detection_weights_path == tmp_path / "object_detection.pt"
        assert result.classification_weights_path == tmp_path / "classifier_calibrated.pt"
        assert result.detection_calibrator_path == tmp_path / "yolo_isotonic_calibrator.pkl"

    def test_returns_config_with_cpu_enabled(self, tmp_path):
        cfg = ProductionConfig(weights_cache_dir=tmp_path)
        with patch("allium_cepa_classifier.serving.weights.get_storage", return_value=_storage_stub()):
            result = ensure_production_weights(cfg)
        assert result.use_cpu is True
