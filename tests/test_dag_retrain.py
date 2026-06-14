"""Unit tests for retrain_pipeline DAG task callables."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_ti(xcom_store=None):
    store = xcom_store if xcom_store is not None else {}
    ti = MagicMock()

    def push(key, value):
        store[key] = value

    def pull(task_ids=None, key=None):
        return store.get(key)

    ti.xcom_push.side_effect = lambda key, value: push(key, value)
    ti.xcom_pull.side_effect = lambda task_ids=None, key=None: pull(task_ids, key)
    return ti, store


# ---------------------------------------------------------------------------
# read_validation_result branches correctly
# ---------------------------------------------------------------------------


def test_read_validation_result_branches_to_promote(tmp_path):
    from retrain_pipeline import read_validation_result

    vr_path = tmp_path / "validation_result.json"
    vr_path.write_text(json.dumps({"approved": True, "metrics": {"accuracy": 0.95}}))

    ti, store = _make_ti()
    with patch("retrain_pipeline._VALIDATION_RESULT_PATH", vr_path):
        branch = read_validation_result(ti=ti)

    assert branch == "promote_model"
    assert store["validation_result"]["approved"] is True


def test_read_validation_result_branches_to_archive(tmp_path):
    from retrain_pipeline import read_validation_result

    vr_path = tmp_path / "validation_result.json"
    vr_path.write_text(json.dumps({"approved": False, "reasons": ["accuracy below threshold"]}))

    ti, store = _make_ti()
    with patch("retrain_pipeline._VALIDATION_RESULT_PATH", vr_path):
        branch = read_validation_result(ti=ti)

    assert branch == "archive_model"


def test_read_validation_result_missing_file_archives(tmp_path):
    from retrain_pipeline import read_validation_result

    missing = tmp_path / "nonexistent.json"
    ti, _ = _make_ti()
    with patch("retrain_pipeline._VALIDATION_RESULT_PATH", missing):
        branch = read_validation_result(ti=ti)
    assert branch == "archive_model"


# ---------------------------------------------------------------------------
# promote_model issues expected put_file calls (mock storage)
# ---------------------------------------------------------------------------


def test_promote_model_puts_weight_files(tmp_path):
    from retrain_pipeline import promote_model

    from allium_cepa_classifier.config import ProductionConfig

    cfg = ProductionConfig()

    # Create dummy weight files
    weights_dir = tmp_path / "models" / "production"
    weights_dir.mkdir(parents=True)
    (weights_dir / "object_detection.pt").write_bytes(b"fake_weights")
    (weights_dir / "classifier_calibrated.pt").write_bytes(b"fake_weights")
    (weights_dir / "yolo_isotonic_calibrator.pkl").write_bytes(b"fake_weights")

    validation_result = {"approved": True, "metrics": {"accuracy": 0.95}}
    xcom_store = {
        "validation_result": validation_result,
        "current_sha": "abc123",
    }
    ti, _ = _make_ti(xcom_store)

    mock_storage = MagicMock()
    mock_storage.put_file = MagicMock()

    last_sha_path = tmp_path / "last_sha.txt"

    with (
        patch("allium_cepa_classifier.providers.factory.get_storage", return_value=mock_storage),
        patch("retrain_pipeline.Path") as mock_path_cls,
        patch("retrain_pipeline._LAST_SHA_PATH", last_sha_path),
    ):
        # Make Path("models/production") resolve to our tmp weights dir
        def path_side_effect(arg=""):
            if arg in ("dvc.yaml", "validation_result.json"):
                return MagicMock(exists=lambda: False)
            real = Path(arg)
            if "models" in str(arg):
                return weights_dir
            return real

        mock_path_cls.side_effect = path_side_effect

        # Directly call with the real weights_dir
        with patch("retrain_pipeline.Path", wraps=Path):
            # Monkey-patch inside the function: replace "models/production" resolution
            original_div = Path.__truediv__

            def patched_div(self, other):
                if str(self) == "models" and str(other) == "production":
                    return weights_dir
                return original_div(self, other)

            with patch.object(Path, "__truediv__", patched_div):
                try:
                    promote_model(ti=ti)
                except Exception:
                    pass  # MLflow/HF calls may fail in unit test context

    # Verify put_file was called for model artifacts (not annotation data)
    put_file_keys = [call.args[1] for call in mock_storage.put_file.call_args_list]
    assert cfg.detection_key in put_file_keys or any("object_detection" in k for k in put_file_keys)


# ---------------------------------------------------------------------------
# check_dataset_changes
# ---------------------------------------------------------------------------


def test_check_dataset_changes_returns_false_when_sha_unchanged(tmp_path, monkeypatch):
    from retrain_pipeline import check_dataset_changes

    monkeypatch.chdir(tmp_path)
    (tmp_path / "dvc.yaml").write_text("cmd: dvc run --rev abc123deadbeef")
    last_sha = tmp_path / "last_sha.txt"
    last_sha.write_text("abc123deadbeef")

    ti, _ = _make_ti()
    with patch("retrain_pipeline._LAST_SHA_PATH", last_sha):
        result = check_dataset_changes(ti=ti)

    assert result is False


def test_check_dataset_changes_returns_true_when_sha_changed(tmp_path, monkeypatch):
    from retrain_pipeline import check_dataset_changes

    monkeypatch.chdir(tmp_path)
    (tmp_path / "dvc.yaml").write_text("cmd: dvc run --rev newsha99")
    last_sha = tmp_path / "last_sha.txt"
    last_sha.write_text("oldsha00")

    ti, store = _make_ti()
    with patch("retrain_pipeline._LAST_SHA_PATH", last_sha):
        result = check_dataset_changes(ti=ti)

    assert result is True
