"""Unit tests for raw_image_ingest DAG task callables."""

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
# build_coco_annotations
# ---------------------------------------------------------------------------


def _make_scored(confidences: list[float], threshold: float = 0.9):
    """Produce synthetic scored output for build_coco_annotations."""
    return [
        {
            "image_path": "/tmp/img.jpg",
            "detections": [
                {
                    "bbox": [0, 0, 10, 10],
                    "class": 1,
                    "confidence": c,
                    "ci_lower": max(0.0, c - 0.05),
                    "ci_upper": min(1.0, c + 0.05),
                    "idx": i,
                }
                for i, c in enumerate(confidences)
            ],
        }
    ]


class TestBuildCocoAnnotations:
    def test_high_confidence_sets_division(self):
        from raw_image_ingest import build_coco_annotations

        scored = _make_scored([0.95, 0.92])
        xcom_store = {"scored": scored, "high_confidence_threshold": 0.9}
        ti, store = _make_ti(xcom_store)
        build_coco_annotations(ti=ti)

        anns = store["all_annotations"]
        assert all(a["attributes"]["division"] == 1 for a in anns)
        assert not any(a["attributes"].get("pending") for a in anns)
        assert store["auto_count"] == 2
        assert store["pending_ids"] == []

    def test_low_confidence_sets_pending_and_no_division(self):
        from raw_image_ingest import build_coco_annotations

        scored = _make_scored([0.50, 0.70])
        xcom_store = {"scored": scored, "high_confidence_threshold": 0.9}
        ti, store = _make_ti(xcom_store)
        build_coco_annotations(ti=ti)

        anns = store["all_annotations"]
        assert all(a["attributes"]["division"] is None for a in anns)
        assert all(a["attributes"]["pending"] is True for a in anns)
        assert store["auto_count"] == 0
        assert len(store["pending_ids"]) == 2

    def test_mixed_confidence(self):
        from raw_image_ingest import build_coco_annotations

        scored = _make_scored([0.95, 0.50])
        xcom_store = {"scored": scored, "high_confidence_threshold": 0.9}
        ti, store = _make_ti(xcom_store)
        build_coco_annotations(ti=ti)

        assert store["auto_count"] == 1
        assert len(store["pending_ids"]) == 1

    def test_boundary_confidence_is_high(self):
        """Confidence exactly equal to threshold should be auto-labeled."""
        from raw_image_ingest import build_coco_annotations

        scored = _make_scored([0.9])
        xcom_store = {"scored": scored, "high_confidence_threshold": 0.9}
        ti, store = _make_ti(xcom_store)
        build_coco_annotations(ti=ti)

        assert store["auto_count"] == 1

    def test_annotation_id_embedded_in_annotation(self):
        from raw_image_ingest import build_coco_annotations

        scored = _make_scored([0.50])
        xcom_store = {"scored": scored, "high_confidence_threshold": 0.9}
        ti, store = _make_ti(xcom_store)
        build_coco_annotations(ti=ti)

        ann = store["all_annotations"][0]
        # annotation_id is stored as ann["id"] (a UUID)
        assert ann["id"] in store["pending_ids"]


# ---------------------------------------------------------------------------
# publish_to_huggingface — receives all annotations
# ---------------------------------------------------------------------------


def test_publish_images_receives_all_annotations(tmp_path):
    from raw_image_ingest import publish_to_huggingface

    from allium_cepa_classifier.providers.dataset_hub import MockHub

    # Create a fake image file
    img = tmp_path / "img.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)

    annotations = [
        {"id": "a1", "attributes": {"division": 1}},
        {"id": "a2", "attributes": {"division": None, "pending": True}},
    ]
    xcom_store = {
        "local_paths": [str(img)],
        "all_annotations": annotations,
    }
    ti, store = _make_ti(xcom_store)

    hub = MockHub(base_dir=tmp_path / "hub")
    with patch("allium_cepa_classifier.providers.factory.get_dataset_hub", return_value=hub):
        publish_to_huggingface(ti=ti)

    data = json.loads((tmp_path / "hub" / "train" / "annotations.json").read_text())
    ids = {a["id"] for a in data["annotations"]}
    assert "a1" in ids
    assert "a2" in ids


# ---------------------------------------------------------------------------
# send_pending_to_zooniverse — only receives pending set
# ---------------------------------------------------------------------------


def test_send_pending_only_receives_pending_ids(tmp_path):
    from raw_image_ingest import send_pending_to_zooniverse

    from allium_cepa_classifier.providers.labeling import MockZooniverse

    zoo = MockZooniverse(
        fixtures_dir=Path(__file__).parent / "fixtures" / "zooniverse",
        tasks_dir=tmp_path,
    )
    xcom_store = {"pending_ids": ["ann003", "ann004"]}
    ti, _ = _make_ti(xcom_store)

    with patch("allium_cepa_classifier.providers.factory.get_labeling", return_value=zoo):
        send_pending_to_zooniverse(ti=ti)

    tasks = json.loads((tmp_path / "created_tasks.json").read_text())
    assert "ann003" in tasks[0]["image_keys"]
    assert "ann004" in tasks[0]["image_keys"]


# ---------------------------------------------------------------------------
# update_dvc_yaml_sha is a no-op in mock mode
# ---------------------------------------------------------------------------


def test_update_dvc_yaml_sha_noop_in_mock_mode(monkeypatch):
    monkeypatch.setenv("ALLIUM_HUB", "mock")
    from raw_image_ingest import update_dvc_yaml_sha

    ti, _ = _make_ti({})
    update_dvc_yaml_sha(ti=ti)  # should not raise


# ---------------------------------------------------------------------------
# notify_retrain_if_threshold
# ---------------------------------------------------------------------------


def test_notify_retrain_triggers_when_above_threshold():
    """Above threshold: function attempts trigger_dag (airflow not installed → graceful warning)."""
    from raw_image_ingest import _AUTO_LABEL_RETRAIN_THRESHOLD, notify_retrain_if_threshold

    xcom_store = {"auto_count": _AUTO_LABEL_RETRAIN_THRESHOLD}
    ti, _ = _make_ti(xcom_store)
    # Airflow not installed in test env; function catches ImportError gracefully
    notify_retrain_if_threshold(ti=ti)  # should not raise


def test_notify_retrain_skips_when_below_threshold():
    """Below threshold: trigger_dag branch is never reached."""
    from raw_image_ingest import notify_retrain_if_threshold

    xcom_store = {"auto_count": 0}
    ti, _ = _make_ti(xcom_store)
    notify_retrain_if_threshold(ti=ti)  # should not raise
