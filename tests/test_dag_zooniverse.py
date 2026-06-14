"""Unit tests for zooniverse_ingest DAG task callables (no Airflow scheduler needed)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ti(xcom_store: dict | None = None):
    """Create a minimal TaskInstance mock with xcom push/pull support."""
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
# filter_by_consensus
# ---------------------------------------------------------------------------


def _run_filter_by_consensus(records, threshold=0.8):
    """Call filter_by_consensus callable with synthetic context."""
    from zooniverse_ingest import filter_by_consensus

    xcom_store = {"records": records}
    ti, store = _make_ti(xcom_store)

    with patch(
        "allium_cepa_classifier.config.ZooniverseConfig.consensus_threshold",
        new_callable=lambda: property(lambda self: threshold),
    ):
        result = filter_by_consensus(ti=ti)

    return result, store.get("accepted", [])


class TestConsensusFilter:
    def test_unanimous_passes(self):
        records = [
            {"subject_id": "s1", "annotation_id": "a1", "split": "train", "value": "prophase"},
            {"subject_id": "s1", "annotation_id": "a1", "split": "train", "value": "prophase"},
            {"subject_id": "s1", "annotation_id": "a1", "split": "train", "value": "prophase"},
        ]
        from zooniverse_ingest import filter_by_consensus
        xcom_store = {"records": records}
        ti, store = _make_ti(xcom_store)
        result = filter_by_consensus(ti=ti)
        assert result is True
        assert len(store["accepted"]) == 1
        assert store["accepted"][0]["phase"] == "prophase"

    def test_below_threshold_discarded(self):
        records = [
            {"subject_id": "s1", "annotation_id": "a1", "split": "train", "value": "prophase"},
            {"subject_id": "s1", "annotation_id": "a1", "split": "train", "value": "metaphase"},
            {"subject_id": "s1", "annotation_id": "a1", "split": "train", "value": "interphase"},
        ]
        from zooniverse_ingest import filter_by_consensus
        xcom_store = {"records": records}
        ti, store = _make_ti(xcom_store)
        result = filter_by_consensus(ti=ti)
        assert result is False
        assert store["accepted"] == []

    def test_short_circuits_on_empty_accepted(self):
        from zooniverse_ingest import filter_by_consensus
        xcom_store = {"records": []}
        ti, store = _make_ti(xcom_store)
        result = filter_by_consensus(ti=ti)
        assert result is False

    def test_majority_two_thirds(self):
        records = [
            {"subject_id": "s1", "annotation_id": "a1", "split": "train", "value": "anaphase"},
            {"subject_id": "s1", "annotation_id": "a1", "split": "train", "value": "anaphase"},
            {"subject_id": "s1", "annotation_id": "a1", "split": "train", "value": "interphase"},
        ]
        from zooniverse_ingest import filter_by_consensus
        xcom_store = {"records": records}
        ti, store = _make_ti(xcom_store)
        # 2/3 ≈ 0.667 < default threshold 0.8 → discarded
        result = filter_by_consensus(ti=ti)
        assert result is False


# ---------------------------------------------------------------------------
# apply_phase_to_annotations — phase→division mapping
# ---------------------------------------------------------------------------


DIVIDING_PHASES = ["prophase", "metaphase", "anaphase", "telophase", "chromosomal_aberration"]
NON_DIVIDING_PHASES = ["interphase"]
NO_OP_PHASES = ["indeterminate", "not_a_cell"]


@pytest.mark.parametrize("phase", DIVIDING_PHASES)
def test_dividing_phase_sets_division_1(phase, tmp_path):
    from zooniverse_ingest import apply_phase_to_annotations

    accepted = [{"annotation_id": "a1", "split": "train", "phase": phase, "agreement": 1.0}]
    xcom_store = {"accepted": accepted}
    ti, store = _make_ti(xcom_store)

    with patch("zooniverse_ingest.Path", return_value=tmp_path):
        apply_phase_to_annotations(ti=ti)

    patches = store["patches_by_split"]["train"]
    assert len(patches) == 1
    assert patches[0]["attributes"]["division"] == 1
    assert patches[0]["attributes"]["phase"] == phase


def test_interphase_sets_division_0(tmp_path):
    from zooniverse_ingest import apply_phase_to_annotations

    accepted = [{"annotation_id": "a1", "split": "train", "phase": "interphase", "agreement": 1.0}]
    xcom_store = {"accepted": accepted}
    ti, store = _make_ti(xcom_store)

    with patch("zooniverse_ingest.Path", return_value=tmp_path):
        apply_phase_to_annotations(ti=ti)

    patches = store["patches_by_split"]["train"]
    assert patches[0]["attributes"]["division"] == 0


def test_indeterminate_writes_phase_but_not_division(tmp_path):
    from zooniverse_ingest import apply_phase_to_annotations

    accepted = [{"annotation_id": "a1", "split": "train", "phase": "indeterminate", "agreement": 1.0}]
    xcom_store = {"accepted": accepted}
    ti, store = _make_ti(xcom_store)

    with patch("zooniverse_ingest.Path", return_value=tmp_path):
        apply_phase_to_annotations(ti=ti)

    patches = store["patches_by_split"]["train"]
    assert patches[0]["attributes"]["phase"] == "indeterminate"
    assert "division" not in patches[0]["attributes"]
    assert "detection_error" not in patches[0].get("attributes", {})


def test_not_a_cell_sets_detection_error_and_no_division(tmp_path):
    from zooniverse_ingest import apply_phase_to_annotations

    accepted = [{"annotation_id": "a1", "split": "train", "phase": "not_a_cell", "agreement": 1.0}]
    xcom_store = {"accepted": accepted}
    ti, store = _make_ti(xcom_store)

    with patch("zooniverse_ingest.Path", return_value=tmp_path):
        apply_phase_to_annotations(ti=ti)

    patches = store["patches_by_split"]["train"]
    assert patches[0]["attributes"]["detection_error"] is True
    assert "division" not in patches[0]["attributes"]


# ---------------------------------------------------------------------------
# _enough_patches short-circuit
# ---------------------------------------------------------------------------


def test_enough_patches_true_when_above_min():
    from zooniverse_ingest import _enough_patches

    patches_by_split = {"train": [{"id": f"a{i}"} for i in range(15)]}
    xcom_store = {"patches_by_split": patches_by_split}
    ti, _ = _make_ti(xcom_store)
    assert _enough_patches(ti=ti) is True


def test_enough_patches_false_when_below_min():
    from zooniverse_ingest import _enough_patches

    patches_by_split = {"train": [{"id": "a1"}]}  # 1 < default min_new_for_patch=10
    xcom_store = {"patches_by_split": patches_by_split}
    ti, _ = _make_ti(xcom_store)
    assert _enough_patches(ti=ti) is False


# ---------------------------------------------------------------------------
# patch_huggingface_annotations calls hub once per split
# ---------------------------------------------------------------------------


def test_patch_hf_called_once_per_split(tmp_path):
    from zooniverse_ingest import patch_huggingface_annotations

    from allium_cepa_classifier.providers.dataset_hub import MockHub

    hub = MockHub(base_dir=tmp_path)
    patches_by_split = {
        "train": [{"id": "a1", "attributes": {"phase": "prophase"}}],
        "validation": [{"id": "a2", "attributes": {"phase": "interphase"}}],
    }
    xcom_store = {"patches_by_split": patches_by_split}
    ti, store = _make_ti(xcom_store)

    with patch("allium_cepa_classifier.providers.factory.get_dataset_hub", return_value=hub):
        patch_huggingface_annotations(ti=ti)

    # Both splits should now have annotations.json
    assert (tmp_path / "train" / "annotations.json").exists()
    assert (tmp_path / "validation" / "annotations.json").exists()
    assert store.get("new_sha") is not None


# ---------------------------------------------------------------------------
# update_dvc_yaml_sha is a no-op in mock mode
# ---------------------------------------------------------------------------


def test_update_dvc_yaml_sha_noop_in_mock_mode(monkeypatch):
    monkeypatch.setenv("ALLIUM_HUB", "mock")
    from zooniverse_ingest import update_dvc_yaml_sha

    ti, _ = _make_ti({})
    # Should not raise even though dvc.yaml need not exist
    update_dvc_yaml_sha(ti=ti)
