"""Tests for the ingestion providers (labeling, drive, dataset_hub) factory selection."""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Factory defaults (mock with no env set)
# ---------------------------------------------------------------------------


def test_get_labeling_default_is_mock(monkeypatch):
    monkeypatch.delenv("ALLIUM_LABELING", raising=False)
    from allium_cepa_classifier.providers.factory import get_labeling

    provider = get_labeling()
    assert type(provider).__name__ == "MockZooniverse"


def test_get_drive_default_is_mock(monkeypatch):
    monkeypatch.delenv("ALLIUM_DRIVE", raising=False)
    from allium_cepa_classifier.providers.factory import get_drive

    provider = get_drive()
    assert type(provider).__name__ == "MockDrive"


def test_get_dataset_hub_default_is_mock(monkeypatch):
    monkeypatch.delenv("ALLIUM_HUB", raising=False)
    from allium_cepa_classifier.providers.factory import get_dataset_hub

    provider = get_dataset_hub()
    assert type(provider).__name__ == "MockHub"


# ---------------------------------------------------------------------------
# Env-switch selects real impl (class only, no network)
# ---------------------------------------------------------------------------


def test_get_labeling_zooniverse_selects_real(monkeypatch):
    monkeypatch.setenv("ALLIUM_LABELING", "zooniverse")
    monkeypatch.setenv("ZOONIVERSE_USERNAME", "user")
    monkeypatch.setenv("ZOONIVERSE_PASSWORD", "pass")

    import sys

    # Provide a stub so panoptes_client import doesn't fail in CI
    import types

    stub = types.ModuleType("panoptes_client")
    stub.Panoptes = type("Panoptes", (), {"connect": staticmethod(lambda **kw: None)})()
    stub.Project = None
    stub.Classification = None
    stub.SubjectSet = None
    stub.Subject = None
    sys.modules.setdefault("panoptes_client", stub)

    from allium_cepa_classifier.providers.factory import get_labeling

    provider = get_labeling()
    assert type(provider).__name__ == "RealZooniverse"


def test_get_drive_gdrive_selects_real(monkeypatch):
    monkeypatch.setenv("ALLIUM_DRIVE", "gdrive")
    monkeypatch.setenv("GOOGLE_DRIVE_CREDENTIALS_PATH", "/nonexistent/creds.json")

    import sys
    import types

    # Stub google.oauth2.service_account
    google_pkg = types.ModuleType("google")
    google_oauth2 = types.ModuleType("google.oauth2")
    google_sa = types.ModuleType("google.oauth2.service_account")

    class _FakeCreds:
        @staticmethod
        def from_service_account_file(path, scopes):
            return object()

    google_sa.Credentials = _FakeCreds
    sys.modules.setdefault("google", google_pkg)
    sys.modules.setdefault("google.oauth2", google_oauth2)
    sys.modules.setdefault("google.oauth2.service_account", google_sa)

    # Stub googleapiclient.discovery
    gapi = types.ModuleType("googleapiclient")
    gapi_disc = types.ModuleType("googleapiclient.discovery")
    gapi_disc.build = lambda *a, **kw: object()
    sys.modules.setdefault("googleapiclient", gapi)
    sys.modules.setdefault("googleapiclient.discovery", gapi_disc)

    from allium_cepa_classifier.providers.factory import get_drive

    provider = get_drive()
    assert type(provider).__name__ == "RealDrive"


def test_get_dataset_hub_hf_selects_real(monkeypatch):
    monkeypatch.setenv("ALLIUM_HUB", "hf")
    monkeypatch.setenv("HF_TOKEN", "hf_fake")
    monkeypatch.setenv("HF_DATASET_REPO", "org/dataset")

    import sys
    import types

    # Stub huggingface_hub.HfApi
    hf_stub = types.ModuleType("huggingface_hub")
    hf_stub.HfApi = type("HfApi", (), {"__init__": lambda self, token=None: None})
    sys.modules["huggingface_hub"] = hf_stub

    from allium_cepa_classifier.providers.factory import get_dataset_hub

    provider = get_dataset_hub()
    assert type(provider).__name__ == "RealHFHub"


# ---------------------------------------------------------------------------
# MockHub behaviour
# ---------------------------------------------------------------------------


def test_mock_hub_publish_images_writes_and_returns_sha(tmp_path):
    from allium_cepa_classifier.providers.dataset_hub import MockHub

    hub = MockHub(base_dir=tmp_path)

    # Create a tiny fake image
    img = tmp_path / "test.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)

    annotations = [{"id": "a1", "bbox": [0, 0, 10, 10], "attributes": {"division": 1}}]
    sha = hub.publish_images("train", [img], annotations)

    assert sha  # non-empty
    out = tmp_path / "train" / "annotations.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert any(a["id"] == "a1" for a in data["annotations"])


def test_mock_hub_patch_annotations_merges(tmp_path):
    from allium_cepa_classifier.providers.dataset_hub import MockHub

    hub = MockHub(base_dir=tmp_path)

    # Seed an annotation with division=None
    ann_path = tmp_path / "train" / "annotations.json"
    ann_path.parent.mkdir(parents=True)
    ann_path.write_text(
        json.dumps({"images": [], "annotations": [{"id": "a1", "attributes": {"division": None}}], "categories": []})
    )

    sha = hub.patch_annotations("train", [{"id": "a1", "attributes": {"division": 1, "phase": "prophase"}}])
    assert sha

    data = json.loads(ann_path.read_text())
    ann = next(a for a in data["annotations"] if a["id"] == "a1")
    assert ann["attributes"]["division"] == 1
    assert ann["attributes"]["phase"] == "prophase"


def test_mock_hub_patch_annotations_returns_deterministic_sha(tmp_path):
    from allium_cepa_classifier.providers.dataset_hub import MockHub

    hub = MockHub(base_dir=tmp_path)
    ann_path = tmp_path / "train" / "annotations.json"
    ann_path.parent.mkdir(parents=True)
    ann_path.write_text(
        json.dumps({"images": [], "annotations": [{"id": "a1", "attributes": {}}], "categories": []})
    )

    # Two patches that produce different states → different SHAs
    sha1 = hub.patch_annotations("train", [{"id": "a1", "attributes": {"phase": "interphase"}}])
    sha2 = hub.patch_annotations("train", [{"id": "a1", "attributes": {"phase": "prophase"}}])
    assert sha1 != sha2

    # Reverting to the same state as sha2 returns the same SHA (deterministic)
    sha3 = hub.patch_annotations("train", [])  # no-op → state unchanged → same SHA as sha2
    assert sha3 == sha2


# ---------------------------------------------------------------------------
# MockDrive behaviour
# ---------------------------------------------------------------------------


def test_mock_drive_list_new_returns_fixture_files():
    from allium_cepa_classifier.providers.drive import MockDrive

    fixtures = Path(__file__).parent / "fixtures" / "drive"
    drive = MockDrive(fixtures_dir=fixtures)
    files = drive.list_new()
    assert "sample_image.jpg" in files


def test_mock_drive_download_copies_file(tmp_path):
    from allium_cepa_classifier.providers.drive import MockDrive

    fixtures = Path(__file__).parent / "fixtures" / "drive"
    drive = MockDrive(fixtures_dir=fixtures)
    dest = tmp_path / "downloaded.jpg"
    result = drive.download("sample_image.jpg", dest)
    assert result == dest
    assert dest.exists()


# ---------------------------------------------------------------------------
# MockZooniverse behaviour
# ---------------------------------------------------------------------------


def test_mock_zooniverse_fetch_classifications_returns_fixture_data():
    from allium_cepa_classifier.providers.labeling import MockZooniverse

    fixtures = Path(__file__).parent / "fixtures" / "zooniverse"
    zoo = MockZooniverse(fixtures_dir=fixtures)
    records = zoo.fetch_classifications()
    assert len(records) > 0
    assert all("annotation_id" in r for r in records)


def test_mock_zooniverse_create_tasks_writes_to_dir(tmp_path):
    from allium_cepa_classifier.providers.labeling import MockZooniverse

    fixtures = Path(__file__).parent / "fixtures" / "zooniverse"
    zoo = MockZooniverse(fixtures_dir=fixtures, tasks_dir=tmp_path)
    zoo.create_tasks(["ann001", "ann002"], priority="high")
    out = tmp_path / "created_tasks.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data[0]["priority"] == "high"
    assert "ann001" in data[0]["image_keys"]
