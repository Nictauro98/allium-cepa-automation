from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


class MockHub:
    """Writes images to a local dir and merges annotations into a local annotations.json.

    Returns a deterministic SHA derived from the serialized annotations so tests can assert
    on stable values without network access.
    """

    def __init__(self, base_dir: Path | None = None):
        self._base = base_dir or Path(os.getenv("MOCK_HUB_DIR", "/tmp/mock_hub"))
        self._base.mkdir(parents=True, exist_ok=True)

    def _annotations_path(self, split: str) -> Path:
        path = self._base / split / "annotations.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _load_annotations(self, split: str) -> dict:
        path = self._annotations_path(split)
        if path.exists():
            return json.loads(path.read_text())
        return {"images": [], "annotations": [], "categories": []}

    def _save_annotations(self, split: str, data: dict) -> str:
        path = self._annotations_path(split)
        serialized = json.dumps(data, sort_keys=True)
        path.write_text(serialized)
        return hashlib.sha1(serialized.encode()).hexdigest()

    def publish_images(self, split: str, images: list[Path], annotations: list[dict]) -> str:
        images_dir = self._base / split / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for img in images:
            shutil.copy2(img, images_dir / Path(img).name)

        data = self._load_annotations(split)
        existing_ids = {a["id"] for a in data["annotations"]}
        for ann in annotations:
            if ann.get("id") not in existing_ids:
                data["annotations"].append(ann)
        return self._save_annotations(split, data)

    def patch_annotations(self, split: str, annotation_patches: list[dict]) -> str:
        data = self._load_annotations(split)
        index = {a["id"]: a for a in data["annotations"]}
        for patch in annotation_patches:
            ann_id = patch["id"]
            if ann_id in index:
                index[ann_id].update(patch)
        data["annotations"] = list(index.values())
        return self._save_annotations(split, data)


class RealHFHub:
    """HuggingFace Hub: publish images (parquet shards) or patch annotations.json.

    Requires HF_TOKEN env var and HF_DATASET_REPO (e.g. "GIAR-UTN/allium-cepa-dataset").
    """

    def __init__(self) -> None:
        from huggingface_hub import HfApi

        self._api = HfApi(token=os.environ["HF_TOKEN"])
        self._repo = os.environ["HF_DATASET_REPO"]

    def publish_images(self, split: str, images: list[Path], annotations: list[dict]) -> str:
        import tempfile

        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Write images as a parquet shard (file_name + bytes columns)
            names, byte_cols = [], []
            for img in images:
                img = Path(img)
                names.append(img.name)
                byte_cols.append(img.read_bytes())

            table = pa.table({"file_name": names, "image_bytes": byte_cols})
            shard_path = tmp / f"shard-{_content_hash(names)}.parquet"
            pq.write_table(table, str(shard_path))

            # Fetch + merge existing annotations
            anns_path = self._fetch_annotations(split, tmp)
            data: dict = json.loads(anns_path.read_text()) if anns_path.exists() else {
                "images": [], "annotations": [], "categories": []
            }
            existing_ids = {a["id"] for a in data["annotations"]}
            for ann in annotations:
                if ann.get("id") not in existing_ids:
                    data["annotations"].append(ann)
            anns_path.write_text(json.dumps(data))

            commit = self._api.upload_large_folder(
                repo_id=self._repo,
                repo_type="dataset",
                folder_path=str(tmp),
                path_in_repo=f"{split}/data",
            )
        return commit

    def patch_annotations(self, split: str, annotation_patches: list[dict]) -> str:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            anns_path = self._fetch_annotations(split, tmp)
            data: dict = json.loads(anns_path.read_text()) if anns_path.exists() else {
                "images": [], "annotations": [], "categories": []
            }
            index = {a["id"]: a for a in data["annotations"]}
            for patch in annotation_patches:
                ann_id = patch["id"]
                if ann_id in index:
                    index[ann_id].update(patch)
            data["annotations"] = list(index.values())
            anns_path.write_text(json.dumps(data))

            commit = self._api.upload_file(
                path_or_fileobj=str(anns_path),
                path_in_repo=f"{split}/data/annotations.json",
                repo_id=self._repo,
                repo_type="dataset",
            )
        return commit.oid if hasattr(commit, "oid") else str(commit)

    def _fetch_annotations(self, split: str, tmp: Path) -> Path:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        local = tmp / "annotations.json"
        try:
            downloaded = hf_hub_download(
                repo_id=self._repo,
                filename=f"{split}/data/annotations.json",
                repo_type="dataset",
                token=self._api.token,
            )
            shutil.copy2(downloaded, local)
        except EntryNotFoundError:
            pass
        return local


def _content_hash(names: list[str]) -> str:
    return hashlib.sha1("|".join(sorted(names)).encode()).hexdigest()[:8]
