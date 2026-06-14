from __future__ import annotations

from pathlib import Path
from typing import Protocol


class StorageProvider(Protocol):
    """Minimal object-store interface. Paths are bucket-relative keys."""

    def get_file(self, key: str, local_path: Path) -> Path: ...
    def put_file(self, local_path: Path, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...
    def read_text(self, key: str) -> str: ...


class LabelingProvider(Protocol):
    """Expert-labeling backend (Zooniverse)."""

    def fetch_classifications(self, since: str | None) -> list[dict]: ...
    def create_tasks(self, image_keys: list[str], priority: str = "normal") -> None: ...


class DriveProvider(Protocol):
    """Raw-image source (Google Drive)."""

    def list_new(self, since: str | None) -> list[str]: ...
    def download(self, file_id: str, local_path: Path) -> Path: ...


class DatasetHubProvider(Protocol):
    """Ground-truth dataset hub (HuggingFace). Both methods return the new commit SHA.

    Two distinct operations:
    - publish_images: upload new full images + their COCO annotations (Drive-originated).
      New annotations carry division=None for pending crops sent to Zooniverse,
      or division set for high-confidence auto-labeled crops.
    - patch_annotations: update attributes on existing annotations only (Zooniverse verdicts).
      Never uploads image data.
    """

    def publish_images(self, split: str, images: list[Path], annotations: list[dict]) -> str: ...
    def patch_annotations(self, split: str, annotation_patches: list[dict]) -> str: ...
