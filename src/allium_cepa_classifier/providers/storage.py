from __future__ import annotations

from pathlib import Path

import fsspec


class FsspecStorage:
    """Single impl for local/MinIO/S3 — differs only by endpoint_url."""

    def __init__(self, bucket: str, endpoint_url: str | None = None):
        self.bucket = bucket
        self.fs = fsspec.filesystem(
            "s3",
            client_kwargs={"endpoint_url": endpoint_url} if endpoint_url else {},
        )

    def _key(self, key: str) -> str:
        return f"{self.bucket}/{key}"

    def get_file(self, key: str, local_path: Path) -> Path:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.fs.get(self._key(key), str(local_path))
        return local_path

    def put_file(self, local_path: Path, key: str) -> None:
        self.fs.put(str(local_path), self._key(key))

    def exists(self, key: str) -> bool:
        return self.fs.exists(self._key(key))

    def read_text(self, key: str) -> str:
        with self.fs.open(self._key(key), "r") as f:
            return f.read()
