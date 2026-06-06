from __future__ import annotations

import os

from .storage import FsspecStorage


def get_storage() -> FsspecStorage:
    """Return a storage backend configured from environment variables.

    ALLIUM_STORAGE=minio (default) — MinIO at MINIO_ENDPOINT (default localhost:9000)
    ALLIUM_STORAGE=s3             — AWS S3 (no endpoint override)
    ALLIUM_BUCKET                 — bucket name (default allium-cepa-ml)
    """
    backend = os.getenv("ALLIUM_STORAGE", "minio")
    bucket = os.getenv("ALLIUM_BUCKET", "allium-cepa-ml")
    endpoint = None if backend == "s3" else os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    return FsspecStorage(bucket=bucket, endpoint_url=endpoint)
