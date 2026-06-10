from __future__ import annotations

import os

from .storage import FsspecStorage


def get_storage() -> FsspecStorage:
    """Return a storage backend configured from environment variables.

    ALLIUM_STORAGE=minio (default) — MinIO at MINIO_ENDPOINT (default localhost:9000)
    ALLIUM_STORAGE=s3             — AWS S3 (no endpoint override; uses normal AWS credential chain)
    ALLIUM_BUCKET                 — bucket name (default allium-cepa-ml)
    """
    backend = os.getenv("ALLIUM_STORAGE", "minio")
    bucket = os.getenv("ALLIUM_BUCKET", "allium-cepa-ml")

    if backend == "s3":
        return FsspecStorage(bucket=bucket, endpoint_url=None, key=None, secret=None)

    endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    key = os.getenv("MINIO_ROOT_USER", "minioadmin")
    secret = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
    return FsspecStorage(bucket=bucket, endpoint_url=endpoint, key=key, secret=secret)
