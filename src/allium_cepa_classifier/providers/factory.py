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


def get_labeling():
    """Return a labeling backend.

    ALLIUM_LABELING=mock (default) — MockZooniverse using fixture files
    ALLIUM_LABELING=zooniverse     — RealZooniverse via panoptes-client
    """
    from .labeling import MockZooniverse, RealZooniverse

    backend = os.getenv("ALLIUM_LABELING", "mock")
    if backend == "zooniverse":
        return RealZooniverse()
    return MockZooniverse()


def get_drive():
    """Return a drive backend.

    ALLIUM_DRIVE=mock   (default) — MockDrive using fixture files
    ALLIUM_DRIVE=gdrive           — RealDrive via google-api-python-client
    """
    from .drive import MockDrive, RealDrive

    backend = os.getenv("ALLIUM_DRIVE", "mock")
    if backend == "gdrive":
        return RealDrive()
    return MockDrive()


def get_dataset_hub():
    """Return a dataset hub backend.

    ALLIUM_HUB=mock (default) — MockHub writing to a local directory
    ALLIUM_HUB=hf             — RealHFHub via huggingface-hub
    """
    from .dataset_hub import MockHub, RealHFHub

    backend = os.getenv("ALLIUM_HUB", "mock")
    if backend == "hf":
        return RealHFHub()
    return MockHub()
