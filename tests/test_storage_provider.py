"""Tests for the storage provider factory — no network, mocks fsspec."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from allium_cepa_classifier.providers.factory import get_storage
from allium_cepa_classifier.providers.storage import FsspecStorage


class TestGetStorage:
    def test_returns_fsspec_storage(self, monkeypatch):
        monkeypatch.delenv("ALLIUM_STORAGE", raising=False)
        monkeypatch.delenv("ALLIUM_BUCKET", raising=False)
        monkeypatch.delenv("MINIO_ENDPOINT", raising=False)

        with patch("allium_cepa_classifier.providers.storage.fsspec.filesystem") as mock_fs:
            mock_fs.return_value = MagicMock()
            storage = get_storage()

        assert isinstance(storage, FsspecStorage)

    def test_defaults_to_minio_backend(self, monkeypatch):
        monkeypatch.delenv("ALLIUM_STORAGE", raising=False)
        monkeypatch.delenv("MINIO_ENDPOINT", raising=False)

        with patch("allium_cepa_classifier.providers.storage.fsspec.filesystem") as mock_fs:
            mock_fs.return_value = MagicMock()
            storage = get_storage()

        call_kwargs = mock_fs.call_args
        assert call_kwargs[1]["client_kwargs"]["endpoint_url"] == "http://localhost:9000"

    def test_minio_endpoint_uses_env_var(self, monkeypatch):
        monkeypatch.setenv("ALLIUM_STORAGE", "minio")
        monkeypatch.setenv("MINIO_ENDPOINT", "http://minio:9000")

        with patch("allium_cepa_classifier.providers.storage.fsspec.filesystem") as mock_fs:
            mock_fs.return_value = MagicMock()
            storage = get_storage()

        call_kwargs = mock_fs.call_args
        assert call_kwargs[1]["client_kwargs"]["endpoint_url"] == "http://minio:9000"

    def test_s3_backend_has_no_endpoint(self, monkeypatch):
        monkeypatch.setenv("ALLIUM_STORAGE", "s3")

        with patch("allium_cepa_classifier.providers.storage.fsspec.filesystem") as mock_fs:
            mock_fs.return_value = MagicMock()
            storage = get_storage()

        call_kwargs = mock_fs.call_args
        assert call_kwargs[1]["client_kwargs"] == {}

    def test_bucket_name_from_env(self, monkeypatch):
        monkeypatch.setenv("ALLIUM_BUCKET", "my-custom-bucket")
        monkeypatch.delenv("ALLIUM_STORAGE", raising=False)

        with patch("allium_cepa_classifier.providers.storage.fsspec.filesystem") as mock_fs:
            mock_fs.return_value = MagicMock()
            storage = get_storage()

        assert storage.bucket == "my-custom-bucket"

    def test_default_bucket_name(self, monkeypatch):
        monkeypatch.delenv("ALLIUM_BUCKET", raising=False)
        monkeypatch.delenv("ALLIUM_STORAGE", raising=False)

        with patch("allium_cepa_classifier.providers.storage.fsspec.filesystem") as mock_fs:
            mock_fs.return_value = MagicMock()
            storage = get_storage()

        assert storage.bucket == "allium-cepa-ml"
