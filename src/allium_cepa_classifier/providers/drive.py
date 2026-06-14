from __future__ import annotations

import os
import shutil
from pathlib import Path


class MockDrive:
    """Lists and downloads from a local fixtures directory."""

    def __init__(self, fixtures_dir: Path | None = None):
        self._fixtures_dir = fixtures_dir or (
            Path(__file__).parents[4] / "tests" / "fixtures" / "drive"
        )

    def list_new(self, since: str | None = None) -> list[str]:
        if not self._fixtures_dir.exists():
            return []
        return [p.name for p in sorted(self._fixtures_dir.iterdir()) if p.is_file()]

    def download(self, file_id: str, local_path: Path) -> Path:
        src = self._fixtures_dir / file_id
        if not src.exists():
            raise FileNotFoundError(f"MockDrive: fixture not found: {src}")
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)
        return local_path


class RealDrive:
    """Google Drive via google-api-python-client. Requires GOOGLE_DRIVE_CREDENTIALS_PATH."""

    def __init__(self) -> None:
        from google.oauth2 import service_account  # noqa: F401
        from googleapiclient.discovery import build  # noqa: F401

        creds_path = os.environ["GOOGLE_DRIVE_CREDENTIALS_PATH"]
        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        credentials = service_account.Credentials.from_service_account_file(
            creds_path, scopes=scopes
        )
        self._service = build("drive", "v3", credentials=credentials)
        self._folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")

    def list_new(self, since: str | None = None) -> list[str]:
        query = f"'{self._folder_id}' in parents and trashed = false"
        if since:
            query += f" and createdTime > '{since}'"
        result = (
            self._service.files()
            .list(q=query, fields="files(id,name)", orderBy="createdTime")
            .execute()
        )
        return [f["id"] for f in result.get("files", [])]

    def download(self, file_id: str, local_path: Path) -> Path:
        from googleapiclient.http import MediaIoBaseDownload

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        request = self._service.files().get_media(fileId=file_id)
        with open(local_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return local_path
