from __future__ import annotations

import json
import os
from pathlib import Path


class MockZooniverse:
    """Reads classifications from a local fixtures file; records created tasks in a local dir."""

    def __init__(self, fixtures_dir: Path | None = None, tasks_dir: Path | None = None):
        self._fixtures_dir = fixtures_dir or (
            Path(__file__).parents[4] / "tests" / "fixtures" / "zooniverse"
        )
        self._tasks_dir = tasks_dir or Path(os.getenv("MOCK_TASKS_DIR", "/tmp/mock_zooniverse_tasks"))
        self._tasks_dir.mkdir(parents=True, exist_ok=True)

    def fetch_classifications(self, since: str | None = None) -> list[dict]:
        path = self._fixtures_dir / "classifications.json"
        if not path.exists():
            return []
        data: list[dict] = json.loads(path.read_text())
        if since:
            data = [r for r in data if r.get("created_at", "") >= since]
        return data

    def create_tasks(self, image_keys: list[str], priority: str = "normal") -> None:
        record = {"image_keys": image_keys, "priority": priority}
        out = self._tasks_dir / "created_tasks.json"
        existing: list[dict] = json.loads(out.read_text()) if out.exists() else []
        existing.append(record)
        out.write_text(json.dumps(existing, indent=2))


class RealZooniverse:
    """Zooniverse via panoptes-client. Requires ZOONIVERSE_USERNAME + ZOONIVERSE_PASSWORD."""

    def __init__(self) -> None:
        import panoptes_client  # noqa: F401 — deferred; not installed in the inference image

        username = os.environ["ZOONIVERSE_USERNAME"]
        password = os.environ["ZOONIVERSE_PASSWORD"]
        panoptes_client.Panoptes.connect(username=username, password=password)
        self._project_id = os.getenv("ZOONIVERSE_PROJECT_ID", "")

    def fetch_classifications(self, since: str | None = None) -> list[dict]:
        import panoptes_client

        project = panoptes_client.Project.find(self._project_id)
        kwargs: dict = {"project": project}
        if since:
            kwargs["created_after"] = since
        return [c.raw for c in panoptes_client.Classification.where(**kwargs)]

    def create_tasks(self, image_keys: list[str], priority: str = "normal") -> None:
        import panoptes_client

        project = panoptes_client.Project.find(self._project_id)
        subject_set = panoptes_client.SubjectSet()
        subject_set.links.project = project
        subject_set.display_name = f"auto-{priority}"
        subject_set.save()

        subjects = []
        for key in image_keys:
            s = panoptes_client.Subject()
            s.links.project = project
            s.add_location({"image/jpeg": key})
            s.metadata.update({"annotation_id": key.split("_")[-1].replace(".jpg", "")})
            s.save()
            subjects.append(s)

        subject_set.add(subjects)
