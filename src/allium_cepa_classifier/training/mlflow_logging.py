"""
Optional MLflow logging helper.

No-ops silently when MLFLOW_TRACKING_URI is unset, so local dev without
Docker still works. Set MLFLOW_TRACKING_URI=http://localhost:5000 to enable.
"""
from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _is_enabled() -> bool:
    return bool(os.getenv("MLFLOW_TRACKING_URI"))


class _NoOpRun:
    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float]) -> None:
        pass

    def log_artifact(self, path: Path) -> None:
        pass


class _MlflowRun:
    def __init__(self) -> None:
        import mlflow

        self._mlflow = mlflow

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._mlflow.log_metrics(metrics)

    def log_artifact(self, path: Path) -> None:
        if Path(path).exists():
            self._mlflow.log_artifact(str(path))


@contextmanager
def run(
    run_name: str | None = None, tags: dict[str, str] | None = None
) -> Generator[_NoOpRun | _MlflowRun, None, None]:
    """Context manager for an MLflow run. No-op when MLFLOW_TRACKING_URI is unset."""
    if not _is_enabled():
        yield _NoOpRun()
        return
    import mlflow

    with mlflow.start_run(run_name=run_name, tags=tags):
        yield _MlflowRun()
