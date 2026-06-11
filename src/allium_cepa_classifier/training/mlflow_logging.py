"""
Optional MLflow logging helper.

Loads MLFLOW_TRACKING_URI from a project-root .env file automatically, so no
manual export is needed. No-ops silently when the var is absent (e.g. CI without
Docker), so the codebase works without MLflow running.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)


def _is_enabled() -> bool:
    return bool(os.getenv("MLFLOW_TRACKING_URI"))


class _NoOpRun:
    """Returned when MLflow is disabled — all calls are silent no-ops."""

    run_id: str | None = None

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float]) -> None:
        pass

    def log_artifact(self, path: Path) -> None:
        pass

    def log_model_file(self, local_path: Path, artifact_subdir: str) -> None:
        pass


class _CalibratedCheckpointModel:
    """Minimal pyfunc wrapper — weights are stored as an artifact, loaded via AlliumCepaModel."""

    def predict(self, context, model_input):  # noqa: ARG002
        raise NotImplementedError("Load via AlliumCepaModel, not MLflow pyfunc.")


class _MlflowRun:
    def __init__(self, run_id: str) -> None:
        import mlflow

        self._mlflow = mlflow
        self.run_id = run_id

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._mlflow.log_metrics(metrics)

    def log_artifact(self, path: Path) -> None:
        if Path(path).exists():
            self._mlflow.log_artifact(str(path))

    def log_model_file(self, local_path: Path, artifact_subdir: str) -> None:
        """Log a .pt file as a pyfunc model so it can be registered in the Model Registry."""
        if not Path(local_path).exists():
            return
        self._mlflow.pyfunc.log_model(
            artifact_path=artifact_subdir,
            python_model=_CalibratedCheckpointModel(),
            artifacts={"weights": str(local_path)},
        )


@contextmanager
def run(
    run_name: str | None = None, tags: dict[str, str] | None = None
) -> Generator[_NoOpRun | _MlflowRun, None, None]:
    """Context manager for an MLflow run. No-op when MLFLOW_TRACKING_URI is unset."""
    if not _is_enabled():
        yield _NoOpRun()
        return
    import mlflow

    with mlflow.start_run(run_name=run_name, tags=tags) as active_run:
        yield _MlflowRun(active_run.info.run_id)
