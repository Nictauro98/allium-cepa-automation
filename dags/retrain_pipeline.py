"""DAG C: retrain_pipeline

Triggered by zooniverse_ingest or raw_image_ingest (or manually):
check for dataset changes → dvc repro → read validation_result.json →
promote or archive the model.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

try:
    from airflow import DAG
    from airflow.operators.python import BranchPythonOperator, PythonOperator, ShortCircuitOperator
    from airflow.utils.dates import days_ago

    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False

log = logging.getLogger(__name__)

_VALIDATION_RESULT_PATH = Path("validation_result.json")
_LAST_SHA_PATH = Path("/tmp/allium_last_trained_sha.txt")


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def check_dataset_changes(**context) -> bool:
    """Short-circuit if dvc.yaml SHA hasn't changed since the last training run."""
    import re

    dvc_yaml = Path("dvc.yaml")
    if not dvc_yaml.exists():
        log.warning("dvc.yaml not found; proceeding with repro")
        return True

    text = dvc_yaml.read_text()
    m = re.search(r"--rev\s+(\S+)", text)
    current_sha = m.group(1) if m else None

    if _LAST_SHA_PATH.exists():
        last_sha = _LAST_SHA_PATH.read_text().strip()
        if current_sha == last_sha:
            log.info("Dataset SHA unchanged (%s); skipping retrain", current_sha)
            return False

    log.info("Dataset SHA changed to %s; proceeding with retrain", current_sha)
    context["ti"].xcom_push(key="current_sha", value=current_sha)
    return True


def run_dvc_repro(**context) -> None:
    """Run dvc repro — runs all affected stages including train/calibrate/evaluate/validate."""
    log.info("Running dvc repro…")
    subprocess.run(["dvc", "repro"], check=True)
    log.info("dvc repro complete")


def read_validation_result(**context) -> str:
    """Load validation_result.json; branch to promote or archive."""
    if not _VALIDATION_RESULT_PATH.exists():
        log.error("validation_result.json not found")
        return "archive_model"

    result = json.loads(_VALIDATION_RESULT_PATH.read_text())
    approved: bool = result.get("approved", False)
    log.info("Validation result: approved=%s  metrics=%s", approved, result.get("metrics", {}))
    context["ti"].xcom_push(key="validation_result", value=result)
    return "promote_model" if approved else "archive_model"


def promote_model(**context) -> None:
    """Copy weight files to models/production/, update metrics.json, restart HF Space."""
    from allium_cepa_classifier.config import ProductionConfig
    from allium_cepa_classifier.providers.factory import get_storage

    cfg = ProductionConfig()
    storage = get_storage()
    validation_result = context["ti"].xcom_pull(
        task_ids="read_validation_result", key="validation_result"
    )

    weights_dir = Path("models") / "production"
    weight_files = {
        cfg.detection_key: weights_dir / "object_detection.pt",
        cfg.classifier_key: weights_dir / "classifier_calibrated.pt",
        cfg.calibrator_key: weights_dir / "yolo_isotonic_calibrator.pkl",
    }

    for key, local_path in weight_files.items():
        if local_path.exists():
            storage.put_file(local_path, key)
            log.info("Uploaded %s → %s", local_path, key)

    # Update metrics.json
    metrics_path = weights_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(validation_result.get("metrics", {}), indent=2))
    storage.put_file(metrics_path, cfg.metrics_key)

    # MLflow: transition model version to Production (best-effort)
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("allium_cepa", stages=["Staging", "None"])
        if versions:
            client.transition_model_version_stage(
                name="allium_cepa",
                version=versions[0].version,
                stage="Production",
            )
            log.info("MLflow model version %s → Production", versions[0].version)
    except Exception as exc:
        log.warning("MLflow promotion skipped: %s", exc)

    # HF Space restart (real mode only)
    import os

    if os.getenv("ALLIUM_HUB", "mock") != "mock" and os.getenv("HF_SPACE_ID"):
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.restart_space(repo_id=os.environ["HF_SPACE_ID"])
        log.info("HF Space restarted")

    # Record the SHA we trained on
    current_sha = context["ti"].xcom_pull(
        task_ids="check_dataset_changes", key="current_sha"
    )
    if current_sha:
        _LAST_SHA_PATH.write_text(current_sha)

    log.info("Model promoted successfully")


def archive_model(**context) -> None:
    """Mark the MLflow version Archived; log comparative metrics."""
    validation_result = context["ti"].xcom_pull(
        task_ids="read_validation_result", key="validation_result"
    )
    log.info("Model rejected. Reasons: %s", validation_result.get("reasons", []))

    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("allium_cepa", stages=["Staging", "None"])
        if versions:
            client.transition_model_version_stage(
                name="allium_cepa",
                version=versions[0].version,
                stage="Archived",
            )
            log.info("MLflow model version %s → Archived", versions[0].version)
    except Exception as exc:
        log.warning("MLflow archiving skipped: %s", exc)


# ---------------------------------------------------------------------------
# DAG definition (skipped when Airflow is not installed, e.g. in unit tests)
# ---------------------------------------------------------------------------

if _AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="retrain_pipeline",
        description="Retrain + validate + promote/archive the AlliumCepa model",
        schedule=None,
        start_date=days_ago(1),
        catchup=False,
        tags=["training", "mlops"],
    ) as dag:
        t_check = ShortCircuitOperator(
            task_id="check_dataset_changes",
            python_callable=check_dataset_changes,
        )
        t_repro = PythonOperator(
            task_id="run_dvc_repro",
            python_callable=run_dvc_repro,
            execution_timeout=None,
        )
        t_read = BranchPythonOperator(
            task_id="read_validation_result",
            python_callable=read_validation_result,
        )
        t_promote = PythonOperator(
            task_id="promote_model",
            python_callable=promote_model,
        )
        t_archive = PythonOperator(
            task_id="archive_model",
            python_callable=archive_model,
        )
        t_check >> t_repro >> t_read >> [t_promote, t_archive]
