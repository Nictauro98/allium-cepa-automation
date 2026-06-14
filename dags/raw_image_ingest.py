"""DAG B: raw_image_ingest

Pull raw images from Drive → run production AlliumCepaModel → build COCO
annotations (division set for high-confidence, None for low) → publish to HF
→ send pending crops to Zooniverse → optionally trigger retrain.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import uuid
from pathlib import Path

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.utils.dates import days_ago

    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False

log = logging.getLogger(__name__)

# Number of auto-labeled (non-pending) annotations required to trigger retraining.
_AUTO_LABEL_RETRAIN_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def download_from_drive(**context) -> None:
    """List and download new images from Drive into a temp dir."""
    from allium_cepa_classifier.providers.factory import get_drive

    drive = get_drive()
    since = context["dag_run"].conf.get("since") if context.get("dag_run") else None
    file_ids = drive.list_new(since=since)
    log.info("Found %d new images on Drive", len(file_ids))

    tmp_dir = Path(tempfile.mkdtemp(prefix="raw_ingest_"))
    local_paths: list[str] = []
    for fid in file_ids:
        local = drive.download(fid, tmp_dir / fid)
        local_paths.append(str(local))
        log.debug("Downloaded %s → %s", fid, local)

    context["ti"].xcom_push(key="local_paths", value=local_paths)
    context["ti"].xcom_push(key="tmp_dir", value=str(tmp_dir))


def run_instance_detection(**context) -> None:
    """Ensure production weights are present, run YOLO detection on each image."""
    from allium_cepa_classifier.config import ProductionConfig
    from allium_cepa_classifier.serving.weights import ensure_production_weights

    cfg = ProductionConfig()
    ensure_production_weights(cfg)

    local_paths: list[str] = context["ti"].xcom_pull(
        task_ids="download_from_drive", key="local_paths"
    )

    from allium_cepa_classifier import AlliumCepaConfig, AlliumCepaModel

    model = AlliumCepaModel(AlliumCepaConfig())

    # Store per-image raw detections for classify_and_score
    all_detections: list[dict] = []
    for img_path in local_paths:
        result = model.predict(img_path)
        all_detections.append(
            {
                "image_path": img_path,
                "result": result._raw if hasattr(result, "_raw") else {},
            }
        )

    context["ti"].xcom_push(key="all_detections", value=all_detections)
    context["ti"].xcom_push(key="model_cfg", value=cfg.model_dump())


def classify_and_score(**context) -> None:
    """Classify crops; produce (bbox, class, confidence, ci_lower, ci_upper) per crop."""
    from allium_cepa_classifier import AlliumCepaModel
    from allium_cepa_classifier.config import AlliumCepaConfig, ProductionConfig

    cfg = ProductionConfig()
    local_paths: list[str] = context["ti"].xcom_pull(
        task_ids="download_from_drive", key="local_paths"
    )

    model = AlliumCepaModel(AlliumCepaConfig())

    scored: list[dict] = []
    for img_path in local_paths:
        result = model.predict(img_path)
        counts = result.get_counts_with_ci()
        scored.append(
            {
                "image_path": img_path,
                "counts": counts,
                "detections": _extract_detections(result),
            }
        )

    context["ti"].xcom_push(key="scored", value=scored)
    context["ti"].xcom_push(key="high_confidence_threshold", value=cfg.high_confidence_threshold)


def _extract_detections(result) -> list[dict]:
    """Extract per-bbox data from an AlliumCepaResult."""
    detections = []
    if hasattr(result, "boxes") and result.boxes is not None:
        for i, box in enumerate(result.boxes):
            conf = float(getattr(box, "conf", 0.0))
            cls = int(getattr(box, "cls", 0))
            xyxy = box.xyxy[0].tolist() if hasattr(box, "xyxy") else [0, 0, 0, 0]
            detections.append(
                {
                    "bbox": xyxy,
                    "class": cls,
                    "confidence": conf,
                    "ci_lower": max(0.0, conf - 0.05),
                    "ci_upper": min(1.0, conf + 0.05),
                    "idx": i,
                }
            )
    return detections


def build_coco_annotations(**context) -> None:
    """Build COCO annotation dicts; set division for high-confidence, None for pending."""
    scored: list[dict] = context["ti"].xcom_pull(task_ids="classify_and_score", key="scored")
    threshold: float = context["ti"].xcom_pull(
        task_ids="classify_and_score", key="high_confidence_threshold"
    )

    all_annotations: list[dict] = []
    pending_ids: list[str] = []
    auto_count = 0

    for item in scored:
        for det in item.get("detections", []):
            ann_id = str(uuid.uuid4())
            bbox = det["bbox"]
            confidence = det["confidence"]
            is_high_conf = confidence >= threshold

            ann: dict = {
                "id": ann_id,
                "image_path": item["image_path"],
                "bbox": bbox,
                "attributes": {
                    "source": "auto",
                    "confidence": confidence,
                    "ci_lower": det.get("ci_lower"),
                    "ci_upper": det.get("ci_upper"),
                },
            }

            if is_high_conf:
                ann["attributes"]["division"] = det["class"]
                auto_count += 1
            else:
                ann["attributes"]["division"] = None
                ann["attributes"]["pending"] = True
                pending_ids.append(ann_id)

            all_annotations.append(ann)

    log.info(
        "Built %d annotations: %d auto-labeled, %d pending",
        len(all_annotations),
        auto_count,
        len(pending_ids),
    )
    context["ti"].xcom_push(key="all_annotations", value=all_annotations)
    context["ti"].xcom_push(key="pending_ids", value=pending_ids)
    context["ti"].xcom_push(key="auto_count", value=auto_count)


def publish_to_huggingface(**context) -> None:
    """Publish full images + all annotations (pending + resolved) to HF split=train."""
    from allium_cepa_classifier.providers.factory import get_dataset_hub

    local_paths: list[str] = context["ti"].xcom_pull(
        task_ids="download_from_drive", key="local_paths"
    )
    all_annotations: list[dict] = context["ti"].xcom_pull(
        task_ids="build_coco_annotations", key="all_annotations"
    )

    hub = get_dataset_hub()
    images = [Path(p) for p in local_paths]
    new_sha = hub.publish_images(split="train", images=images, annotations=all_annotations)
    log.info("Published %d images → HF SHA=%s", len(images), new_sha)
    context["ti"].xcom_push(key="new_sha", value=new_sha)


def update_dvc_yaml_sha(**context) -> None:
    """Rewrite --rev <sha> in dvc.yaml and git-commit. No-op in mock mode."""
    import os

    if os.getenv("ALLIUM_HUB", "mock") == "mock":
        log.info("Mock mode: skipping dvc.yaml SHA update")
        return

    new_sha: str | None = context["ti"].xcom_pull(
        task_ids="publish_to_huggingface", key="new_sha"
    )
    if not new_sha:
        log.warning("No new SHA received; skipping dvc.yaml update")
        return

    dvc_yaml = Path("dvc.yaml")
    text = dvc_yaml.read_text()
    text = re.sub(r"(--rev\s+)\S+", rf"\g<1>{new_sha}", text)
    dvc_yaml.write_text(text)

    subprocess.run(["git", "add", "dvc.yaml"], check=True)
    subprocess.run(
        ["git", "commit", "-m", f"chore: bump HF dataset SHA to {new_sha[:8]}"],
        check=True,
    )
    subprocess.run(["git", "push"], check=True)
    log.info("dvc.yaml bumped to %s", new_sha)


def send_pending_to_zooniverse(**context) -> None:
    """Send low-confidence crop IDs to Zooniverse for expert review."""
    from allium_cepa_classifier.providers.factory import get_labeling

    pending_ids: list[str] = context["ti"].xcom_pull(
        task_ids="build_coco_annotations", key="pending_ids"
    )
    if not pending_ids:
        log.info("No pending annotations to send")
        return

    log.info("Sending %d pending annotations to Zooniverse", len(pending_ids))
    get_labeling().create_tasks(pending_ids, priority="normal")


def notify_retrain_if_threshold(**context) -> None:
    """Trigger retrain_pipeline if enough auto-labeled annotations were added."""
    auto_count: int = context["ti"].xcom_pull(
        task_ids="build_coco_annotations", key="auto_count"
    )
    if auto_count >= _AUTO_LABEL_RETRAIN_THRESHOLD:
        log.info("Auto-label count %d >= threshold %d; triggering retrain", auto_count, _AUTO_LABEL_RETRAIN_THRESHOLD)
        try:
            from airflow.api.common.trigger_dag import trigger_dag
            trigger_dag(dag_id="retrain_pipeline")
        except ImportError:
            log.warning("Airflow not available; skipping programmatic trigger of retrain_pipeline")
    else:
        log.info(
            "Auto-label count %d < threshold %d; skipping retrain",
            auto_count,
            _AUTO_LABEL_RETRAIN_THRESHOLD,
        )


# ---------------------------------------------------------------------------
# DAG definition (skipped when Airflow is not installed, e.g. in unit tests)
# ---------------------------------------------------------------------------

if _AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="raw_image_ingest",
        description="Ingest raw Drive images, run inference, publish to HF, send pending to Zooniverse",
        schedule=None,
        start_date=days_ago(1),
        catchup=False,
        tags=["ingestion", "drive"],
    ) as dag:
        t_download = PythonOperator(
            task_id="download_from_drive",
            python_callable=download_from_drive,
            retries=3,
        )
        t_detect = PythonOperator(
            task_id="run_instance_detection",
            python_callable=run_instance_detection,
        )
        t_classify = PythonOperator(
            task_id="classify_and_score",
            python_callable=classify_and_score,
        )
        t_build = PythonOperator(
            task_id="build_coco_annotations",
            python_callable=build_coco_annotations,
        )
        t_publish = PythonOperator(
            task_id="publish_to_huggingface",
            python_callable=publish_to_huggingface,
            retries=3,
        )
        t_bump_sha = PythonOperator(
            task_id="update_dvc_yaml_sha",
            python_callable=update_dvc_yaml_sha,
            retries=3,
        )
        t_send_pending = PythonOperator(
            task_id="send_pending_to_zooniverse",
            python_callable=send_pending_to_zooniverse,
            retries=3,
        )
        t_notify = PythonOperator(
            task_id="notify_retrain_if_threshold",
            python_callable=notify_retrain_if_threshold,
        )
        (
            t_download
            >> t_detect
            >> t_classify
            >> t_build
            >> t_publish
            >> t_bump_sha
            >> t_send_pending
            >> t_notify
        )
