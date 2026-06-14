"""DAG A: zooniverse_ingest

Pull expert phase classifications from Zooniverse → filter by consensus →
patch HuggingFace annotations → bump dvc.yaml SHA → trigger retrain.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections import Counter
from pathlib import Path

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator, ShortCircuitOperator
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator
    from airflow.utils.dates import days_ago

    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def download_zooniverse_classifications(**context) -> None:
    """Fetch classifications from Zooniverse since the last successful run."""
    from allium_cepa_classifier.providers.factory import get_labeling

    last_ts = context["dag_run"].conf.get("since") if context.get("dag_run") else None
    records = get_labeling().fetch_classifications(since=last_ts)
    log.info("Fetched %d classifications", len(records))
    context["ti"].xcom_push(key="records", value=records)


def filter_by_consensus(**context) -> bool:
    """Keep subjects where expert agreement >= consensus_threshold.

    Returns False (short-circuit) if no subjects pass the filter.
    """
    from allium_cepa_classifier.config import ZooniverseConfig

    cfg = ZooniverseConfig()
    records: list[dict] = context["ti"].xcom_pull(
        task_ids="download_zooniverse_classifications", key="records"
    )

    # Group by subject_id and pick majority vote
    by_subject: dict[str, list[str]] = {}
    for r in records:
        sid = r["subject_id"]
        by_subject.setdefault(sid, []).append(r["value"])

    accepted = []
    for sid, votes in by_subject.items():
        total = len(votes)
        majority_phase, majority_count = Counter(votes).most_common(1)[0]
        agreement = majority_count / total
        if agreement >= cfg.consensus_threshold:
            # Carry annotation_id + split through from the first matching record
            ref = next(r for r in records if r["subject_id"] == sid)
            accepted.append(
                {
                    "annotation_id": ref["annotation_id"],
                    "split": ref.get("split", "train"),
                    "phase": majority_phase,
                    "agreement": agreement,
                }
            )
        else:
            log.debug("Subject %s discarded (agreement=%.2f)", sid, agreement)

    log.info("Accepted %d / %d subjects", len(accepted), len(by_subject))
    context["ti"].xcom_push(key="accepted", value=accepted)
    return len(accepted) > 0


def apply_phase_to_annotations(**context) -> None:
    """Merge Zooniverse verdicts into locally-fetched annotations.json per split."""
    from allium_cepa_classifier.config import ZooniverseConfig

    cfg = ZooniverseConfig()
    accepted: list[dict] = context["ti"].xcom_pull(
        task_ids="filter_by_consensus", key="accepted"
    )

    # Load annotations per split from the mock/real hub local cache
    hub_dir = Path("/tmp/mock_hub")

    patches_by_split: dict[str, list[dict]] = {}
    for verdict in accepted:
        split = verdict["split"]
        ann_id = verdict["annotation_id"]
        phase = verdict["phase"]

        patch: dict = {"id": ann_id, "attributes": {"phase": phase}}

        division = cfg.phase_division_map.get(phase)
        if division is not None:
            patch["attributes"]["division"] = division
        if phase == "not_a_cell":
            patch["attributes"]["detection_error"] = True

        patches_by_split.setdefault(split, []).append(patch)

    # Persist patches locally so the next task can push them
    out_path = hub_dir / "pending_patches.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(patches_by_split))
    context["ti"].xcom_push(key="patches_by_split", value=patches_by_split)
    log.info("Prepared patches for splits: %s", list(patches_by_split))


def _enough_patches(**context) -> bool:
    """ShortCircuitOperator: skip HF push if resolved count < min_new_for_patch."""
    from allium_cepa_classifier.config import ZooniverseConfig

    cfg = ZooniverseConfig()
    patches_by_split: dict[str, list] = context["ti"].xcom_pull(
        task_ids="apply_phase_to_annotations", key="patches_by_split"
    )
    total = sum(len(v) for v in patches_by_split.values())
    log.info("Resolved annotations: %d (min required: %d)", total, cfg.min_new_for_patch)
    return total >= cfg.min_new_for_patch


def patch_huggingface_annotations(**context) -> None:
    """Push annotation patches to HuggingFace (one call per split)."""
    from allium_cepa_classifier.providers.factory import get_dataset_hub

    hub = get_dataset_hub()
    patches_by_split: dict[str, list[dict]] = context["ti"].xcom_pull(
        task_ids="apply_phase_to_annotations", key="patches_by_split"
    )
    new_sha = None
    for split, patches in patches_by_split.items():
        new_sha = hub.patch_annotations(split, patches)
        log.info("Patched %d annotations in split=%s → SHA=%s", len(patches), split, new_sha)
    context["ti"].xcom_push(key="new_sha", value=new_sha)


def update_dvc_yaml_sha(**context) -> None:
    """Rewrite --rev <sha> in dvc.yaml and git-commit. No-op in mock mode."""
    import os

    if os.getenv("ALLIUM_HUB", "mock") == "mock":
        log.info("Mock mode: skipping dvc.yaml SHA update")
        return

    new_sha: str | None = context["ti"].xcom_pull(
        task_ids="patch_huggingface_annotations", key="new_sha"
    )
    if not new_sha:
        log.warning("No new SHA received; skipping dvc.yaml update")
        return

    dvc_yaml = Path("dvc.yaml")
    text = dvc_yaml.read_text()
    import re

    text = re.sub(r"(--rev\s+)\S+", rf"\g<1>{new_sha}", text)
    dvc_yaml.write_text(text)

    subprocess.run(["git", "add", "dvc.yaml"], check=True)
    subprocess.run(
        ["git", "commit", "-m", f"chore: bump HF dataset SHA to {new_sha[:8]}"],
        check=True,
    )
    subprocess.run(["git", "push"], check=True)
    log.info("dvc.yaml bumped to %s", new_sha)


# ---------------------------------------------------------------------------
# DAG definition (skipped when Airflow is not installed, e.g. in unit tests)
# ---------------------------------------------------------------------------

if _AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="zooniverse_ingest",
        description="Ingest Zooniverse expert classifications and update HF dataset",
        schedule="0 3 * * *",
        start_date=days_ago(1),
        catchup=False,
        tags=["ingestion", "zooniverse"],
    ) as dag:
        t_download = PythonOperator(
            task_id="download_zooniverse_classifications",
            python_callable=download_zooniverse_classifications,
            retries=3,
        )
        t_filter = PythonOperator(
            task_id="filter_by_consensus",
            python_callable=filter_by_consensus,
        )
        t_apply = PythonOperator(
            task_id="apply_phase_to_annotations",
            python_callable=apply_phase_to_annotations,
        )
        t_enough = ShortCircuitOperator(
            task_id="check_enough_patches",
            python_callable=_enough_patches,
        )
        t_patch_hf = PythonOperator(
            task_id="patch_huggingface_annotations",
            python_callable=patch_huggingface_annotations,
            retries=3,
        )
        t_bump_sha = PythonOperator(
            task_id="update_dvc_yaml_sha",
            python_callable=update_dvc_yaml_sha,
            retries=3,
        )
        t_notify = TriggerDagRunOperator(
            task_id="notify_retrain",
            trigger_dag_id="retrain_pipeline",
            wait_for_completion=False,
        )
        t_download >> t_filter >> t_apply >> t_enough >> t_patch_hf >> t_bump_sha >> t_notify
