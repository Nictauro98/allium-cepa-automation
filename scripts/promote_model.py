#!/usr/bin/env python
"""
Promote the validated champion classifier to production in the MLflow Model Registry.

Guards:
  - validation_result.json must exist and report approved=true
  - The champion experiment must have mlflow_run_id.txt (trained with MLflow enabled)

After promotion:
  - A new version of "allium-classifier" is registered in the MLflow Model Registry
  - The "@production" alias points to this version
  - metrics/evaluation_report.json is uploaded to MinIO as models/production/metrics.json,
    seeding the baseline for the next validate_model run

Usage:
  uv run python scripts/promote_model.py
  uv run python scripts/promote_model.py --dry-run

To roll back a promotion:
  mlflow aliases set --model-name allium-classifier --alias production --version <prev-version>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env", override=False)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_MODEL_NAME = "allium-classifier"
_PRODUCTION_ALIAS = "production"
_PRODUCTION_KEY = "models/production/metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote validated champion to MLflow production.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without making any changes.",
    )
    args = parser.parse_args()

    # 1. Load champion from params.yaml
    params = yaml.safe_load((_ROOT / "params.yaml").read_text())
    champion = params["champion_classifier"]
    exp_dir = _ROOT / "experiments" / "binary_classifier" / champion
    log.info("Champion: %s", champion)

    # 2. Check validation result
    result_path = _ROOT / "validation_result.json"
    if not result_path.exists():
        log.error("validation_result.json not found. Run: dvc repro evaluate validate_model")
        sys.exit(1)
    result = json.loads(result_path.read_text())
    if not result["approved"]:
        log.error("Model is not approved. Promotion blocked. Reasons:")
        for r in result["reasons"]:
            log.error("  • %s", r)
        sys.exit(1)
    log.info("Validation: APPROVED")

    # 3. Get MLflow run_id saved by train_classifier.py
    run_id_file = exp_dir / "mlflow_run_id.txt"
    if not run_id_file.exists():
        log.error(
            "No mlflow_run_id.txt found in %s. "
            "Re-train with MLFLOW_TRACKING_URI set so the run is tracked.",
            exp_dir,
        )
        sys.exit(1)
    run_id = run_id_file.read_text().strip()
    log.info("MLflow run_id: %s", run_id)

    model_uri = f"runs:/{run_id}/calibrated_classifier"
    report_path = _ROOT / "metrics" / "evaluation_report.json"

    if args.dry_run:
        log.info("[dry-run] Would register %s as %s", model_uri, _MODEL_NAME)
        log.info("[dry-run] Would set alias '%s' on the new version", _PRODUCTION_ALIAS)
        log.info("[dry-run] Would upload %s → MinIO %s", report_path, _PRODUCTION_KEY)
        return

    # 4. Register in MLflow Model Registry
    import mlflow

    client = mlflow.tracking.MlflowClient()
    try:
        client.create_registered_model(_MODEL_NAME)
        log.info("Created registered model: %s", _MODEL_NAME)
    except Exception:
        log.info("Registered model already exists: %s", _MODEL_NAME)

    # Ensure the run has a proper MLmodel artifact (runs trained before this fix only
    # logged a plain .pt file via log_artifact, which register_model cannot use).
    artifacts = [a.path for a in client.list_artifacts(run_id, "calibrated_classifier")]
    if "calibrated_classifier/MLmodel" not in artifacts:
        log.info("No MLmodel found in run — re-logging weights as pyfunc model.")
        weights_path = exp_dir / "weights" / "classifier_calibrated.pt"

        class _CalibratedCheckpointModel(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input):  # noqa: ARG002
                raise NotImplementedError("Load via AlliumCepaModel, not MLflow pyfunc.")

        with mlflow.start_run(run_id=run_id):
            mlflow.pyfunc.log_model(
                artifact_path="calibrated_classifier",
                python_model=_CalibratedCheckpointModel(),
                artifacts={"weights": str(weights_path)},
            )
        log.info("MLmodel artifact logged into run %s.", run_id)

    version = mlflow.register_model(model_uri, _MODEL_NAME)
    log.info("Registered version: %s", version.version)

    client.set_registered_model_alias(_MODEL_NAME, _PRODUCTION_ALIAS, version.version)
    log.info("Set alias '%s' → v%s", _PRODUCTION_ALIAS, version.version)

    # 5. Upload evaluation metrics to MinIO as the new production baseline
    from allium_cepa_classifier.providers.factory import get_storage

    storage = get_storage()
    storage.put_file(report_path, _PRODUCTION_KEY)
    log.info("Uploaded new production baseline → %s", _PRODUCTION_KEY)

    log.info(
        "\nDone. %s v%s is now @%s in the MLflow Model Registry.\n"
        "Load in inference: AlliumCepaModel(AlliumCepaConfig(use_registry=True))\n"
        "Roll back:         mlflow aliases set --model-name %s --alias %s --version <prev>",
        _MODEL_NAME,
        version.version,
        _PRODUCTION_ALIAS,
        _MODEL_NAME,
        _PRODUCTION_ALIAS,
    )


if __name__ == "__main__":
    main()
