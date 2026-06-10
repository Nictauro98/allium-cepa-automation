"""
Usage:
    uv run python scripts/train_classifier.py --config experiments/binary_classifier/efficientnet_b1/config.yaml
    uv run python scripts/train_classifier.py --config experiments/binary_classifier/efficientnet_b1/config.yaml --dry-run
    uv run python scripts/train_classifier.py --config experiments/binary_classifier/efficientnet_b1/config.yaml --no-calibrate
"""

import argparse
import logging
import re
from pathlib import Path

import yaml

from allium_cepa_classifier.config.experiment_config import ExperimentConfig
from allium_cepa_classifier.training.mlflow_logging import run as mlflow_run
from allium_cepa_classifier.training.trainer import run_training

_ROOT = Path(__file__).resolve().parents[1]


def _dataset_rev() -> str | None:
    """Extract the HuggingFace dataset SHA pinned in dvc.yaml, or None if not found."""
    try:
        dvc = yaml.safe_load((_ROOT / "dvc.yaml").read_text())
        cmd = dvc["stages"]["download_dataset"]["cmd"]
        m = re.search(r"--rev\s+([0-9a-f]{40})", cmd.replace("\n", " "))
        return m.group(1) if m else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and build model only, do not train",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip temperature calibration after training",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    run_dir = Path(args.config).parent
    (run_dir / "weights").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "train.log"),
        ],
    )

    if args.dry_run:
        from allium_cepa_classifier.training.model_builder import build_model

        model = build_model(cfg.model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Dry run OK. Run dir: {run_dir}")
        print(f"Trainable params: {trainable:,} / {total:,}")
        return

    with mlflow_run(run_name=f"classifier_{cfg.model.arch}") as mlctx:
        params = {
            "arch": cfg.model.arch,
            "lr": cfg.training.lr,
            "batch_size": cfg.data.batch_size,
            "epochs": cfg.training.epochs,
            "freeze_stages": cfg.model.freeze_stages,
        }
        if (rev := _dataset_rev()) is not None:
            params["dataset_rev"] = rev
        mlctx.log_params(params)
        mlctx.log_artifact(args.config)

        metrics = run_training(cfg, run_dir)
        mlctx.log_metrics({
            "train_acc": metrics["train_acc"],
            "val_acc": metrics["val_acc"],
            "test_acc": metrics["test_acc"],
            "best_val_loss": metrics["best_val_loss"],
        })
        mlctx.log_artifact(run_dir / "weights" / "classifier.pt")

        if not args.no_calibrate:
            from allium_cepa_classifier.training.calibrator import run_calibration

            logging.info("\n--- Calibration ---")
            cal_metrics = run_calibration(run_dir)
            mlctx.log_metrics({
                "ece_before": cal_metrics["ece_before"],
                "ece_after": cal_metrics["ece_after"],
            })
            # Log under a stable subdir so it can be registered in the MLflow Model Registry
            mlctx.log_model_file(
                run_dir / "weights" / "classifier_calibrated.pt",
                "calibrated_classifier",
            )
            print(f"ECE before: {cal_metrics['ece_before']:.4f}  after: {cal_metrics['ece_after']:.4f}")
            print(f"Temperature: {cal_metrics['temperature']}")

    # Persist the run_id so promote_model.py can register this version later
    if mlctx.run_id is not None:
        (run_dir / "mlflow_run_id.txt").write_text(mlctx.run_id)

    print(f"\nDone. Artifacts in: {run_dir}")
    print(f"Test accuracy: {metrics['test_acc']:.4f}")


if __name__ == "__main__":
    main()
