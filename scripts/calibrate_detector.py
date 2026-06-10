"""
Usage:
    uv run python scripts/calibrate_detector.py --experiment experiments/yolo/yolo11n_200e/20260503-180000
"""

import argparse
import logging
from pathlib import Path

from allium_cepa_classifier.training.detector_calibrator import run_detection_calibration
from allium_cepa_classifier.training.mlflow_logging import run as mlflow_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        required=True,
        type=Path,
        help="Path to detector experiment directory, e.g. experiments/yolo/yolo11n_200e",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.experiment / "calibrate.log"),
        ],
    )

    with mlflow_run(run_name="calibrate_detector") as mlctx:
        metrics = run_detection_calibration(args.experiment)
        mlctx.log_metrics({"ece_before": metrics["ece_before"], "ece_after": metrics["ece_after"]})
        mlctx.log_artifact(args.experiment / "weights" / "yolo_isotonic_calibrator.pkl")

    print(f"ECE before: {metrics['ece_before']:.4f}  after: {metrics['ece_after']:.4f}")


if __name__ == "__main__":
    main()
