#!/usr/bin/env python
"""
Sync production weight files between the local weights directory and MinIO.

  --push   Upload local weights → MinIO models/production/   (first-time setup)
  --check  Verify files exist in the bucket without downloading
  (default) Download MinIO models/production/ → local weights dir

Usage:
  uv run python scripts/pull_production_weights.py --push          # local → MinIO
  uv run python scripts/pull_production_weights.py                 # MinIO → local
  uv run python scripts/pull_production_weights.py --check         # verify bucket

  # Override source/dest dir (default: src/allium_cepa_classifier/weights/)
  uv run python scripts/pull_production_weights.py --push --src /path/to/weights
  uv run python scripts/pull_production_weights.py --dest /tmp/weights

Environment:
  ALLIUM_STORAGE=minio (default) or s3
  MINIO_ENDPOINT, MINIO_ROOT_USER, MINIO_ROOT_PASSWORD
  ALLIUM_BUCKET (default: allium-cepa-ml)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env", override=False)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_DEFAULT_WEIGHTS_DIR = _ROOT / "src/allium_cepa_classifier/weights"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync production weights between local disk and MinIO."
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload local weights to MinIO (first-time setup or after re-training).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify that the files exist in the bucket; do not transfer anything.",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=_DEFAULT_WEIGHTS_DIR,
        help=f"Local weights directory used as source for --push (default: {_DEFAULT_WEIGHTS_DIR})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=_DEFAULT_WEIGHTS_DIR,
        help=f"Local directory to write files into when pulling (default: {_DEFAULT_WEIGHTS_DIR})",
    )
    args = parser.parse_args()

    from allium_cepa_classifier.config import ProductionConfig
    from allium_cepa_classifier.providers.factory import get_storage

    cfg = ProductionConfig()
    storage = get_storage()

    # (bucket_key, local_filename)
    files = [
        (cfg.detection_key, "object_detection.pt"),
        (cfg.classifier_key, "classifier_calibrated.pt"),
        (cfg.calibrator_key, "yolo_isotonic_calibrator.pkl"),
    ]

    if args.check:
        all_present = True
        for key, _ in files:
            exists = storage.exists(key)
            status = "OK" if exists else "MISSING"
            log.info("[%s] %s", status, key)
            if not exists:
                all_present = False
        if not all_present:
            log.error("One or more production weight files are missing from the bucket.")
            sys.exit(1)
        log.info("All production weight files are present in the bucket.")
        return

    if args.push:
        for key, filename in files:
            local = args.src / filename
            if not local.exists():
                log.error("Local file not found: %s", local)
                sys.exit(1)
            log.info("Uploading %s → %s", local.name, key)
            storage.put_file(local, key)
            log.info("  done (%d bytes)", local.stat().st_size)
        log.info("\nWeights uploaded to bucket. Run --check to verify.")
        return

    # Pull
    args.dest.mkdir(parents=True, exist_ok=True)
    for key, filename in files:
        dest = args.dest / filename
        if dest.exists():
            log.info("[skip] %s already exists", filename)
            continue
        log.info("Downloading %s → %s", key, dest)
        storage.get_file(key, dest)
        log.info("  done (%d bytes)", dest.stat().st_size)
    log.info("\nWeights written to: %s", args.dest)


if __name__ == "__main__":
    main()
