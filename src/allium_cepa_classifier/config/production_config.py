from __future__ import annotations

from pathlib import Path

from pydantic import ConfigDict

from .base_config import BaseConfig


class ProductionConfig(BaseConfig):
    """Serving config: where prod weights are cached locally + the gate-relevant thresholds."""

    model_config = ConfigDict(frozen=True)

    weights_cache_dir: Path = Path("/tmp/allium_weights")
    # bucket-relative keys under the storage provider (Plan A)
    detection_key: str = "models/production/object_detection.pt"
    classifier_key: str = "models/production/classifier_calibrated.pt"
    calibrator_key: str = "models/production/yolo_isotonic_calibrator.pkl"
    metrics_key: str = "models/production/metrics.json"

    # active-learning routing threshold (consumed by Plan D; defined here for symmetry)
    high_confidence_threshold: float = 0.90
