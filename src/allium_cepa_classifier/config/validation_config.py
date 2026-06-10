from __future__ import annotations

from pydantic import ConfigDict

from .base_config import BaseConfig


class ValidationConfig(BaseConfig):
    """Thresholds for the model validation gate."""

    model_config = ConfigDict(frozen=True)

    min_f1_delta: float = 0.01
    per_class_tolerance: float = 0.03
    ece_tolerance: float = 0.02
    metric_key: str = "macro_f1"
