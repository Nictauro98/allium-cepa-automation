"""
Model validation gate.

Compares a new evaluation_report.json against the production baseline.
Writes validation_result.json and exits 0 (approved) or 1 (rejected).

Production baseline is read via the storage provider:
  key = "models/production/metrics.json"
If absent (first run), the new model is auto-approved and a warning is emitted.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from allium_cepa_classifier.config.validation_config import ValidationConfig
from allium_cepa_classifier.providers.factory import get_storage

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_PRODUCTION_KEY = "models/production/metrics.json"


@dataclass
class ValidationResult:
    approved: bool
    reasons: list[str]
    new_metrics: dict
    current_metrics: dict | None


def _check(
    new: dict,
    prod: dict,
    cfg: ValidationConfig,
) -> list[str]:
    """Return a list of rejection reasons (empty → approved)."""
    reasons: list[str] = []

    new_f1 = new["macro_f1"]
    prod_f1 = prod["macro_f1"]
    if new_f1 - prod_f1 < cfg.min_f1_delta:
        reasons.append(
            f"Macro F1 improvement {new_f1 - prod_f1:.4f} < required delta {cfg.min_f1_delta}"
        )

    new_per = new.get("f1_per_class", {})
    prod_per = prod.get("f1_per_class", {})
    for cls, prod_val in prod_per.items():
        new_val = new_per.get(cls, 0.0)
        drop = prod_val - new_val
        if drop > cfg.per_class_tolerance:
            reasons.append(f"F1[{cls}] dropped {drop:.4f} > tolerance {cfg.per_class_tolerance}")

    new_ece = new.get("ece", 0.0)
    prod_ece = prod.get("ece", 0.0)
    ece_increase = new_ece - prod_ece
    if ece_increase > cfg.ece_tolerance:
        reasons.append(f"ECE increased {ece_increase:.4f} > tolerance {cfg.ece_tolerance}")

    return reasons


def run_validation(report_path: Path, cfg: ValidationConfig | None = None) -> ValidationResult:
    if cfg is None:
        cfg = ValidationConfig()

    new_metrics = json.loads(report_path.read_text())

    storage = get_storage()
    if not storage.exists(_PRODUCTION_KEY):
        log.warning(
            "No production baseline found at '%s'. "
            "Auto-approving as first model — upload metrics to seed the baseline.",
            _PRODUCTION_KEY,
        )
        return ValidationResult(
            approved=True,
            reasons=["No production baseline — auto-approved as first model."],
            new_metrics=new_metrics,
            current_metrics=None,
        )

    prod_metrics = json.loads(storage.read_text(_PRODUCTION_KEY))
    log.info(f"Production baseline: macro_f1={prod_metrics['macro_f1']:.4f}")
    log.info(f"New model:           macro_f1={new_metrics['macro_f1']:.4f}")

    reasons = _check(new_metrics, prod_metrics, cfg)
    approved = len(reasons) == 0

    if approved:
        log.info("APPROVED — all validation criteria passed.")
    else:
        log.warning("REJECTED — %d criterion/criteria failed:", len(reasons))
        for r in reasons:
            log.warning("  • %s", r)

    return ValidationResult(
        approved=approved,
        reasons=reasons,
        new_metrics=new_metrics,
        current_metrics=prod_metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Model validation gate.")
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path to evaluation_report.json produced by evaluate.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write validation_result.json",
    )
    args = parser.parse_args()

    result = run_validation(args.report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(result), indent=2))
    log.info(f"Written: {args.output}")

    sys.exit(0 if result.approved else 1)


if __name__ == "__main__":
    main()
