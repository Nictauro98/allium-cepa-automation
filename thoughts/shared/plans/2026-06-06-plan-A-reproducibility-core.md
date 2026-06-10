# Plan A — Reproducibility & Tracking Core — Implementation Plan

> Milestone 1 of 4. See [`2026-06-06-portfolio-roadmap.md`](2026-06-06-portfolio-roadmap.md) for the full program and the "providers" mock→real convention this plan establishes.

## Overview

Make the existing two-stage pipeline fully **reproducible** and **observable**, and add the **model validation gate**. After this plan: `dvc repro` runs the complete chain through `evaluate` → `validate_model`; every training run is logged to MLflow; a new model is automatically accepted or rejected against the current production baseline. All infrastructure (MinIO + MLflow) runs locally via Docker — zero external accounts.

This plan also establishes two foundations the later plans reuse:

1. The `providers/` storage abstraction (local/MinIO ↔ real S3 via one env var).
2. The `tests/` directory (currently none exists).

## Current State Analysis

Verified against the repo on branch `main` (2026-06-06):

- `dvc.yaml` already has `download_dataset`, `coco_to_yolo`, `prepare_crops`, `train_classifier` (foreach 4 archs), `calibrate_classifier`, `train_detector`, `calibrate_detector`. The architecture doc calling these "new" is stale. **Genuinely missing: `evaluate` and `validate_model`.**
- **Orphaned dvclive block** at [`dvc.yaml:107-124`](../../../dvc.yaml#L107) — top-level `params:`/`metrics:`/`plots:`/`artifacts:` not wired to any stage. Leftover; remove.
- **Package is import-broken:** [`config/__init__.py`](../../../src/allium_cepa_classifier/config/__init__.py#L6) still imports `controlnet_config` and `vae_config`, both already deleted from the tree. `import allium_cepa_classifier` raises `ModuleNotFoundError` right now.
- **No `tests/` directory exists.** `pyproject.toml` already configures pytest: `testpaths=["tests"]`, `--cov=src`.
- **Storage is HuggingFace, not S3/MinIO:** `.dvc/config` uses HF remotes. The architecture doc's entire storage model (S3 buckets, `production/metrics.json`) assumes object storage. No MinIO/S3 remote exists yet.
- **Metrics format today** (classifier `metrics.json`): `train_acc`/`val_acc`/`test_acc`, `best_val_loss`, `epochs_run`, `history`. No Macro F1, per-class F1, or ECE. Calibration ECE lives separately in `calibration_metrics.json`. The `evaluate` stage must produce a **new consolidated** `evaluation_report.json`.
- **Inference entrypoint** is `AlliumCepaModel.predict()` ([`data_models/allium_cepa_model.py:323`](../../../src/allium_cepa_classifier/data_models/allium_cepa_model.py#L323)), returning `AlliumCepaResult`. Class ordering: classifier index `0 = mitosis`, `1 = no_mitosis` ([`allium_cepa_model.py:217-222`](../../../src/allium_cepa_classifier/data_models/allium_cepa_model.py#L217)).
- **Config pattern:** all configs extend `BaseConfig(BaseModel)` with `from_yaml()` ([`config/base_config.py`](../../../src/allium_cepa_classifier/config/base_config.py)); `DetectorConfig` uses `ConfigDict(frozen=True)` + `Path` defaults.
- **Trainer** writes `metrics.json` and uses sklearn `classification_report`/`confusion_matrix` ([`training/trainer.py`](../../../src/allium_cepa_classifier/training/trainer.py)).
- **MLflow is NOT a dependency yet.** `dvc`, `dvclive`, `sklearn`, `scipy` are present. **Diffusers/accelerate/transformers/datasets/safetensors remain in `pyproject.toml` but are now dead weight** after VAE/ControlNet removal.

### Key Discoveries

- The legacy assumption that the train/calibrate stages are "new" is wrong — only `evaluate` + `validate_model` are missing.
- The package does not import cleanly on `main`; **Phase 0 must run first** or nothing else can be tested.
- The fixed test set the gate needs already exists locally as `datasets/crops/binary_classifier/test` (produced by `prepare_crops`). The doc's `s3://dataset/test_fixed/` is a later swap-in via the storage provider.

## Desired End State

- `uv run python -c "import allium_cepa_classifier.config"` succeeds (no VAE/ControlNet imports).
- `dvc repro evaluate validate_model` runs end-to-end producing `metrics/evaluation_report.json` and `validation_result.json`.
- `docker-compose up minio mlflow` brings up MinIO (`:9000`/`:9001`) and MLflow (`:5000`); a training run appears in the MLflow UI.
- Flipping `ALLIUM_STORAGE=s3` (plus AWS env vars) switches the storage provider from local/MinIO to real S3 **with no code change**.
- `uv run pytest` passes with real tests covering the gate and the storage provider.

## What We're NOT Doing (this plan)

- FastAPI / Streamlit / Docker images for serving (**Plan B**).
- CI workflow / README (**Plan C**).
- Zooniverse / Drive / HF providers and Airflow DAGs (**Plan D**) — only the storage provider here.
- Real AWS S3 usage (only the swap path; default stays MinIO/local).
- Changing the training/calibration algorithms.

## Implementation Approach

Bottom-up: unbreak the package and clean cruft first, then add the config + storage foundation, then infra (remotes + compose), then the two new DVC stages, then MLflow logging, then tests. Each phase is independently verifiable.

---

## Phase 0: Finalize cleanup & unbreak imports

### Changes Required

#### 1. Config package init
**File**: `src/allium_cepa_classifier/config/__init__.py`
**Changes**: Remove `controlnet_config`/`vae_config` imports and `__all__` entries.

#### 2. Remove orphaned dvclive block
**File**: `dvc.yaml`
**Changes**: Delete the top-level `params:`/`metrics:`/`plots:`/`artifacts:` block (lines 107-124).

#### 3. Trim dead deps
**File**: `pyproject.toml`
**Changes**: Remove `diffusers`, `accelerate`, `transformers`, `datasets`, `safetensors` from `[project].dependencies` (only used by removed VAE/ControlNet). Keep `tensorboard`. Re-lock with `uv lock`. Also remove any residual VAE field still referenced in `config/training_config.py`.

Verify nothing else imports the removed modules before trimming:
```bash
grep -rn "diffusers\|accelerate\|vae_\|controlnet" src/ scripts/
```

### Success Criteria

#### Automated Verification:
- [x] Package imports: `uv run python -c "import allium_cepa_classifier.config"`
- [x] No dangling refs: `grep -rn "vae_config\|controlnet_config" src/ scripts/` returns nothing
- [x] DVC config parses: `uv run dvc status` (no YAML error)
- [x] Lock resolves: `uv lock --check` (or `uv sync` succeeds)
- [ ] Lint clean: `uv run ruff check .` (pre-existing issues in notebook/ui/push_weights, unrelated to Phase 0)

#### Manual Verification:
- [ ] `git status` shows the VAE/ControlNet deletions as intentional, nothing unexpected.

**Implementation Note**: Pause for confirmation before proceeding.

---

## Phase 1: ValidationConfig + storage provider foundation

### Changes Required

#### 1. ValidationConfig
**File**: `src/allium_cepa_classifier/config/validation_config.py`

```python
from __future__ import annotations

from pydantic import ConfigDict

from .base_config import BaseConfig


class ValidationConfig(BaseConfig):
    """Thresholds for the model validation gate."""

    model_config = ConfigDict(frozen=True)

    min_f1_delta: float = 0.01          # new Macro F1 must beat prod by at least this
    per_class_tolerance: float = 0.03   # no class F1 may drop more than this
    ece_tolerance: float = 0.02         # new ECE may not exceed prod ECE by more than this
    metric_key: str = "macro_f1"
```
Register in `config/__init__.py`.

#### 2. Storage provider (the mock→real swap foundation)
**File**: `src/allium_cepa_classifier/providers/base.py`

```python
from pathlib import Path
from typing import Protocol


class StorageProvider(Protocol):
    """Minimal object-store interface. Paths are bucket-relative keys."""

    def get_file(self, key: str, local_path: Path) -> Path: ...
    def put_file(self, local_path: Path, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...
    def read_text(self, key: str) -> str: ...
```

**File**: `src/allium_cepa_classifier/providers/storage.py`

```python
from pathlib import Path

import fsspec  # via s3fs; works for both MinIO and AWS S3


class FsspecStorage:
    """One impl for local/MinIO/S3 — differs only by endpoint_url, exactly per the doc."""

    def __init__(self, bucket: str, endpoint_url: str | None):
        self.bucket = bucket
        self.fs = fsspec.filesystem(
            "s3",
            client_kwargs={"endpoint_url": endpoint_url} if endpoint_url else {},
        )

    # implement get_file/put_file/exists/read_text against f"{self.bucket}/{key}"
```

**File**: `src/allium_cepa_classifier/providers/factory.py`

```python
import os


def get_storage():
    backend = os.getenv("ALLIUM_STORAGE", "minio")   # minio (default) | s3
    bucket = os.getenv("ALLIUM_BUCKET", "allium-cepa-ml")
    endpoint = None if backend == "s3" else os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    return FsspecStorage(bucket=bucket, endpoint_url=endpoint)
```

Add `s3fs` to dependencies. This is the entire mock→real swap: set `ALLIUM_STORAGE=s3`.

### Success Criteria

#### Automated Verification:
- [x] `uv run python -c "from allium_cepa_classifier.config import ValidationConfig; ValidationConfig()"`
- [x] `uv run python -c "from allium_cepa_classifier.providers.factory import get_storage"`
- [ ] Lint passes: `uv run ruff check .` (pre-existing issues unrelated to Phase 1)

#### Manual Verification:
- [ ] Defaults (no env vars set) resolve to MinIO at `localhost:9000`.

---

## Phase 2: DVC remotes (MinIO + production S3)

### Changes Required

**File**: `.dvc/config` (and `.dvc/config.local` for secrets)
**Changes**: Add two remotes alongside the existing HF remotes (keep those for the public dataset cache). Document the switch in `CLAUDE.md`.

```ini
['remote "local_minio"']
    url = s3://allium-cepa-ml
    endpointurl = http://localhost:9000
['remote "production"']
    url = s3://allium-cepa-ml
```

Credentials (`access_key_id`/`secret_access_key`) go in `.dvc/config.local` (git-ignored). Switch with `dvc remote default local_minio` ↔ `dvc remote default production`.

### Success Criteria

#### Automated Verification:
- [x] `uv run dvc remote list` shows `local_minio` and `production`
- [x] `uv run dvc config core.remote` returns the chosen default

#### Manual Verification:
- [ ] With MinIO up (Phase 3): `dvc push -r local_minio` of a small out succeeds.

---

## Phase 3: docker-compose for MinIO + MLflow (dev infra)

### Changes Required

**File**: `docker-compose.yml`
**Changes**: Services `minio` (`:9000` API, `:9001` console), `mlflow` (`:5000`, artifact store `s3://allium-cepa-ml/mlflow`, backend sqlite), and a `createbuckets` init container that makes the `allium-cepa-ml` bucket. Env from `.env` (`.env.example` committed, `.env` git-ignored).

**File**: `.env.example` — `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`, `MLFLOW_TRACKING_URI`, etc.

### Success Criteria

#### Automated Verification:
- [x] `docker-compose config` validates the file
- [ ] `docker-compose up -d minio mlflow` then `curl -s localhost:9000/minio/health/live` returns 200
- [ ] `curl -s localhost:5000` returns the MLflow UI

#### Manual Verification:
- [ ] MinIO console at `:9001` shows the `allium-cepa-ml` bucket.
- [ ] MLflow UI at `:5000` loads with no experiments yet.

**Implementation Note**: Pause for confirmation before proceeding.

---

## Phase 4: `evaluate` DVC stage

### Overview
Evaluate the full two-stage `AlliumCepaModel` on the fixed test set and emit a consolidated report with the gate's metrics (Macro F1, per-class F1, accuracy, ECE).

### Changes Required

#### 1. Evaluation script
**File**: `scripts/evaluate.py`
**Changes**: Load the calibrated detector+classifier into `AlliumCepaModel`, run over the fixed test split, match predictions to ground truth (reuse the IoU≥0.5 greedy matching already in `detector_calibrator.py`), compute classifier metrics on matched crops with sklearn (`f1_score(average=None)` + macro, `accuracy_score`) and ECE (reuse the calibration ECE helper), and write `metrics/evaluation_report.json`:

```json
{"macro_f1": 0.0, "f1_per_class": {"mitosis": 0.0, "no_mitosis": 0.0},
 "accuracy": 0.0, "ece": 0.0, "n_samples": 0}
```

Test set source: for Plan A use the local test split already produced by `prepare_crops` (`datasets/crops/binary_classifier/test`); the doc's `s3://dataset/test_fixed/` becomes a swap-in later via the storage provider (note this in the script).

#### 2. DVC stage
**File**: `dvc.yaml`

```yaml
  evaluate:
    cmd: .venv/bin/python scripts/evaluate.py --output metrics/evaluation_report.json
    deps:
    - scripts/evaluate.py
    - src/allium_cepa_classifier/data_models/
    - experiments/binary_classifier/efficientnet_b1/weights/classifier_calibrated.pt
    - experiments/yolo/yolo11n_200e/weights/object_detection.pt
    - experiments/yolo/yolo11n_200e/weights/yolo_isotonic_calibrator.pkl
    - datasets/crops/binary_classifier/test
    metrics:
    - metrics/evaluation_report.json:
        cache: false
```
(Which classifier arch feeds production is a config choice — default `efficientnet_b1`; confirm.)

### Success Criteria

#### Automated Verification:
- [ ] `uv run python scripts/evaluate.py --output /tmp/report.json` produces valid JSON with all keys
- [ ] `uv run dvc repro evaluate` succeeds and writes `metrics/evaluation_report.json`
- [x] Lint passes: `uv run ruff check scripts/evaluate.py`

#### Manual Verification:
- [ ] Reported Macro F1 is in a sane range vs the existing `metrics.json` accuracy.

---

## Phase 5: `validate_model` gate + DVC stage

### Changes Required

#### 1. Gate
**File**: `src/allium_cepa_classifier/validation/validate_model.py`
**Changes**: Load new `evaluation_report.json` and the production baseline `metrics.json` (via `get_storage().read_text("models/production/metrics.json")`; if absent → auto-approve "first model" and warn). Apply `ValidationConfig` rules: Macro F1 beats prod by ≥ `min_f1_delta`; no class F1 drops > `per_class_tolerance`; ECE not worse than prod by > `ece_tolerance`. Write `validation_result.json` `{approved, reasons, new_metrics, current_metrics}` and exit `0`/`1`.

#### 2. DVC stage
**File**: `dvc.yaml`

```yaml
  validate_model:
    cmd: .venv/bin/python -m allium_cepa_classifier.validation.validate_model
      --report metrics/evaluation_report.json --output validation_result.json
    deps:
    - src/allium_cepa_classifier/validation/validate_model.py
    - metrics/evaluation_report.json
    outs:
    - validation_result.json:
        cache: false
```

### Success Criteria

#### Automated Verification:
- [ ] `uv run pytest tests/test_validate_model.py` passes (Phase 7)
- [ ] `uv run dvc repro validate_model` writes `validation_result.json`
- [x] Exit code reflects decision: approve→0, reject→1 (implemented in main())
- [x] Lint passes: `uv run ruff check src/allium_cepa_classifier/validation/`
- [x] Imports: `from allium_cepa_classifier.validation import ValidationResult, run_validation`

#### Manual Verification:
- [ ] With no production baseline present, gate auto-approves and warns clearly.
- [ ] Hand-crafted worse metrics produce `approved:false` with readable reasons.

**Implementation Note**: Pause for confirmation before proceeding.

---

## Phase 6: MLflow integration in existing trainers/calibrators

### Changes Required

**Files**: `src/allium_cepa_classifier/training/trainer.py`, `scripts/train_detector.py`, `training/calibrator.py`, `training/detector_calibrator.py`
**Changes**: Add a tiny optional helper (`training/mlflow_logging.py`) that **no-ops when `MLFLOW_TRACKING_URI` is unset** (so local dev without Docker still works). Wrap each run: `mlflow.start_run`, `log_params` (from the config), `log_metrics` (the metrics dict already computed), `log_artifact(used_config.yaml)` and the produced weights. ECE before/after logged in the calibrators. Add `mlflow` to dependencies.

### Success Criteria

#### Automated Verification:
- [x] mlflow imports: `uv run python -c "import mlflow"`
- [x] `MLFLOW_TRACKING_URI` unset → training runs unaffected (no-op): context manager is a no-op
- [x] Lint passes: `uv run ruff check` on all modified scripts

#### Manual Verification:
- [ ] With MLflow up, a short `train_classifier` run appears in the UI with params, metrics, and the config artifact.

---

## Phase 7: Tests (`tests/` bootstrap)

### Changes Required

**Files**: `tests/__init__.py`, `tests/conftest.py`, `tests/test_validate_model.py`, `tests/test_storage_provider.py`, `tests/test_evaluate.py`
**Changes**:

- `test_validate_model.py`: synthetic new/prod metric dicts → assert approve/reject for each rule (beats threshold, per-class regression, ECE regression, missing baseline auto-approve).
- `test_storage_provider.py`: factory returns MinIO impl by default, S3 impl when `ALLIUM_STORAGE=s3` (monkeypatch env; no network — assert config/endpoint, mock `fsspec.filesystem`).
- `test_evaluate.py`: metric computation on a tiny synthetic detections/GT fixture.

### Success Criteria

#### Automated Verification:
- [x] `uv run pytest` passes (22/22)
- [x] `uv run pytest --cov=src` reports coverage for `validation/` (85%) and `providers/factory` (100%)
- [ ] Lint passes: `uv run ruff check .` (pre-existing issues in notebook/ui, unrelated to Phase 7)

#### Manual Verification:
- [x] Tests run with no Docker/MinIO/network dependency (fully isolated — storage provider is mocked).

---

## Testing Strategy

- **Unit:** gate decision logic (all branches), storage factory selection, evaluate metric math.
- **Integration:** `dvc repro evaluate validate_model` on the real local artifacts.
- **Manual:** MLflow UI shows a run; MinIO console shows the bucket; gate rejects a deliberately worse model.

## Performance Considerations

`evaluate` runs full inference over the test split once — bounded by test-set size, acceptable for a DVC stage. MLflow logging is negligible. No GPU needed for the gate or tests.

## Migration Notes

- First `validate_model` run has no `models/production/metrics.json` → auto-approves and seeds the baseline (documented behavior, not an error).
- Existing per-experiment `metrics.json` files are untouched; `evaluation_report.json` is new and consolidated. The gate reads only the consolidated report + production baseline.

## References

- Architecture doc: [`thoughts/shared/research/allium_cepa_implementation_plan.md`](../research/allium_cepa_implementation_plan.md) (§7, §11)
- Roadmap: [`2026-06-06-portfolio-roadmap.md`](2026-06-06-portfolio-roadmap.md)
- Inference model: [`src/allium_cepa_classifier/data_models/allium_cepa_model.py:323`](../../../src/allium_cepa_classifier/data_models/allium_cepa_model.py#L323)
- IoU matching to reuse: [`src/allium_cepa_classifier/training/detector_calibrator.py`](../../../src/allium_cepa_classifier/training/detector_calibrator.py)
- Config pattern: [`src/allium_cepa_classifier/config/detector_config.py`](../../../src/allium_cepa_classifier/config/detector_config.py)
- Existing DVC stages: [`dvc.yaml`](../../../dvc.yaml)
