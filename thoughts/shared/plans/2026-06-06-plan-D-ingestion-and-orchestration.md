# Plan D — Ingestion & Orchestration — Implementation Plan

> Milestone 4 of 4 (heaviest, scheduled last). See [`2026-06-06-portfolio-roadmap.md`](2026-06-06-portfolio-roadmap.md) for the full program. Depends on **Plan A** (the `providers/` base + storage provider, the `evaluate`/`validate_model` DVC stages, the `models/production/` layout). Reuses **Plan B**'s `ProductionConfig` thresholds.

## Overview

Close the **continuous-training, human-in-the-loop loop**: ingest expert classifications from Zooniverse and raw images from Google Drive, route model predictions by confidence, publish validated data to HuggingFace (bumping the pinned dataset SHA), and orchestrate the whole retraining + promotion cycle with Airflow. After this plan: the Airflow UI runs three DAGs (`zooniverse_ingest`, `raw_image_ingest`, `retrain_pipeline`) **entirely against mock providers with zero external accounts**; flipping any one provider to real is a single env-var change, exactly as established in Plan A.

This plan completes the "providers" convention: it adds the **labeling**, **drive**, and **dataset_hub** providers (mock + real each) alongside Plan A's storage provider, and wires them into the DAGs.

## Current State Analysis

Verified against the repo on branch `main` (2026-06-06):

- **No `airflow/`, no `ingestion`/labeling/drive/hub providers, no `ZooniverseConfig`.** All greenfield.
- **Plan A delivers the foundation D builds on:** `providers/base.py` (Protocol pattern), `providers/factory.py` (the `get_*()` env-switch convention), `providers/storage.py` (boto3/fsspec to MinIO/S3), and the `evaluate` + `validate_model` DVC stages that emit `validation_result.json`.
- **`AlliumCepaModel`** is directly importable and is the detection+classification engine `raw_image_ingest` needs (instantiate with `ProductionConfig` weights pulled via the storage provider, per Plan B's `ensure_production_weights`).
- **Dataset versioning is SHA-pinned:** `download_dataset` in `dvc.yaml` pins the HF dataset via `--rev <sha>`. The loop's publish step must update that `--rev` and commit `dvc.yaml` so the next `dvc repro` re-downloads.
- **The gate already exists (Plan A):** `validate_model` writes `validation_result.json {approved, ...}` and exits 0/1. `retrain_pipeline` only needs to *read* it and act — it does not reimplement gate logic.
- **`airflow` is not a dependency**, and Airflow's own deps are heavy; it lives in its own Docker service + dependency group, not the inference image.

### Key Discoveries

- The retrain DAG is thin: it shells out to `dvc repro` and reads `validation_result.json`. DVC + the gate (Plan A) do the real work — Airflow is orchestration only, matching the architecture doc ("Airflow no conoce los internals del pipeline").
- Every external touchpoint (Zooniverse, Drive, HF) maps cleanly onto one provider with a mock impl, so the entire loop is testable and demoable offline with fixtures.
- Confidence routing reuses `ProductionConfig.high_confidence_threshold` (Plan B) — no new threshold config needed beyond Zooniverse's consensus settings.

## Desired End State

- `docker-compose up airflow` (extending Plan A/B's compose) serves the Airflow UI at `:8080` with the three DAGs registered.
- With all providers at their **mock** defaults, each DAG runs green end-to-end from the Airflow UI using committed fixtures — no Zooniverse/Drive/HF/AWS account required.
- `ALLIUM_LABELING=zooniverse` / `ALLIUM_DRIVE=gdrive` / `ALLIUM_HUB=hf` (plus the relevant secrets) switch the corresponding provider to its real impl **with no code change**.
- `retrain_pipeline` runs `dvc repro`, reads `validation_result.json`, and on approval copies weights to `models/production/` + updates `models/production/metrics.json` via the storage provider.
- `uv run pytest` covers each provider's factory selection and each DAG's task logic against mocks (no network).

## What We're NOT Doing (this plan)

- Wiring real Zooniverse/Drive/AWS/HF credentials by default — mocks are the default; real impls are exercised only when the env switch + secrets are set.
- Running Airflow 24/7 in production — it's on-demand/nightly (local machine or optional EC2 `t2.micro`), per the architecture doc.
- Re-implementing the validation gate or DVC stages (Plan A owns them).
- Building a new model — `raw_image_ingest` uses the **production** model for inference only.
- Standing up a managed Airflow (MWAA/Composer) — local Docker `LocalExecutor` + SQLite/Postgres is enough for the portfolio.

## Implementation Approach

Providers first (each a mock + real behind the factory), then `ZooniverseConfig`, then the three DAGs (in dependency order: ingest DAGs before the retrain DAG), then the Airflow Docker service, then tests against mocks. Every phase is verifiable offline.

---

## Phase 1: Labeling, Drive, and Dataset-Hub providers

### Overview
Three new capabilities following Plan A's `base.py` Protocol + `factory.py` env-switch convention. Each gets a **mock** impl (fixtures/local dirs, the default) and a **real** impl (the actual SDK), selected by one env var.

### Changes Required

#### 1. Protocols
**File**: `src/allium_cepa_classifier/providers/base.py` (extend)

```python
from pathlib import Path
from typing import Protocol


class LabelingProvider(Protocol):
    """Expert-labeling backend (Zooniverse)."""
    def fetch_classifications(self, since: str | None) -> list[dict]: ...
    def create_tasks(self, image_keys: list[str], priority: str = "normal") -> None: ...


class DriveProvider(Protocol):
    """Raw-image source (Google Drive)."""
    def list_new(self, since: str | None) -> list[str]: ...
    def download(self, file_id: str, local_path: Path) -> Path: ...


class DatasetHubProvider(Protocol):
    """Ground-truth dataset hub (HuggingFace). Returns the new commit SHA on publish."""
    def publish(self, images_dir: Path, annotations: dict) -> str: ...
```

#### 2. Implementations
**Files**:
- `src/allium_cepa_classifier/providers/labeling.py` — `MockZooniverse` (reads `tests/fixtures/zooniverse/*.json`, records created tasks in a local dir) | `RealZooniverse` (`panoptes-client`).
- `src/allium_cepa_classifier/providers/drive.py` — `MockDrive` (lists/copies from a local fixtures dir) | `RealDrive` (`google-api-python-client` + `google-auth`).
- `src/allium_cepa_classifier/providers/dataset_hub.py` — `MockHub` (writes to a local dir, returns a fake deterministic SHA) | `RealHFHub` (`huggingface_hub.HfApi.upload_folder`, returns the real SHA).

#### 3. Factory
**File**: `src/allium_cepa_classifier/providers/factory.py` (extend)

```python
def get_labeling():
    backend = os.getenv("ALLIUM_LABELING", "mock")   # mock (default) | zooniverse
    ...

def get_drive():
    backend = os.getenv("ALLIUM_DRIVE", "mock")       # mock (default) | gdrive
    ...

def get_dataset_hub():
    backend = os.getenv("ALLIUM_HUB", "mock")         # mock (default) | hf
    ...
```

Add `panoptes-client`, `google-api-python-client`, `google-auth` to an `ingestion` dependency group (not the inference image). `huggingface_hub` is already present (used by `download_dataset`).

### Success Criteria

#### Automated Verification:
- [ ] `uv run python -c "from allium_cepa_classifier.providers.factory import get_labeling, get_drive, get_dataset_hub"`
- [ ] Defaults resolve to mock impls: a test asserts `type(get_labeling()).__name__ == "MockZooniverse"` with no env set.
- [ ] Env switch works: `ALLIUM_LABELING=zooniverse` selects `RealZooniverse` (assert class, no network).
- [ ] Lint passes: `uv run ruff check .`

#### Manual Verification:
- [ ] `MockHub.publish()` returns a deterministic SHA and writes the local dir; `MockDrive`/`MockZooniverse` operate purely on fixtures.

**Implementation Note**: Pause for confirmation before proceeding.

---

## Phase 2: ZooniverseConfig

### Changes Required

**File**: `src/allium_cepa_classifier/config/zooniverse_config.py`

```python
from __future__ import annotations

from pydantic import ConfigDict

from .base_config import BaseConfig


class ZooniverseConfig(BaseConfig):
    """Zooniverse ingestion + consensus settings."""

    model_config = ConfigDict(frozen=True)

    project_id: str = ""                 # from env/secret in real mode
    consensus_threshold: float = 0.8     # min expert agreement to accept a label
    min_new_for_publish: int = 100       # don't publish to HF until this many validated samples
```
Register in `config/__init__.py`. Credentials come from env (`ZOONIVERSE_USERNAME`/`ZOONIVERSE_PASSWORD`), never the config file.

### Success Criteria

#### Automated Verification:
- [ ] `uv run python -c "from allium_cepa_classifier.config import ZooniverseConfig; ZooniverseConfig()"`
- [ ] Lint passes: `uv run ruff check .`

---

## Phase 3: DAG A — `zooniverse_ingest`

### Overview
Daily DAG: pull expert classifications → filter by consensus → normalize to COCO → stage to storage → (if enough new data) publish to HF and bump the pinned SHA in `dvc.yaml` → notify retrain.

### Changes Required

**File**: `airflow/dags/zooniverse_ingest.py`
**Tasks** (each a thin Python callable delegating to providers / existing code):

1. `download_zooniverse_classifications` — `get_labeling().fetch_classifications(since=last_ts)`; persist raw to a temp/staging key. *(retry: network)*
2. `filter_by_consensus` — keep agreement ≥ `ZooniverseConfig.consensus_threshold`; log accepted/discarded. *(deterministic, no retry)*
3. `normalize_to_coco_format` — convert to the COCO schema used by `download_dataset`; validate one label per image; tag `source: "zooniverse"`. *(deterministic)*
4. `upload_validated_to_s3` — `get_storage().put_file(...)` under `dataset/labeled/zooniverse/`. *(retry)*
5. `publish_to_huggingface` — **only if** new count ≥ `ZooniverseConfig.min_new_for_publish`; `get_dataset_hub().publish(images_dir, annotations)` → returns new SHA. *(retry)*
6. `update_dvc_yaml_sha` — rewrite `--rev <sha>` in `dvc.yaml`; `git commit` + `git push` (subprocess). *(retry)*
7. `notify_retrain` — TriggerDagRunOperator → `retrain_pipeline`.

- **Schedule:** `@daily` (3 AM). Manually triggerable from the UI.
- **Branching:** tasks 5-7 only fire when the volume threshold is met (BranchPythonOperator or short-circuit).

### Success Criteria

#### Automated Verification:
- [ ] DAG imports without errors: `airflow dags list` includes `zooniverse_ingest` (or `python -c` parse via `DagBag` in a test).
- [ ] `uv run pytest tests/test_dag_zooniverse.py` — task callables run against mocks: consensus filter math, COCO normalization shape, publish-gating on `min_new_for_publish`.
- [ ] Lint passes: `uv run ruff check .`

#### Manual Verification:
- [ ] In the Airflow UI (mock providers), a manual run completes; below-threshold runs short-circuit before publish; above-threshold runs reach `notify_retrain`.

---

## Phase 4: DAG B — `raw_image_ingest`

### Overview
On-demand DAG: pull raw images from Drive → run the **production** `AlliumCepaModel` (detect + classify + score) → route crops by confidence → send low-confidence crops to Zooniverse → notify retrain if enough new auto-labeled data.

### Changes Required

**File**: `airflow/dags/raw_image_ingest.py`
**Tasks**:

1. `download_from_drive` — `get_drive().list_new()` + `download()`; validate format; `put_file` to `dataset/raw/`. *(retry)*
2. `run_instance_detection` — `ensure_production_weights(ProductionConfig())` (Plan B) → `AlliumCepaModel`; run detection, emit per-cell crops. *(no retry — deterministic given weights)*
3. `classify_and_score` — classify crops; produce `(class, confidence, ci_lower, ci_upper)` per crop via the model.
4. `route_by_confidence` — `confidence ≥ ProductionConfig.high_confidence_threshold` → `dataset/labeled/auto/` (`source: "auto"`); else → `dataset/review/`. Log the confidence distribution.
5. `send_low_confidence_to_zooniverse` — `get_labeling().create_tasks(review_keys, priority=...)`; very-low confidence flagged high priority.
6. `notify_retrain_if_threshold` — if `labeled/auto/` volume exceeds a threshold, TriggerDagRunOperator → `retrain_pipeline`.

- **Schedule:** `None` (manual / file-sensor trigger).

### Success Criteria

#### Automated Verification:
- [ ] DAG parses and registers.
- [ ] `uv run pytest tests/test_dag_raw_image.py` — routing logic with synthetic `(confidence)` lists asserts auto/review split at the threshold; `create_tasks` called with the review set (mock labeling).
- [ ] Lint passes: `uv run ruff check .`

#### Manual Verification:
- [ ] UI run against `MockDrive` fixtures produces a logged confidence distribution and a non-empty review set routed to `MockZooniverse`.

**Implementation Note**: Pause for confirmation before proceeding.

---

## Phase 5: DAG C — `retrain_pipeline`

### Overview
Triggered by A or B (or manual): check for dataset changes → `dvc repro` (which runs train→calibrate→evaluate→validate_model from Plan A) → read `validation_result.json` → promote or archive.

### Changes Required

**File**: `airflow/dags/retrain_pipeline.py`
**Tasks**:

1. `check_dataset_changes` — compare current `dvc.yaml` SHA vs last-trained SHA; short-circuit downstream if unchanged.
2. `run_dvc_repro` — `subprocess` `dvc repro` (re-runs only affected stages; MLflow logs during the run). *(long-running)*
3. `read_validation_result` — load `validation_result.json`; BranchPythonOperator → `promote_model` or `archive_model`.
4. `promote_model` *(if approved)* — `get_storage().put_file` the three weight files to `models/production/`; update `models/production/metrics.json`; transition the MLflow registry version to `Production`; call the HF Spaces restart API (real mode) so Plan B's API re-pulls weights; notify success with the metric delta.
5. `archive_model` *(if rejected)* — mark the MLflow version `Archived`; log the comparative metrics; notify rejection with the gate's reasons.

- **Schedule:** `None` (triggered). The HF Spaces restart is itself behind the same mock→real pattern (no-op in mock mode).

### Success Criteria

#### Automated Verification:
- [ ] DAG parses and registers.
- [ ] `uv run pytest tests/test_dag_retrain.py` — `read_validation_result` branches correctly for `{approved:true}` vs `{approved:false}` fixtures; `promote_model` issues the expected `put_file` calls against mock storage.
- [ ] Lint passes: `uv run ruff check .`

#### Manual Verification:
- [ ] UI run with an `approved:true` `validation_result.json` reaches `promote_model` and writes mock `models/production/`; an `approved:false` reaches `archive_model`.
- [ ] `dvc repro` invocation is correct (dry-run/echo acceptable in the mock demo).

---

## Phase 6: Airflow Docker service + compose wiring

### Overview
Add Airflow to the stack as its own service + dependency group, sharing the storage/provider env so DAGs reach MinIO and (mock) providers.

### Changes Required

- **File**: `docker-compose.yml` (extend) — `airflow` service (`LocalExecutor`, SQLite or a small Postgres), mounts `airflow/dags/`, `:8080`, env includes `ALLIUM_*` provider switches (defaulting to mock), `MINIO_ENDPOINT`, MinIO creds. Init container runs `airflow db init` + creates an admin user.
- **File**: `pyproject.toml` — `ingestion` group with `apache-airflow` (pinned, with the matching constraints file) + the provider SDKs from Phase 1.
- **File**: `.env.example` (extend) — `ZOONIVERSE_*`, `GOOGLE_DRIVE_CREDENTIALS_PATH`, `HF_TOKEN`, `AIRFLOW__CORE__*` as needed (all optional; mocks need none).

### Success Criteria

#### Automated Verification:
- [ ] `docker-compose config` validates with the `airflow` service.
- [ ] `docker-compose up -d airflow` → `curl -s localhost:8080/health` returns healthy.
- [ ] `airflow dags list` (inside the container) shows all three DAGs with no import errors.

#### Manual Verification:
- [ ] Airflow UI at `:8080` lists the three DAGs; each can be triggered manually and runs green against mock providers.

**Implementation Note**: Pause for confirmation before proceeding.

---

## Phase 7: Tests + fixtures

### Changes Required

**Files**: `tests/fixtures/zooniverse/*.json`, `tests/fixtures/drive/*`, `tests/test_providers_ingestion.py`, `tests/test_dag_zooniverse.py`, `tests/test_dag_raw_image.py`, `tests/test_dag_retrain.py`

- `test_providers_ingestion.py`: factory returns mock by default, real on env switch (assert class, no network) for labeling/drive/hub.
- DAG tests: import each DAG via `DagBag` (assert no import errors), then unit-test the task callables against mocks (consensus filter, COCO normalization, confidence routing, validation-result branching, promote `put_file` calls).

### Success Criteria

#### Automated Verification:
- [ ] `uv run pytest` passes (all DAG + provider tests).
- [ ] `DagBag(...).import_errors == {}` for `airflow/dags/`.
- [ ] Tests run with no network/Drive/Zooniverse/HF/AWS access.
- [ ] Lint passes: `uv run ruff check .`

#### Manual Verification:
- [ ] Coverage shows the new providers + DAG task functions exercised.

---

## Testing Strategy

- **Unit:** provider factory selection (mock vs real); each DAG task callable against mocks — consensus filtering, COCO normalization, confidence routing thresholds, validation-result branching, promotion `put_file` calls.
- **Integration:** `DagBag` import (no parse/import errors); `docker-compose up airflow` then a manual UI run of each DAG end-to-end against mock providers + MinIO.
- **Manual:** Airflow UI screenshots for Plan C's `docs/screenshots/airflow_dags.png`; flip one provider to real with creds to confirm the env switch path.

## Performance Considerations

- `run_dvc_repro` is the only heavy task — it runs full training on the local machine's GPU; Airflow just supervises it. No GPU is needed for any other task or for the tests.
- Airflow's own footprint is contained in its Docker service; the inference image (Plan B) stays slim and does not depend on Airflow.

## Migration Notes

- `update_dvc_yaml_sha` performs a real `git commit`/`push` in production; in the mock demo it operates on a throwaway branch or is a no-op (guard behind the hub mode) to avoid polluting history.
- The first promotion seeds `models/production/metrics.json`, consistent with Plan A's "first model auto-approves" behavior.
- Auto-labeled (`source: "auto"`) samples are tagged so training can include/exclude them via `ExperimentConfig`; the **fixed test set never receives auto-labeled data** (architecture doc §12) — preserve this when normalizing/publishing.

## References

- Architecture doc: [`thoughts/shared/research/allium_cepa_implementation_plan.md`](../research/allium_cepa_implementation_plan.md) (§5.5, §6, §8, §12, §13)
- Roadmap: [`2026-06-06-portfolio-roadmap.md`](2026-06-06-portfolio-roadmap.md)
- Plan A (providers base, factory, storage, gate, DVC stages): [`2026-06-06-plan-A-reproducibility-core.md`](2026-06-06-plan-A-reproducibility-core.md)
- Plan B (`ProductionConfig`, `ensure_production_weights`): [`2026-06-06-plan-B-serving-and-demo.md`](2026-06-06-plan-B-serving-and-demo.md)
- Inference engine for DAG B: [`src/allium_cepa_classifier/data_models/allium_cepa_model.py`](../../../src/allium_cepa_classifier/data_models/allium_cepa_model.py)
- SHA-pinned dataset stage: [`dvc.yaml`](../../../dvc.yaml) (`download_dataset`)
