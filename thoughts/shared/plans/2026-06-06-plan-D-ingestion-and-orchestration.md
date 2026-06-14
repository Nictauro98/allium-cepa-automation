# Plan D — Ingestion & Orchestration — Implementation Plan

> Milestone 4 of 4 (heaviest, scheduled last). See [`2026-06-06-portfolio-roadmap.md`](2026-06-06-portfolio-roadmap.md) for the full program. Depends on **Plan A** (the `providers/` base + storage provider, the `evaluate`/`validate_model` DVC stages, the `models/production/` layout). Reuses **Plan B**'s `ProductionConfig` thresholds.

## Overview

Close the **continuous-training, human-in-the-loop loop**: ingest expert phase classifications from Zooniverse and raw images from Google Drive, route model predictions by confidence, publish new images and enriched annotations to HuggingFace (bumping the pinned dataset SHA), and orchestrate the whole retraining + promotion cycle with Airflow. After this plan: the Airflow UI runs three DAGs (`zooniverse_ingest`, `raw_image_ingest`, `retrain_pipeline`) **entirely against mock providers with zero external accounts**; flipping any one provider to real is a single env-var change, exactly as established in Plan A.

This plan completes the "providers" convention: it adds the **labeling**, **drive**, and **dataset_hub** providers (mock + real each) alongside Plan A's storage provider, and wires them into the DAGs.

**Storage split (deliberate architectural decision):**
- **HuggingFace** — single source of truth for the dataset (full images + COCO annotations, versioned by commit SHA). All new images enter HF immediately with pending annotations; all Zooniverse verdicts are written back to HF.
- **S3 / MinIO** — model artifacts only: production weights and MLflow run artifacts. No image or annotation data goes through S3.

This separation is intentional: HF is a dataset hub with built-in versioning and free storage — the right tool for data. S3 is the industry-standard artifact store for model weights and MLflow, worth showing in a portfolio context.

## Current State Analysis

Verified against the repo on branch `main` (2026-06-06):

- **No `airflow/`, no `ingestion`/labeling/drive/hub providers, no `ZooniverseConfig`.** All greenfield.
- **Plan A delivers the foundation D builds on:** `providers/base.py` (Protocol pattern), `providers/factory.py` (the `get_*()` env-switch convention), `providers/storage.py` (boto3/fsspec to MinIO/S3, **scoped to model artifacts only in this plan**), and the `evaluate` + `validate_model` DVC stages that emit `validation_result.json`.
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
- Using S3 for image or annotation storage — HF is the single dataset store; S3/MinIO is scoped to model artifacts and MLflow only.
- Writing Drive images to S3 staging — images go directly from Drive to HF (local temp dir only during the DAG run).
- Adding Drive-originated images to the test or validation splits — those are frozen; `raw_image_ingest` always publishes to `"train"`.

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
    """Ground-truth dataset hub (HuggingFace). Both methods return the new commit SHA.

    Two distinct operations:
    - publish_images: upload new full images + their COCO annotations (Drive-originated).
      New annotations carry division=None for pending crops sent to Zooniverse,
      or division set for high-confidence auto-labeled crops.
    - patch_annotations: update attributes on existing annotations only (Zooniverse verdicts).
      Never uploads image data.
    """
    def publish_images(self, split: str, images: list[Path], annotations: list[dict]) -> str: ...
    def patch_annotations(self, split: str, annotation_patches: list[dict]) -> str: ...
```

#### 2. Implementations
**Files**:
- `src/allium_cepa_classifier/providers/labeling.py` — `MockZooniverse` (reads `tests/fixtures/zooniverse/*.json`, records created tasks in a local dir) | `RealZooniverse` (`panoptes-client`).
- `src/allium_cepa_classifier/providers/drive.py` — `MockDrive` (lists/copies from a local fixtures dir) | `RealDrive` (`google-api-python-client` + `google-auth`).
- `src/allium_cepa_classifier/providers/dataset_hub.py` — `MockHub` (writes images to a local dir, merges annotations into a local `annotations.json`, returns a fake deterministic SHA) | `RealHFHub`:
  - `publish_images`: packs images into parquet shards (`upload_hf_dataset.py` logic), uploads shards + updated `annotations.json` via `HfApi.upload_large_folder` / `upload_file`.
  - `patch_annotations`: uploads only the updated `{split}/data/annotations.json` via `HfApi.upload_file`.
  Both return the real commit SHA.

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
- [x] `uv run python -c "from allium_cepa_classifier.providers.factory import get_labeling, get_drive, get_dataset_hub"`
- [x] Defaults resolve to mock impls: a test asserts `type(get_labeling()).__name__ == "MockZooniverse"` with no env set.
- [x] Env switch works: `ALLIUM_LABELING=zooniverse` selects `RealZooniverse` (assert class, no network).
- [x] Lint passes: `uv run ruff check .`

#### Manual Verification:
- [ ] `MockHub.publish_images()` writes images to a local dir and merges new annotations into local `annotations.json`; returns deterministic SHA.
- [ ] `MockHub.patch_annotations()` merges the attribute patch into the local `annotations.json`; returns deterministic SHA.
- [ ] `MockDrive`/`MockZooniverse` operate purely on fixtures.

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
    min_new_for_patch: int = 10          # don't push to HF until this many annotations resolved

    # Phase taxonomy returned by Zooniverse classifiers.
    # Determines what is written to attributes.phase and whether attributes.division is updated.
    #
    # Dividing phases (division → 1):
    #   prophase, metaphase, anaphase, telophase, chromosomal_aberration
    # Non-dividing (division → 0):
    #   interphase
    # Quality flags — phase is recorded but division is NOT changed:
    #   indeterminate  → keep existing division; mark attributes.phase = "indeterminate"
    #   not_a_cell     → keep existing division; set attributes.detection_error = True
    #
    # The mapping below is the authoritative lookup used by the DAG.
    phase_division_map: dict[str, int | None] = {
        "prophase": 1,
        "metaphase": 1,
        "anaphase": 1,
        "telophase": 1,
        "chromosomal_aberration": 1,
        "interphase": 0,
        "indeterminate": None,   # None → do not update division
        "not_a_cell": None,      # None → do not update division; set detection_error flag
    }
```
Register in `config/__init__.py`. Credentials come from env (`ZOONIVERSE_USERNAME`/`ZOONIVERSE_PASSWORD`), never the config file.

### Success Criteria

#### Automated Verification:
- [x] `uv run python -c "from allium_cepa_classifier.config import ZooniverseConfig; ZooniverseConfig()"`
- [x] Lint passes: `uv run ruff check .`

---

## Phase 3: DAG A — `zooniverse_ingest`

### Overview
Daily DAG: pull expert phase classifications from Zooniverse → filter by consensus → **enrich existing COCO annotations** on HF (writing `attributes.phase` and updating `attributes.division` where unambiguous) → bump the pinned SHA in `dvc.yaml` → notify retrain.

**Key design constraint:** The HF dataset's `images/` parquet shards are **never touched** — only `{split}/data/annotations.json` is patched. Each Zooniverse classification is matched back to its source annotation by `annotation_id`, which was embedded as subject metadata when crops were uploaded to Zooniverse (crop filename convention: `{image_name}_{annotation_id}.jpg`).

### Changes Required

**File**: `airflow/dags/zooniverse_ingest.py`
**Tasks** (each a thin Python callable delegating to providers / existing code):

1. `download_zooniverse_classifications` — `get_labeling().fetch_classifications(since=last_ts)`; each record carries `annotation_id`, `split`, and the raw phase verdict. *(retry: network)*

2. `filter_by_consensus` — keep only subjects where expert agreement ≥ `ZooniverseConfig.consensus_threshold`. Majority phase is the verdict. Log accepted/discarded counts. *(deterministic, no retry)*

3. `apply_phase_to_annotations` — for each accepted verdict:
   - Look up the target annotation by `annotation_id` in the locally-fetched `annotations.json` for the correct split.
   - Write `attributes.phase = "<verdict>"`.
   - If `ZooniverseConfig.phase_division_map[verdict]` is not `None`: overwrite `attributes.division` (resolves `division=None` pending annotations from Drive images).
   - If verdict == `"not_a_cell"`: additionally set `attributes.detection_error = True`.
   - Produces one updated `annotations.json` per affected split. *(deterministic)*

4. `patch_huggingface_annotations` — **only if** resolved count ≥ `ZooniverseConfig.min_new_for_patch`; `get_dataset_hub().patch_annotations(split, annotation_patches)` for each affected split → returns new commit SHA. *(retry)*

5. `update_dvc_yaml_sha` — rewrite `--rev <sha>` in `dvc.yaml`; `git commit` + `git push` (subprocess). In mock mode: no-op (guard on hub provider type). *(retry)*

6. `notify_retrain` — TriggerDagRunOperator → `retrain_pipeline`.

- **Schedule:** `@daily` (3 AM). Manually triggerable from the UI.
- **Branching:** tasks 4-6 only fire when resolved count ≥ `min_new_for_patch` (ShortCircuitOperator).

### Success Criteria

#### Automated Verification:
- [x] DAG imports without errors: `airflow dags list` includes `zooniverse_ingest` (or `DagBag` import test).
- [x] `uv run pytest tests/test_dag_zooniverse.py` — task callables run against mocks:
  - [x] Consensus filter: majority-vote math at various agreement levels.
  - [x] Phase→division mapping: all 8 phase labels produce the correct `division` update or no-op.
  - [x] `"not_a_cell"` sets `detection_error=True` and leaves `division` unchanged.
  - [x] `"indeterminate"` writes `phase` attribute but does not touch `division`.
  - [x] `patch_annotations` is called once per affected split; `update_dvc_yaml_sha` is a no-op in mock mode.
  - [x] Below-threshold runs short-circuit before `patch_huggingface_annotations`.
- [x] Lint passes: `uv run ruff check .`

#### Manual Verification:
- [ ] In the Airflow UI (mock providers), a manual run with fixture classifications completes green; phase and division fields in the local `annotations.json` match expected values; below-threshold runs short-circuit before HF patch; above-threshold runs reach `notify_retrain`.

---

## Phase 4: DAG B — `raw_image_ingest`

### Overview
On-demand DAG: pull raw images from Drive → run the **production** `AlliumCepaModel` (detect + classify + score) → build COCO annotations with `division` set for high-confidence crops and `division=None` (pending) for low-confidence ones → publish the full image + all annotations to HF immediately → send low-confidence crops to Zooniverse for expert review → notify retrain if enough high-confidence auto-labels were added.

**Key design:** Every Drive image enters HF on this DAG run, regardless of confidence. Low-confidence bboxes are pending (`division=None`) until `zooniverse_ingest` resolves them. The training pipeline filters out `division=None` annotations via `prepare_crops`. This way the base dataset always grows — Zooniverse only adds quality, not gating.

### Changes Required

**File**: `airflow/dags/raw_image_ingest.py`
**Tasks**:

1. `download_from_drive` — `get_drive().list_new()` + `download()` to a local temp dir; validate image format. *(retry)*
2. `run_instance_detection` — `ensure_production_weights(ProductionConfig())` (Plan B) → `AlliumCepaModel`; run detection, emit per-cell crops with bboxes. *(no retry — deterministic given weights)*
3. `classify_and_score` — classify crops; produce `(bbox, class, confidence, ci_lower, ci_upper)` per crop via the model.
4. `build_coco_annotations` — for each detected bbox:
   - `confidence ≥ ProductionConfig.high_confidence_threshold` → `division` set from class prediction, `attributes.source = "auto"`. *(deterministic)*
   - else → `division = None` (pending), `attributes.source = "auto"`, `attributes.pending = True`.
   - Log the confidence distribution and auto/pending split counts.
5. `publish_to_huggingface` — `get_dataset_hub().publish_images(split="train", images=[full_image], annotations=coco_annotations)` → returns new commit SHA. The split for Drive images is always `"train"` — the fixed test set is never modified by this DAG. *(retry)*
6. `update_dvc_yaml_sha` — rewrite `--rev <sha>` in `dvc.yaml`; `git commit` + `git push`. In mock mode: no-op. *(retry)*
7. `send_pending_to_zooniverse` — `get_labeling().create_tasks(pending_annotation_ids, priority=...)` with subject metadata carrying `annotation_id`; very-low confidence flagged high priority. *(retry)*
8. `notify_retrain_if_threshold` — if auto-labeled (non-pending) count from this run ≥ threshold, TriggerDagRunOperator → `retrain_pipeline`.

- **Schedule:** `None` (manual / file-sensor trigger).

### Success Criteria

#### Automated Verification:
- [x] DAG parses and registers.
- [x] `uv run pytest tests/test_dag_raw_image.py` — task callables against mocks:
  - [x] `build_coco_annotations` with synthetic confidence lists: correct `division` set / `pending=True` at the threshold; `annotation_id` embedded in subject metadata.
  - [x] `publish_images` called with the full image and all annotations (pending + resolved).
  - [x] `update_dvc_yaml_sha` is a no-op in mock mode.
  - [x] `create_tasks` called only with the pending set.
  - [x] `notify_retrain_if_threshold` triggers only when auto-labeled count ≥ threshold.
- [x] Lint passes: `uv run ruff check .`

#### Manual Verification:
- [ ] UI run against `MockDrive` fixtures: logged confidence distribution shows auto/pending split; `MockHub.publish_images` receives the full image + both resolved and pending annotations; `MockZooniverse.create_tasks` receives only the pending set.

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
- [x] DAG parses and registers.
- [x] `uv run pytest tests/test_dag_retrain.py` — `read_validation_result` branches correctly for `{approved:true}` vs `{approved:false}` fixtures; `promote_model` issues the expected `put_file` calls against mock storage.
- [x] Lint passes: `uv run ruff check .`

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
- [x] `docker compose config` validates with the `airflow` service.
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
- `tests/fixtures/zooniverse/classifications.json` — synthetic Zooniverse export: a mix of all 8 phase labels, some below consensus threshold, one `not_a_cell`, one `indeterminate`, including `annotation_id` values that match `division=None` pending rows in `tests/fixtures/annotations_sample.json` (simulating Drive-originated crops).
- `tests/fixtures/drive/` — one or two small synthetic microscopy images for `MockDrive.list_new()` / `download()`.
- `tests/fixtures/annotations_sample.json` — a minimal COCO `annotations.json` with a mix of resolved (`division=0/1`) and pending (`division=None`, `pending=True`) annotations across splits.
- DAG tests: import each DAG via `DagBag` (assert no import errors), then unit-test task callables against mocks:
  - `test_dag_zooniverse.py`: consensus filter math; all 8 phase→division mapping rules (including no-op for `indeterminate`/`not_a_cell`); `detection_error` flag written correctly; pending `division=None` annotations resolved to correct `division` value after verdict; `patch_annotations` called per split; short-circuit at `min_new_for_patch`.
  - `test_dag_raw_image.py`: `build_coco_annotations` produces correct `division`/`pending` split at threshold; `publish_images` receives all annotations (pending + resolved); `create_tasks` receives only pending set with correct `annotation_id` metadata; `update_dvc_yaml_sha` is no-op in mock mode.
  - `test_dag_retrain.py`: validation-result branching, promote `put_file` calls (weights only, not annotation data).

### Success Criteria

#### Automated Verification:
- [x] `uv run pytest` passes (all DAG + provider tests).
- [x] DAG modules import without errors when `dags/` is on the Python path.
- [x] Tests run with no network/Drive/Zooniverse/HF/AWS access.
- [x] Lint passes: `uv run ruff check .`

#### Manual Verification:
- [ ] Coverage shows the new providers + DAG task functions exercised.

---

## Testing Strategy

- **Unit:** provider factory selection (mock vs real); each DAG task callable against mocks — consensus filtering, phase→division mapping (all 8 labels), `detection_error` flagging, annotation patching per split, confidence routing thresholds, validation-result branching, promotion `put_file` calls.
- **Integration:** `DagBag` import (no parse/import errors); `docker-compose up airflow` then a manual UI run of each DAG end-to-end against mock providers + MinIO.
- **Manual:** Airflow UI screenshots for Plan C's `docs/screenshots/airflow_dags.png`; flip one provider to real with creds to confirm the env switch path.

## Performance Considerations

- `run_dvc_repro` is the only heavy task — it runs full training on the local machine's GPU; Airflow just supervises it. No GPU is needed for any other task or for the tests.
- Airflow's own footprint is contained in its Docker service; the inference image (Plan B) stays slim and does not depend on Airflow.

## Migration Notes

- `update_dvc_yaml_sha` performs a real `git commit`/`push` in production; in the mock demo it operates on a throwaway branch or is a no-op (guard behind the hub mode) to avoid polluting history.
- The first promotion seeds `models/production/metrics.json`, consistent with Plan A's "first model auto-approves" behavior.
- Auto-labeled (`source: "auto"`) samples from `raw_image_ingest` are tagged so training can include/exclude them via `ExperimentConfig`; the **fixed test and validation splits never receive new Drive images** — `raw_image_ingest` always publishes to `"train"` only.
- Zooniverse enrichment (`apply_phase_to_annotations`) may update `division`/`phase` on existing test-set annotations (they were originally human-labeled, so correction is valid); it never adds new rows to any split.
- `division=None` (pending) annotations are excluded by `prepare_crops` at training time — they exist in HF but contribute no training signal until resolved by Zooniverse.
- S3/MinIO is strictly for model artifacts (weights, MLflow). No annotation or image data flows through S3 in this plan.

## References

- Architecture doc: [`thoughts/shared/research/allium_cepa_implementation_plan.md`](../research/allium_cepa_implementation_plan.md) (§5.5, §6, §8, §12, §13)
- Roadmap: [`2026-06-06-portfolio-roadmap.md`](2026-06-06-portfolio-roadmap.md)
- Plan A (providers base, factory, storage, gate, DVC stages): [`2026-06-06-plan-A-reproducibility-core.md`](2026-06-06-plan-A-reproducibility-core.md)
- Plan B (`ProductionConfig`, `ensure_production_weights`): [`2026-06-06-plan-B-serving-and-demo.md`](2026-06-06-plan-B-serving-and-demo.md)
- Inference engine for DAG B: [`src/allium_cepa_classifier/data_models/allium_cepa_model.py`](../../../src/allium_cepa_classifier/data_models/allium_cepa_model.py)
- SHA-pinned dataset stage: [`dvc.yaml`](../../../dvc.yaml) (`download_dataset`)
