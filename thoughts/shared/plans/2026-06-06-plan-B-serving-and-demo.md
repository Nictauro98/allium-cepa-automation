# Plan B — Serving & Demo — Implementation Plan

> Milestone 2 of 4. See [`2026-06-06-portfolio-roadmap.md`](2026-06-06-portfolio-roadmap.md) for the full program. Depends on **Plan A** (the `providers/` storage abstraction, MinIO/MLflow infra, and `models/production/` layout).

## Overview

Wrap the existing `AlliumCepaModel` in a thin **FastAPI** service and a non-technical **Streamlit** UI, package both with Docker, and deploy a live, $0 demo to **Hugging Face Spaces**. After this plan: `docker-compose up` brings up the full local stack (MinIO + MLflow from Plan A, plus `api` + `streamlit`); a researcher uploads a micrograph in Streamlit and gets the mitotic index with a confidence interval; the same UI is publicly reachable on an HF Space.

The serving layer adds **no inference logic** — all of it already lives in `AlliumCepaModel.predict()`. The new work is: (1) a `ProductionConfig` that points weight paths at a local cache, (2) startup code that pulls the three production weight files via Plan A's storage provider, (3) the FastAPI/Streamlit/Docker wrappers, and (4) deployment.

**A working Streamlit UI already exists at [`src/ui/app.py`](../../../src/ui/app.py)** (untracked, hand-crafted) with rich features — annotation modes (off / all / mitosis-only / not-mitosis), a confidence-threshold slider, display-width control, a detections table, CSV download, and a side-by-side metrics layout. Plan B **adapts this existing UI** rather than writing one from scratch: it is repointed from the current in-process model call to the FastAPI service over HTTP, and its uncalibrated mitotic index is replaced by the calibrated `get_counts_with_ci()` (MI ± 95.45% CI) as the headline. To keep its table / annotation / CSV features, the API's `/predict` is expanded to return the full per-detection records alongside the counts.

## Current State Analysis

Verified against the repo on branch `main` (2026-06-06):

- **Inference is complete and self-contained.** `AlliumCepaModel(AlliumCepaConfig)` ([`data_models/allium_cepa_model.py:51`](../../../src/allium_cepa_classifier/data_models/allium_cepa_model.py#L51)) loads three weight files in `__init__` and exposes `predict(path)` → `AlliumCepaResult` with `get_counts()`, `get_counts_with_ci()`, `show_annotated()` (returns a PIL image), and `save_csv()`.
- **`AlliumCepaConfig`** ([`config/allium_cepa_config.py`](../../../src/allium_cepa_classifier/config/allium_cepa_config.py)) hard-codes three weight paths under `src/allium_cepa_classifier/weights/` via `find_project_root()`, plus `image_size`, `batch_size`, `use_cpu`. It already accepts overridden paths (it's a `BaseConfig`), so serving can point it at a cache dir without code changes.
- **Weight loading raises** `FileNotFoundError` if the detector/classifier weights are missing ([`allium_cepa_model.py:69`](../../../src/allium_cepa_classifier/data_models/allium_cepa_model.py#L69)); the isotonic calibrator merely **warns** if absent. So the startup pull must guarantee at least the two `.pt` files.
- **No `app/`, no Dockerfiles, no serving deps.** FastAPI/uvicorn/python-multipart/requests are not yet dependencies (`streamlit`, `pandas`, `pillow` are already present and used by the existing UI).
- **An existing Streamlit UI lives at `src/ui/app.py` + `src/ui/cli.py`** (untracked). It is feature-rich but **does not match the current implementation** and is in fact broken against it:
  - Calls `model.predict(image)` with a **PIL Image**, but [`predict()`](../../../src/allium_cepa_classifier/data_models/allium_cepa_model.py#L323) requires a **file path** (`Path(image_path).exists()`) → would crash. Needs an upload→temp-file step.
  - Loads `AlliumCepaConfig.from_yaml("config.yaml")` — no such file exists at repo root.
  - Imports via `from src.allium_cepa_classifier import …` (wrong; should be `from allium_cepa_classifier import …`).
  - Computes an **uncalibrated** mitotic index (`mitotic / total * 100`) and shows **no CI** — bypassing `get_counts_with_ci()`.
  - Loads the model **in-process** via `@st.cache_resource`, contradicting the approved Streamlit→FastAPI split.
- **Plan A delivers what B consumes:** `providers/factory.get_storage()` (the `ALLIUM_STORAGE=minio|s3` switch), the `models/production/` key layout in the bucket (`object_detection.pt`, `classifier_calibrated.pt`, `yolo_isotonic_calibrator.pkl`, `metrics.json`), and a `docker-compose.yml` to extend.

### Key Discoveries

- `get_counts_with_ci()` returns `{mitotic_index, ci_lower, ci_upper, sigma_mi, total_cells, mitotic_cells, ...}` ([`allium_cepa_result.py:167`](../../../src/allium_cepa_classifier/data_models/allium_cepa_result.py#L167)) — the headline serialization. CI requires the `p_hat`/`q_interphase`/`q_mitosis` columns produced by the calibrated pipeline.
- `AlliumCepaResult` exposes the full per-detection DataFrame as `result.detections` ([`allium_cepa_result.py:39`](../../../src/allium_cepa_classifier/data_models/allium_cepa_result.py#L39)) with columns `x_min,y_min,x_max,y_max,confidence,p_hat,class_id,class_name,image,mitosis,mitosis_score,q_interphase,q_mitosis`. The existing UI's table / annotation-drawing / CSV features operate on this DataFrame — so the API must return it for those features to survive over HTTP.
- The existing UI already has reusable helpers — `draw_annotated()`, `resize_image_and_detections()`, `compute_summary()` — that work on a detections DataFrame and can be kept as-is once the DataFrame is reconstructed from the API's JSON records.
- The storage provider's `get_file(key, local_path)` (Plan A) is exactly the primitive needed for startup weight pulls; B adds no new storage code.

## Desired End State

- `docker-compose up` brings up `minio`, `mlflow` (Plan A) + `api` (`:8000`) + `streamlit` (`:8501`).
- `curl -F "file=@sample.png" localhost:8000/predict` returns JSON with a `counts` object (`mitotic_index`, `ci_lower`, `ci_upper`, totals) **and** a `detections` array of per-cell records.
- Streamlit at `localhost:8501` (the adapted `src/ui/app.py`) accepts an upload, calls the API over HTTP, shows the calibrated **MI ± CI** as the headline, and retains its annotation modes / confidence slider / detections table / CSV download driven by the returned `detections`.
- On a fresh container with an empty weights cache and `ALLIUM_STORAGE=minio`, the API pulls the three weight files from `models/production/` at startup and serves successfully.
- A public HF Space runs the same image and is reachable from a browser.

## What We're NOT Doing (this plan)

- CI workflow / README rewrite / screenshots (**Plan C**).
- Airflow, Zooniverse/Drive ingestion, the labeling/drive/hub providers (**Plan D**).
- Logging UI predictions to `ui_logs/` for active learning — that consumer lives in **Plan D**; B only serves. *(If trivial, a single `put_file` log line may be included behind a flag — see Phase 2 note.)*
- Authentication / rate limiting (out of scope for a portfolio demo).
- Changing inference behavior or model code.
- Keeping the existing in-process model load in the UI — it's replaced by the HTTP call (the model lives only in the API process).

## Implementation Approach

Inside-out: add `ProductionConfig` + the weight-pull helper (reusing Plan A's storage provider), then the FastAPI app, then Streamlit, then Docker packaging + compose wiring, then HF Spaces deploy. Each phase is independently runnable; the API works locally before any container exists.

---

## Phase 1: ProductionConfig + startup weight loader

### Overview
A config that points `AlliumCepaConfig`'s weight paths at a local cache directory, plus a helper that ensures those files exist by pulling them from `models/production/` via the storage provider.

### Changes Required

#### 1. ProductionConfig
**File**: `src/allium_cepa_classifier/config/production_config.py`

```python
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

    # active-learning routing thresholds (consumed by Plan D; defined here per the doc)
    high_confidence_threshold: float = 0.90
```
Register in `config/__init__.py`.

#### 2. Weight loader
**File**: `src/allium_cepa_classifier/serving/weights.py`

```python
from pathlib import Path

from allium_cepa_classifier.config import AlliumCepaConfig, ProductionConfig
from allium_cepa_classifier.providers.factory import get_storage


def ensure_production_weights(cfg: ProductionConfig) -> AlliumCepaConfig:
    """Pull prod weights from object storage into the local cache, return an inference config."""
    storage = get_storage()
    cfg.weights_cache_dir.mkdir(parents=True, exist_ok=True)

    det = cfg.weights_cache_dir / "object_detection.pt"
    clf = cfg.weights_cache_dir / "classifier_calibrated.pt"
    cal = cfg.weights_cache_dir / "yolo_isotonic_calibrator.pkl"

    for key, dest in [
        (cfg.detection_key, det),
        (cfg.classifier_key, clf),
        (cfg.calibrator_key, cal),
    ]:
        if not dest.exists():
            storage.get_file(key, dest)

    return AlliumCepaConfig(
        detection_weights_path=det,
        classification_weights_path=clf,
        detection_calibrator_path=cal,
        use_cpu=True,  # HF Spaces free tier has no GPU
    )
```

### Success Criteria

#### Automated Verification:
- [x] `uv run python -c "from allium_cepa_classifier.config import ProductionConfig; ProductionConfig()"`
- [x] `uv run python -c "from allium_cepa_classifier.serving.weights import ensure_production_weights"`
- [x] Lint passes: `uv run ruff check .`

#### Manual Verification:
- [x] With MinIO up and the three files present under `models/production/`, `ensure_production_weights(ProductionConfig())` populates the cache dir and returns a valid `AlliumCepaConfig`.

---

## Phase 2: FastAPI service (thin wrapper)

### Overview
A `/predict` endpoint that accepts an uploaded image, runs `AlliumCepaModel.predict()`, and returns **both** `get_counts_with_ci()` (the headline MI ± CI + totals) **and** the full per-detection records, so the Streamlit UI's table / annotation / CSV features work client-side. Model is built **once** at startup.

### Changes Required

#### 1. API app
**File**: `app/api.py`

```python
import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile

from allium_cepa_classifier import AlliumCepaModel
from allium_cepa_classifier.config import ProductionConfig
from allium_cepa_classifier.serving.weights import ensure_production_weights

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    inference_cfg = ensure_production_weights(ProductionConfig())
    _state["model"] = AlliumCepaModel(inference_cfg)
    yield
    _state.clear()


app = FastAPI(title="Allium Cepa — Mitotic Index API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in _state}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    result = _state["model"].predict(tmp_path)
    return {
        "counts": result.get_counts_with_ci(),                       # MI ± CI + totals
        # round-trip through to_json so numpy/bool dtypes serialize cleanly
        "detections": json.loads(result.detections.to_json(orient="records")),
    }
```

- The model is constructed in `lifespan` startup so the weight pull + load happens once.
- `/predict` returns `{counts, detections}`. `counts` is the calibrated headline; `detections` is the full DataFrame as records (columns `x_min..y_max`, `confidence`, `mitosis`, `class_name`, …) that powers the UI's table, annotation drawing, and CSV — without the UI ever importing the model.
- `result.detections.to_json(orient="records")` avoids `numpy`/`bool` JSON-serialization issues that a raw `to_dict()` + default encoder can hit.
- `/health` lets Docker/compose healthchecks and HF Spaces probe readiness.
- **`ui_logs/` note:** logging each upload to `ui_logs/<timestamp>.png` via `get_storage().put_file` is a one-liner that could be added here behind `PRODUCTION_LOG_UPLOADS=1`, but the *consumer* of those logs is Plan D — keep it off by default and out of scope unless trivially added.

#### 2. Dependencies
**File**: `pyproject.toml` — add a `serving` dependency group: `fastapi`, `uvicorn[standard]`, `python-multipart`, `requests`. (`streamlit`, `pandas`, `pillow` are already dependencies used by the existing UI.)

### Success Criteria

#### Automated Verification:
- [x] `uv run python -c "import app.api"` imports without building the model (lifespan not triggered)
- [ ] `uv run pytest tests/test_api.py` passes (Phase 5) — `/predict` with a fixture image, model mocked; asserts both `counts` and `detections` keys
- [x] Lint passes: `uv run ruff check .`

#### Manual Verification:
- [ ] `uv run uvicorn app.api:app` then `curl -F "file=@<sample>.png" localhost:8000/predict` returns JSON with `counts.mitotic_index`/`ci_lower`/`ci_upper` and a non-empty `detections` array.
- [ ] `GET /health` returns `model_loaded: true` after startup.

**Implementation Note**: Pause for confirmation before proceeding.

---

## Phase 3: Streamlit UI

### Overview
**Adapt the existing `src/ui/app.py`** (don't rewrite it). Keep its rich UI — sidebar options, annotation modes, confidence slider, display-width control, detections table, CSV download, side-by-side layout — but change *where the data comes from*: POST the upload to the API and reconstruct the detections DataFrame from the JSON, instead of loading the model in-process. Make the calibrated MI ± CI from `counts` the headline metric.

### Changes Required

#### 1. Repoint the UI from in-process model → API
**File**: `src/ui/app.py` (adapt the existing file)
**Changes**:

- **Delete** the in-process model path: remove `from src.allium_cepa_classifier import …`, `@st.cache_resource load_model()`, and the `AlliumCepaConfig.from_yaml("config.yaml")` call.
- **Add** an API client that returns a `(counts, detections_df)` pair:

  ```python
  import os
  import pandas as pd
  import requests

  API_URL = os.getenv("API_URL", "http://localhost:8000")

  def run_prediction(uploaded) -> tuple[dict, pd.DataFrame]:
      resp = requests.post(
          f"{API_URL}/predict",
          files={"file": (uploaded.name, uploaded.getvalue())},
          timeout=120,
      )
      resp.raise_for_status()
      payload = resp.json()
      return payload["counts"], pd.DataFrame(payload["detections"])
  ```

- **Replace** `result = model.predict(image)` / `result.detections` with `counts, detections = run_prediction(uploaded)`. Wrap the call in `try/except` → `st.error(...)` so a down/erroring API shows a readable message, not a stack trace.
- **Headline metric:** replace `compute_summary()`'s uncalibrated `mitotic_index` with the calibrated values from `counts`:

  ```python
  st.metric("Mitotic index", f"{counts['mitotic_index']:.4f}")
  st.caption(f"95.45% CI: [{counts['ci_lower']:.4f}, {counts['ci_upper']:.4f}]")
  a, b, c = st.columns(3)
  a.metric("Cells", counts["total_cells"])
  b.metric("Mitotic", counts["mitotic_cells"])
  ```

  Keep `compute_summary()` only for the *post-confidence-filter* tallies shown beside the table, clearly distinguished from the calibrated headline (see note below).
- **Keep as-is:** `draw_annotated()`, `resize_image_and_detections()`, the sidebar widgets, the annotation-mode mapping, the detections table, and the CSV `st.download_button` — they all operate on the `detections` DataFrame, which now comes from the API.
- **Confidence-filter nuance:** the calibrated MI ± CI in `counts` is computed server-side over **all** detections. The sidebar confidence slider only filters what is *drawn/listed/exported* client-side; it does **not** change the headline MI/CI. State this in a `st.caption` so the two numbers aren't read as inconsistent.

#### 2. Update the launcher
**File**: `src/ui/cli.py`
**Changes**: keep the `streamlit run <app.py>` subprocess launcher; drop the stale "Poetry script" wording. Optionally expose it as a console-script entry point in `pyproject.toml` (e.g. `allium-ui = "ui.cli:run_streamlit"`).

> **Location note:** the UI stays at `src/ui/` (where it already lives); the API stays at `app/api.py`. Phase 4's Docker/compose references use `src/ui/app.py` for the Streamlit command accordingly.

### Success Criteria

#### Automated Verification:
- [x] Syntax check: `uv run python -c "import ast; ast.parse(open('src/ui/app.py').read())"`
- [x] No in-process model import remains: `grep -n "AlliumCepaModel\|from_yaml\|cache_resource" src/ui/app.py` returns nothing
- [x] Lint passes: `uv run ruff check .`

#### Manual Verification:
- [ ] With the API running, `uv run streamlit run src/ui/app.py`, upload a sample → calibrated MI + CI headline renders; annotation modes, confidence slider, table, and CSV download all still work off the API's detections.
- [ ] API down → the UI shows a readable error, not a stack trace.

---

## Phase 4: Docker packaging + compose wiring

### Overview
Two images (lightweight inference vs full training) and compose services for `api` + `streamlit`, extending Plan A's `docker-compose.yml`.

### Changes Required

#### 1. Inference image
**File**: `Dockerfile`
**Changes**: `uv`-based, `uv sync --no-dev --group serving` (runtime + serving deps only, no training stack). Copies `src/` (includes both `allium_cepa_classifier` and `ui`) and `app/`. Default `CMD` runs uvicorn: `uvicorn app.api:app --host 0.0.0.0 --port 8000`.

#### 2. Training image
**File**: `Dockerfile.train`
**Changes**: `uv sync --all-groups` (full training stack incl. CUDA wheels). Used by Plan D / local training in containers.

#### 3. Compose services
**File**: `docker-compose.yml` (extend Plan A's)
**Changes**: Add
- `api`: builds `Dockerfile`, depends_on `minio` (healthy), env `ALLIUM_STORAGE=minio`, `MINIO_ENDPOINT`, MinIO creds; ports `8000:8000`; healthcheck on `/health`.
- `streamlit`: builds `Dockerfile` with `CMD streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0`, env `API_URL=http://api:8000`, depends_on `api`; ports `8501:8501`.

### Success Criteria

#### Automated Verification:
- [ ] `docker-compose config` validates
- [ ] `docker build -f Dockerfile .` succeeds
- [ ] `docker-compose up -d` → `curl localhost:8000/health` returns `model_loaded: true`
- [ ] `curl localhost:8501` returns the Streamlit page

#### Manual Verification:
- [ ] Full round trip in the browser at `:8501` against the dockerized API.
- [ ] Killing the weights cache and restarting `api` re-pulls weights from MinIO.

**Implementation Note**: Pause for confirmation before proceeding.

---

## Phase 5: Tests

### Changes Required

**Files**: `tests/test_api.py`, `tests/test_production_weights.py`, `tests/test_ui_client.py`

- `test_api.py`: FastAPI `TestClient`; monkeypatch `_state["model"]` with a stub whose `predict()` returns a fake result (`get_counts_with_ci()` → fixed dict; `.detections` → a tiny DataFrame) → assert `/predict` returns both `counts` and `detections` (and that `detections` round-trips cleanly to JSON), and `/health` reflects state. No real model load.
- `test_production_weights.py`: monkeypatch `get_storage()` with a fake that writes dummy files; assert `ensure_production_weights` populates the cache and returns an `AlliumCepaConfig` with the cache paths. No network.
- `test_ui_client.py`: monkeypatch `requests.post` to return a canned `{counts, detections}` payload; assert `run_prediction()` returns `(counts_dict, DataFrame)` with the expected columns. No network, no Streamlit runtime.

### Success Criteria

#### Automated Verification:
- [ ] `uv run pytest tests/test_api.py tests/test_production_weights.py tests/test_ui_client.py` passes
- [ ] Tests run with no Docker/MinIO/network/GPU dependency
- [ ] Lint passes: `uv run ruff check .`

#### Manual Verification:
- [ ] Coverage shows `serving/`, `app/api.py`, and `src/ui/app.py`'s client exercised.

---

## Phase 6: Deploy to Hugging Face Spaces

### Overview
Deploy a **single** Space (Docker SDK) running the inference image, exposing Streamlit, with the API started in the same container. $0, always-on (subject to free-tier sleep).

### Changes Required

- **HF Space (Docker SDK):** a `README.md` Space header + a Space-specific entrypoint that launches uvicorn (background) **and** Streamlit (`src/ui/app.py`, foreground), with `API_URL=http://localhost:8000` inside the container. Single combined Space per the locked decision.
- **Weights source for the Space:** set `ALLIUM_STORAGE=s3` + AWS env (HF Space secrets) to pull from real S3, **or** bake a lightweight `mock`/local provider that reads weights committed to the Space via Git LFS / HF model repo. Default for the public demo: pull from the real S3 `models/production/` using read-only credentials stored as Space secrets.
- **Space secrets:** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `ALLIUM_STORAGE=s3`, `ALLIUM_BUCKET`.

### Success Criteria

#### Automated Verification:
- [ ] The Space's Docker image is the same `Dockerfile` (or a thin Space wrapper over it) — `docker build` succeeds locally.

#### Manual Verification:
- [ ] Public Space URL loads the Streamlit UI.
- [ ] Uploading a sample micrograph returns MI + CI in the browser.
- [ ] Cold start (Space wakes from sleep) completes the weight pull and serves within the free-tier timeout.

---

## Testing Strategy

- **Unit:** API serialization with a mocked model; weight-loader cache population with a fake storage provider.
- **Integration:** `docker-compose up` full round trip (upload → MI/CI) against dockerized API + MinIO weights.
- **Manual:** browser round trip locally and on the live HF Space.

## Performance Considerations

- HF Spaces free tier is **CPU-only** → `use_cpu=True` in the serving `AlliumCepaConfig`. Inference is one image at a time; latency dominated by YOLO + the classifier batch, acceptable for a demo.
- Weights are pulled **once** at startup and cached on the container filesystem; subsequent requests hit the in-memory model.
- Cold starts on HF Spaces include the weight pull — keep the three files small (they already are) so wake-from-sleep stays within timeout.

## Migration Notes

- The inference image deliberately excludes the training stack (`diffusers`/training-only deps removed in Plan A keep it small).
- `ProductionConfig.weights_cache_dir` defaults to `/tmp` so it works on the ephemeral HF Spaces filesystem; mount a volume in compose if persistence across local restarts is wanted.

## References

- Architecture doc: [`thoughts/shared/research/allium_cepa_implementation_plan.md`](../research/allium_cepa_implementation_plan.md) (§5.6, §5.7, §9, §10)
- Roadmap: [`2026-06-06-portfolio-roadmap.md`](2026-06-06-portfolio-roadmap.md)
- Plan A (storage provider, prod weight layout, compose base): [`2026-06-06-plan-A-reproducibility-core.md`](2026-06-06-plan-A-reproducibility-core.md)
- Inference model + result API: [`src/allium_cepa_classifier/data_models/allium_cepa_model.py`](../../../src/allium_cepa_classifier/data_models/allium_cepa_model.py)
- Result API (`detections`, `get_counts_with_ci`): [`src/allium_cepa_classifier/data_models/allium_cepa_result.py`](../../../src/allium_cepa_classifier/data_models/allium_cepa_result.py)
- Inference config: [`src/allium_cepa_classifier/config/allium_cepa_config.py`](../../../src/allium_cepa_classifier/config/allium_cepa_config.py)
- Existing Streamlit UI to adapt: [`src/ui/app.py`](../../../src/ui/app.py), [`src/ui/cli.py`](../../../src/ui/cli.py)
