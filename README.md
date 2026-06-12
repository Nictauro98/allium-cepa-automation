# Allium Cepa Automation

Continuous-training, human-in-the-loop CV pipeline for mitotic-index estimation in *Allium cepa* microscopy images.

[![CI](https://github.com/Nictauro98/allium-cepa-automation/actions/workflows/ci.yml/badge.svg)](https://github.com/Nictauro98/allium-cepa-automation/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-HF%20Spaces-orange)](https://huggingface.co/spaces/ntaurozzi/allium-cepa)

---

## Live Demo

Try the pipeline on your own microscopy image — no local setup required.

**[huggingface.co/spaces/ntaurozzi/allium-cepa](https://huggingface.co/spaces/ntaurozzi/allium-cepa)**

![Demo screenshot](docs/screenshots/demo.png)

Upload a full field-of-view micrograph and get back detected cells, a calibrated Mitotic Index (MI), and a 95.45% confidence interval — all within seconds.

---

## What This Demonstrates

| Capability | Implementation |
|---|---|
| **Two-stage CV pipeline** | YOLO detector → EfficientNet/ResNet/VGG classifier, both with post-hoc calibration |
| **Calibrated uncertainty** | Vector scaling (per-class temperature) on classifier logits; isotonic regression on detector confidence |
| **Delta Method CI** | Closed-form 95.45% CI on MI via error propagation — no bootstrap required |
| **Reproducible training** | DVC pipeline, HF-pinned dataset SHA, MLflow tracking + model registry |
| **Automated quality gate** | Model promoted to production only if it beats baseline on Macro F1, without regressing per-class F1 or ECE |
| **Active learning loop** | Low-confidence predictions routed to Zooniverse for expert labeling; validated data flows back into the dataset |
| **MLOps surface** | Dockerized FastAPI + Streamlit serving, Airflow orchestration, $0 deployment on HF Spaces |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                │
│  HuggingFace (ground truth)  Zooniverse (experts)  Drive (raw)       │
└──────────┬───────────────────────────┬──────────────────┬────────────┘
           │                           │                  │
           ▼                           ▼                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      AIRFLOW — ORCHESTRATION                         │
│  DAG A: zooniverse_ingest    DAG B: raw_image_ingest                 │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
           ┌───────────────────┼──────────────────────┐
           ▼                   ▼                      ▼
  HuggingFace             S3 / MinIO              S3 / MinIO
  GIAR-UTN/               dataset/labeled/        dataset/review/
  allium-cepa-dataset     (validated)             (pending experts)
  (curated ground truth)
           │
           └───────────────────┐
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 DVC PIPELINE — REPRODUCIBLE TRAINING                 │
│  download_dataset → coco_to_yolo → prepare_crops →                   │
│  train_detector → calibrate_detector →                               │
│  train_classifier → calibrate_classifier →                           │
│  evaluate → validate_model                                           │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│               MLFLOW — EXPERIMENT TRACKING & REGISTRY                │
│  Runs / Metrics / Model Registry (Staging → Production)              │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        SERVING — DOCKER                              │
│  FastAPI (AlliumCepaModel wrapper)    Streamlit (user UI)            │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
         High confidence              Low confidence
         → S3 ui_logs/ (candidates)   → Zooniverse (new labeling task)
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| **Training** | PyTorch ≥ 2.3, `timm` ≥ 1.0 (EfficientNet/ResNet/VGG), Ultralytics YOLO |
| **Calibration** | scikit-learn (isotonic regression), SciPy (L-BFGS-B for vector scaling) |
| **Tracking** | MLflow (runs, metrics, model registry), TensorBoard |
| **Reproducibility** | DVC (pipeline + remote storage), HuggingFace Hub (dataset pinning) |
| **Serving** | FastAPI, Streamlit, Docker, HuggingFace Spaces |
| **Orchestration** | Apache Airflow (continuous training + active learning) |
| **Dev** | Python 3.12, `uv`, Ruff, pytest |

---

## Quick Start

```bash
# Install dependencies (no GPU needed for inference)
uv sync --group serving --group dev

# Pull inference weights from HF
dvc pull src/allium_cepa_classifier/weights/*.dvc --remote weights

# Run inference
python -c "
from allium_cepa_classifier import AlliumCepaModel, AlliumCepaConfig
model = AlliumCepaModel(AlliumCepaConfig())
result = model.predict('path/to/image.png')
print(result.get_counts_with_ci())
"

# Or spin up the full serving stack locally
docker compose up
```

> **Note:** Datasets and experiment artifacts are not in git. Pull them with `dvc repro` (see [Dataset Version Management](#dataset-version-management) below).

---

## Repo Map

```
├── src/allium_cepa_classifier/
│   ├── data_models/allium_cepa_model.py   ← AlliumCepaModel (inference entry point)
│   ├── statistics/error_propagation.py    ← Delta Method CI
│   ├── training/                          ← trainer, calibrators, model_builder
│   └── config/                            ← Pydantic config models
├── app/                                   ← FastAPI + Streamlit serving
├── dvc.yaml                               ← Full training pipeline definition
├── experiments/                           ← Per-experiment configs + run artifacts
├── airflow/dags/                          ← Zooniverse + raw-image ingestion DAGs
└── tests/                                 ← Unit + integration tests (37 tests, no GPU)
```

---

## Design Decisions

**No softmax in the model** — `build_model()` returns raw logits so calibration operates directly on the pre-softmax space. Softmax is applied only at inference time.

**Vector scaling over scalar temperature** — Per-class temperature vector `[T_neg, T_pos]` allows asymmetric calibration; a single scalar would force the same correction on both classes.

**Isotonic regression for detector calibration** — Makes no assumptions about the calibration curve shape and handles non-monotonic confidence distributions from YOLO.

**Delta Method for MI confidence intervals** — `compute_mi_with_ci()` propagates per-detection Bernoulli uncertainty through both stages via the Delta Method, giving a closed-form 95.45% CI (MI ± 2σ) without bootstrap. The covariance term `cov(N_mit, N_cel) = var_mit` follows from the independence assumption on the same set of detections contributing to both sums.

**Automated promotion gate** — a newly trained model is only written to production if it: (1) beats the registered baseline on Macro F1, (2) does not regress per-class F1 for either class, and (3) does not regress ECE. This prevents silent quality degradation during continuous training.

---

## Dataset Version Management

The dataset is pinned to a specific HuggingFace commit SHA in `dvc.yaml`. To bump it:

```bash
python -c "from huggingface_hub import repo_info; print(repo_info('GIAR-UTN/allium-cepa-dataset', repo_type='dataset').sha)"
# Edit dvc.yaml: change --rev <old-sha> → --rev <new-sha>
dvc repro download_dataset
git add dvc.yaml dvc.lock && git commit -m "chore: bump HF dataset to <new-sha>"
```

To push updated inference weights to HF after retraining:

```bash
uv run python scripts/utils/push_weights_to_hf.py
```

---

## Training

```bash
# Single classifier experiment (calibration runs automatically after)
uv run python scripts/train_classifier.py --config experiments/binary_classifier/efficientnet_b1/config.yaml

# Sweep all classifier configs
uv run python scripts/sweep.py --configs experiments/binary_classifier/*/config.yaml

# YOLO detector
uv run python scripts/train_detector.py --config experiments/yolo/yolo11n_200e/config.yaml

# Re-run calibration only
uv run python scripts/calibrate_classifier.py --experiment experiments/binary_classifier/efficientnet_b1/<run-dir>
uv run python scripts/calibrate_detector.py   --experiment experiments/yolo/yolo11n_200e/<run-dir>
```

> The experiment configs in `experiments/` are set to **3 epochs** — enough for an end-to-end pipeline check. Increase `epochs` in the config for a real training run.

---

## MLflow Experiment Tracking

![MLflow runs](docs/screenshots/mlflow_runs.png)

Each training run logs metrics (accuracy, ECE, per-class F1), calibration curves, and confusion matrices. Promoted models are registered in the MLflow Model Registry under the `production` alias.

---

## License

MIT — developed at **UTN / INA** (Universidad Tecnológica Nacional / Instituto Nacional del Agua), Argentina.
