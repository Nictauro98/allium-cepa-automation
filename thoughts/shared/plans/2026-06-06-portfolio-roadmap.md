# Allium Cepa Portfolio — Implementation Roadmap (4 Milestone Plans)

## Why this document

The architecture doc ([`thoughts/shared/research/allium_cepa_implementation_plan.md`](../research/allium_cepa_implementation_plan.md)) describes **13 components**. Implementing them as one plan is unwieldy, and 13 micro-plans hide the dependency structure. This roadmap groups the work into **4 milestone plans**, each independently runnable, testable, and demoable — ordered by dependency and recruiter signal.

The end goal is a **professional portfolio repository**: a reproducible, observable, served, CI-gated, continuously-trained two-stage ML pipeline that an AI/ML recruiter can clone, run with one command, and see working end-to-end at $0 infrastructure cost.

## Decisions locked with the user (2026-06-06)

- **Structure:** 4 milestone plans, written and implemented one at a time.
- **External services — hybrid, with trivial mock→real swap.** Every external integration is built behind a small interface with a mock/local impl and a real impl, selected by a single config/env switch. Flipping any one service to real is a one-line config change — no code edits. This mirrors the architecture doc's MinIO↔S3 "change only `endpoint_url`" philosophy and generalizes it to every dependency.
- **First milestone:** Plan A (reproducibility core).
- **Airflow + ingestion:** full scope (Plan D), planned and implemented last.

## The cross-cutting convention: "providers"

Every external dependency follows the same shape so that mock→real is one line:

```
src/allium_cepa_classifier/providers/
├── base.py            # Protocol/ABC per capability (StorageProvider, LabelingProvider, ...)
├── storage.py         # FsspecStorage: local/MinIO ↔ real S3, differs only by endpoint_url
├── labeling.py        # MockZooniverse (fixtures) | RealZooniverse (panoptes-client)
├── drive.py           # MockDrive (local fixtures dir) | RealDrive (google-api-python-client)
├── dataset_hub.py     # MockHub (local dir) | RealHFHub (huggingface_hub)
└── factory.py         # get_storage(), get_labeling(), ... — each reads ONE env var to choose impl
```

Selection is driven by env vars (e.g. `ALLIUM_STORAGE=minio|s3`, `ALLIUM_LABELING=mock|zooniverse`), each defaulting to the **mock/local** impl. The factory is the only place that branches; all callers depend on the Protocol, never on a concrete class.

**Plan A introduces** `base.py` + `storage.py` + `factory.py` (storage only). **Later plans add** `labeling.py`, `drive.py`, `dataset_hub.py`.

## The four plans

| Plan | Name | External deps | Demoable result | Status |
|---|---|---|---|---|
| **A** | Reproducibility & tracking core | None (all local: MinIO + MLflow in Docker) | `dvc repro` runs `evaluate`→`validate_model`; MLflow UI shows runs; gate accepts/rejects | **Plan written** |
| **B** | Serving & demo | HF Spaces (real, free) | `docker-compose up` → Streamlit UI + FastAPI `/predict`; live HF Space | Not started |
| **C** | CI & README / polish | GitHub Actions (real, free) | Green CI badge; recruiter-facing README + screenshots | Not started |
| **D** | Ingestion & orchestration | Zooniverse / Drive (mock by default) | Airflow UI runs DAGs A/B/C against mock providers; flip to real via env | **Plan written** |

## Dependency order

```
A  (config patterns + storage provider + DVC stages + MLflow + gate)
└─> B  (serving reuses storage provider for prod weights; ProductionConfig)
    └─> C  (CI runs the tests A/B created; README documents A/B/D)

A ──────────────────────────────> D  (DAGs call evaluate/validate from A; providers from A's base)
```

- **C** can start once A+B exist.
- **D** depends only on A (the gate + storage providers) but is scheduled **last** because it is the heaviest and benefits from A/B being proven.

## Per-plan scope (detail lives in each plan's own file)

### Plan A — Reproducibility & tracking core → [`2026-06-06-plan-A-reproducibility-core.md`](2026-06-06-plan-A-reproducibility-core.md)
Finalize VAE/ControlNet cleanup; `ValidationConfig`; `providers/` base + storage; `.dvc/config` MinIO/S3 remotes; `docker-compose` (MinIO + MLflow); `evaluate` + `validate_model` DVC stages; MLflow logging in existing trainers/calibrators; first `tests/`.

### Plan B — Serving & demo → [`2026-06-06-plan-B-serving-and-demo.md`](2026-06-06-plan-B-serving-and-demo.md)
`Dockerfile` (inference) + `Dockerfile.train`; `app/api.py` (FastAPI thin wrapper); `app/streamlit_app.py`; `ProductionConfig`; weights pulled via storage provider at startup; `docker-compose` wiring of `api`+`streamlit`; deploy to HF Spaces.

### Plan C — CI & README / polish → [`2026-06-06-plan-C-ci-and-readme.md`](2026-06-06-plan-C-ci-and-readme.md)
`.github/workflows/ci.yml` (uv + ruff + pytest, no GPU); README rewrite for AI recruiters; `docs/screenshots/` (MLflow, Airflow, demo).

### Plan D — Ingestion & orchestration (full scope) → [`2026-06-06-plan-D-ingestion-and-orchestration.md`](2026-06-06-plan-D-ingestion-and-orchestration.md)
`providers/labeling.py`, `drive.py`, `dataset_hub.py` (mock + real each); `ZooniverseConfig`; `airflow/dags/` (`zooniverse_ingest`, `raw_image_ingest`, `retrain_pipeline`); tests against mocks.

## What we're NOT doing (whole program)

- Reintroducing VAE / ControlNet / diffusers (being removed; deps trimmed in Plan A).
- Paid infra (EC2 always-on, paid MLflow). Everything targets $0 free tiers + local.
- Wiring real Zooniverse / Drive / AWS credentials by default — those stay mock unless the env switch is set.

## References

- Architecture doc: [`thoughts/shared/research/allium_cepa_implementation_plan.md`](../research/allium_cepa_implementation_plan.md)
- Plan A: [`2026-06-06-plan-A-reproducibility-core.md`](2026-06-06-plan-A-reproducibility-core.md)
- Plan B: [`2026-06-06-plan-B-serving-and-demo.md`](2026-06-06-plan-B-serving-and-demo.md)
- Plan C: [`2026-06-06-plan-C-ci-and-readme.md`](2026-06-06-plan-C-ci-and-readme.md)
- Plan D: [`2026-06-06-plan-D-ingestion-and-orchestration.md`](2026-06-06-plan-D-ingestion-and-orchestration.md)
