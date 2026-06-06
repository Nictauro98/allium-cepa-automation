# Plan C — CI & README / Polish — Implementation Plan

> Milestone 3 of 4. See [`2026-06-06-portfolio-roadmap.md`](2026-06-06-portfolio-roadmap.md) for the full program. Depends on **Plan A** (tests + DVC stages exist) and **Plan B** (serving exists to document and screenshot). Can start once A+B are in place.

## Overview

Turn the working pipeline into a **portfolio artifact**: a GitHub Actions CI that runs lint + tests on every push (green badge), a **full rewrite** of the README aimed at AI/ML recruiters, and a `docs/screenshots/` set (MLflow runs, the live demo, Airflow once Plan D lands). After this plan: a recruiter landing on the repo sees a green CI badge, a live demo link, an architecture diagram, and a concise narrative of the engineering decisions — without reading any code.

This plan ships **no application logic**. It is CI config + documentation + images.

## Current State Analysis

Verified against the repo on branch `main` (2026-06-06):

- **No CI exists** — there is no `.github/` directory.
- **Toolchain is CI-ready:** `requires-python = "==3.12.*"`, ruff configured (`line-length = 100`, rules `E,W,F,I,B,C4,UP`), pytest configured (`testpaths=["tests"]`, `addopts = "--cov=src --cov-report=term-missing"`). Pre-commit pins `ruff-pre-commit` `v0.9.0` + standard hooks (incl. `check-added-large-files --maxkb=500`).
- **Tests will exist after Plan A/B** (`tests/` is created in Plan A). Before Plan A, `pytest` collects nothing — CI must run *after* A.
- **Package currently does not import on `main`** (Plan A Phase 0 fixes this). CI added before Plan A would be red; **C is correctly scheduled after A/B.**
- **README today** ([`README.md`](../../../README.md), ~141 lines) is a functional dev-setup doc: quick start, inference snippet, DVC notes. Accurate but **not recruiter-facing** — no demo link, no architecture diagram, no badges, no narrative of the ML-engineering decisions (calibration, the Delta-Method CI, continuous training, the validation gate).
- **No GPU on GitHub runners** and training exceeds free-tier time → **CI never trains**; it only lints + runs the CPU/network-free tests A/B wrote.

### Key Discoveries

- CI is a thin wrapper over commands the repo already defines (`uv run ruff check .`, `uv run pytest`) — the work is the workflow YAML + caching, not new test infrastructure.
- The `--maxkb=500` pre-commit guard means screenshots must be optimized PNGs (or kept under that limit / committed via the same LFS policy as weights) to avoid blocking commits.
- The README rewrite can pull its technical narrative directly from CLAUDE.md's "Key Design Decisions" (no-softmax, vector scaling, isotonic regression, Delta-Method CI) and the architecture doc.

## Desired End State

- Every push / PR triggers `.github/workflows/ci.yml`, which installs via `uv`, runs `ruff check`, `ruff format --check`, and `pytest` on Python 3.12 — green.
- The README leads with: project one-liner, **live demo badge/link** (HF Space from Plan B), a CI badge, an architecture diagram, and a "what this demonstrates" section for recruiters; setup/usage moved below the fold.
- `docs/screenshots/` contains optimized images referenced by the README (MLflow runs UI, the Streamlit demo; Airflow DAG graph added in Plan D).

## What We're NOT Doing (this plan)

- Training in CI (no GPU, too slow).
- Deploying anything (Plan B owns the HF Space; Plan D owns Airflow).
- Auto-publishing the package to PyPI.
- Coverage gating / required-status enforcement beyond a passing run (can be added later; not needed for the portfolio signal).
- Writing new tests (those belong to the plan that owns the code under test — A/B/D).

## Implementation Approach

CI first (so the badge is real before the README references it), then the README rewrite, then screenshots. README + screenshots are iterative and can be refined after Plan D adds the Airflow material.

---

## Phase 1: GitHub Actions CI

### Overview
A single workflow: checkout → install `uv` + Python 3.12 → `uv sync` → ruff lint + format-check → pytest. Cached for speed. No GPU, no training, no network-dependent tests.

### Changes Required

**File**: `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Set up Python
        run: uv python install 3.12

      - name: Sync (dev + serving, no GPU training stack needed for tests)
        run: uv sync --group serving --group dev

      - name: Ruff lint
        run: uv run ruff check .

      - name: Ruff format check
        run: uv run ruff format --check .

      - name: Tests
        run: uv run pytest
```

- `uv sync` group selection: install only what the **tests** need (dev + serving). If any test imports the full training stack, fall back to `uv sync --all-groups` — confirm against the actual `tests/` from Plan A/B.
- The tests A/B wrote are deliberately **network/Docker/GPU-free**, so they run on a stock `ubuntu-latest` runner.

### Success Criteria

#### Automated Verification:
- [ ] Workflow file is valid YAML: `uv run python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"`
- [ ] The same commands pass locally: `uv run ruff check . && uv run ruff format --check . && uv run pytest`
- [ ] A push to a branch triggers the workflow and it goes green (observed in the Actions tab).

#### Manual Verification:
- [ ] A PR shows the CI status check.
- [ ] The badge (Phase 2) resolves to the passing run.

**Implementation Note**: Pause for confirmation before proceeding (confirm the workflow is green on a real push before wiring the badge into the README).

---

## Phase 2: README rewrite (recruiter-facing)

### Overview
A full rewrite. The current dev-doc content is moved below a recruiter-oriented top section. **No existing prose is preserved** (per the locked decision) — the setup/DVC details are re-expressed concisely.

### Changes Required

**File**: `README.md`
**Changes**: New structure, top to bottom:

1. **Title + one-liner** — "Continuous-training, human-in-the-loop CV pipeline for mitotic-index estimation in *Allium cepa* microscopy."
2. **Badges** — CI (from Phase 1), Python 3.12, license, **live demo** (HF Space link from Plan B).
3. **Live demo** — direct link + one screenshot (`docs/screenshots/demo.png`).
4. **What this demonstrates** (the recruiter hook) — bullet list:
   - Two-stage detection→classification with **post-hoc calibration** (vector scaling + isotonic regression).
   - **Uncertainty quantification:** closed-form 95.45% CI on the mitotic index via the Delta Method (no bootstrap).
   - **Reproducible training:** DVC pipeline, HF-pinned dataset SHA, MLflow tracking + model registry.
   - **Automated quality gate:** a model is promoted to production only if it beats the baseline on Macro F1 without regressing per-class F1 or ECE.
   - **Continuous training / active learning:** low-confidence predictions routed to expert labeling (Zooniverse), validated data flows back into the dataset (Plan D).
   - **MLOps surface:** Dockerized serving (FastAPI + Streamlit), Airflow orchestration, $0 deployment.
5. **Architecture diagram** — reuse/adapt the ASCII diagram from the architecture doc, or an exported image in `docs/screenshots/architecture.png`.
6. **Tech stack** — table (from CLAUDE.md's stack section).
7. **Quick start** — condensed: `uv sync`, `dvc pull` weights, run inference / `docker-compose up`.
8. **Repo map** — short tree pointing at the meaningful entry points (`AlliumCepaModel`, `dvc.yaml`, `app/`, `airflow/`).
9. **Design decisions** — distilled from CLAUDE.md "Key Design Decisions".
10. **License / attribution** (UTN / INA).

### Success Criteria

#### Automated Verification:
- [ ] Markdown links resolve: a link-check (e.g. `uv run python` script walking relative links) reports no broken local paths.
- [ ] Referenced images exist in `docs/screenshots/`.

#### Manual Verification:
- [ ] Rendered on GitHub: badges resolve, demo link works, diagram and screenshots display.
- [ ] A non-expert reader understands what the project does and can find the live demo in <30s.

---

## Phase 3: Screenshots

### Overview
Capture and optimize the images the README references.

### Changes Required

**Directory**: `docs/screenshots/`
**Files**: `demo.png` (Streamlit result view), `mlflow_runs.png` (MLflow runs table / a run's metrics), `architecture.png` (optional rendered diagram). `airflow_dags.png` is added in **Plan D**.

- Optimize each PNG to stay under the pre-commit `--maxkb=500` guard (resize / `pngquant`), or document an LFS/exception path if a screenshot must exceed it.

### Success Criteria

#### Automated Verification:
- [ ] Each file in `docs/screenshots/` is < 500 KB (so `check-added-large-files` doesn't block the commit): `find docs/screenshots -size +500k` returns nothing.
- [ ] `uv run pre-commit run --all-files` passes with the screenshots staged.

#### Manual Verification:
- [ ] Screenshots are legible at the size GitHub renders them inline.
- [ ] They show *this* project's real UIs (not placeholders).

---

## Testing Strategy

- **Automated:** the workflow itself is the test — it must go green on a real push. Local parity: `ruff check`, `ruff format --check`, `pytest` all pass.
- **Manual:** README renders correctly on GitHub; all badges/links/images resolve; the recruiter-facing top section reads clearly.

## Performance Considerations

- `uv` caching (`enable-cache: true`) keeps CI under a couple of minutes. Installing only the groups the tests need (not the full CUDA training stack) avoids pulling heavy GPU wheels on the runner.

## Migration Notes

- The README's `dvc pull` weights instruction stays valid (the `weights` remote is unchanged by this plan).
- If Plan A renamed/added DVC remotes, reflect the current remote names in the Quick Start.
- The README's Airflow section + `airflow_dags.png` are stubbed here and completed in **Plan D**.

## References

- Architecture doc: [`thoughts/shared/research/allium_cepa_implementation_plan.md`](../research/allium_cepa_implementation_plan.md) (§5.8, §3)
- Roadmap: [`2026-06-06-portfolio-roadmap.md`](2026-06-06-portfolio-roadmap.md)
- Plan A (tests, DVC stages): [`2026-06-06-plan-A-reproducibility-core.md`](2026-06-06-plan-A-reproducibility-core.md)
- Plan B (serving to document + screenshot): [`2026-06-06-plan-B-serving-and-demo.md`](2026-06-06-plan-B-serving-and-demo.md)
- Design narrative source: `CLAUDE.md` "Key Design Decisions" + "Tech Stack"
- Current README (to be replaced): [`README.md`](../../../README.md)
