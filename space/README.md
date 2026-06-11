---
title: Allium Cepa Mitosis Detector
emoji: 🔬
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Allium Cepa — Cell Detection & Mitotic Index

Computer vision demo for automated cell detection and mitosis classification in *Allium cepa* (onion root tip) microscopy images, developed at UTN/INA.

Upload a full-FOV micrograph to get:
- Detected cells with bounding boxes
- Calibrated Mitotic Index (MI) with 95.45% confidence interval
- Per-cell classification table and CSV export

## Pipeline

Two-stage inference:
1. **YOLO** detects individual cells, calibrated with isotonic regression.
2. **EfficientNet** classifies each crop as *mitosis* / *no mitosis*, calibrated with vector scaling.

## Space secrets required

Set these in the Space settings before the first build:

| Secret | Description |
|---|---|
| `ALLIUM_STORAGE` | `s3` |
| `AWS_ACCESS_KEY_ID` | Read-only AWS key for the model bucket |
| `AWS_SECRET_ACCESS_KEY` | Corresponding secret |
| `ALLIUM_BUCKET` | Bucket name (e.g. `allium-cepa-ml`) |

Weights are pulled from `models/production/` in the bucket at startup and cached for the lifetime of the container.
