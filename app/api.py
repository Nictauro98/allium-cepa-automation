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
async def predict(file: UploadFile = File(...)) -> dict:  # noqa: B008
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    result = _state["model"].predict(tmp_path)
    # Merge raw integer counts (total_cells, mitotic_cells, …) with the
    # calibrated CI dict (mi, ci_lower, ci_upper, sigma_mi, …) so the UI
    # has both in one object.
    counts = {**result.get_counts(), **result.get_counts_with_ci()}
    return {
        "counts": counts,
        # round-trip through to_json so numpy/bool dtypes serialize cleanly
        "detections": json.loads(result.detections.to_json(orient="records")),
    }
