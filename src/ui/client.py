from __future__ import annotations

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
