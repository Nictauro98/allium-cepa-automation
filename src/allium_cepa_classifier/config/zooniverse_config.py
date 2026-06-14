from __future__ import annotations

from pydantic import ConfigDict

from .base_config import BaseConfig


class ZooniverseConfig(BaseConfig):
    """Zooniverse ingestion + consensus settings."""

    model_config = ConfigDict(frozen=True)

    project_id: str = ""
    consensus_threshold: float = 0.8
    min_new_for_patch: int = 10

    # Authoritative phase→division lookup used by zooniverse_ingest DAG.
    # None → do not update division (indeterminate / not_a_cell).
    phase_division_map: dict[str, int | None] = {
        "prophase": 1,
        "metaphase": 1,
        "anaphase": 1,
        "telophase": 1,
        "chromosomal_aberration": 1,
        "interphase": 0,
        "indeterminate": None,
        "not_a_cell": None,
    }
