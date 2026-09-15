"""The one place that answers "which model is in production right now".

Before this module the question was implemented three times -- in predict, in
the season simulator and in the dashboard -- with subtly different fallback
rules. They agreed by coincidence rather than by construction.

Resolution order, unchanged from the previous behaviour:

1. ``model_metrics.json`` if it names an algorithm the codebase can build, then
   the newest ``model_runs`` row for that algorithm that has an artifact.
2. Otherwise the newest ``model_runs`` row with an artifact.

The artifact path always comes from the ``model_runs`` row, never from a name
guessed on disk. That is what makes an old run reproducible after training has
written a newer artifact.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from football_pipeline.config import ROOT
from football_pipeline.constants import PRODUCTION_ALGORITHMS
from football_pipeline.competitions import DEFAULT_COMPETITION, processed_dir
from football_pipeline.db import connect

logger = logging.getLogger(__name__)

REPORT_PATH = ROOT / "data" / "processed" / "model_metrics.json"
MODEL_DIR = ROOT / "data" / "processed" / "models"


def report_path_for(competition: str = DEFAULT_COMPETITION) -> Path:
    return processed_dir(competition, root=ROOT) / "model_metrics.json"


def _competition_id(code: str) -> int | None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM competitions WHERE code = %s", (code,))
            row = cur.fetchone()
    return int(row[0]) if row else None

# Report keys carried for backward compatibility, newest first.
SELECTION_KEYS = (
    "selected_by_walkforward_log_loss",
    "selected_by_valid_log_loss",
    "selected_sklearn_by_valid_log_loss",
)


class ProductionModelUnavailable(RuntimeError):
    """No trained production model with a usable artifact could be resolved."""


@dataclass(frozen=True)
class ProductionModel:
    """The canonical production model, resolved once and passed around."""

    algorithm: str
    model_run_id: int
    artifact_path: str | None
    feature_version: str
    trained_at: Any = None

    @property
    def artifact(self) -> Path | None:
        return Path(self.artifact_path) if self.artifact_path else None


def known_algorithms() -> frozenset[str]:
    """Algorithm names this codebase can actually construct and score with.

    Read from constants, not from models, so resolving the production model
    never drags scikit-learn into the dashboard container.
    """
    return PRODUCTION_ALGORITHMS


def reported_algorithm(
    report_path: Path | None = None,
    *,
    competition: str = DEFAULT_COMPETITION,
) -> str | None:
    """Algorithm named by the training report, if it is one we can build."""
    path = report_path or report_path_for(competition)
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Ignoring unreadable %s: %s", path, exc)
        return None
    buildable = known_algorithms()
    for key in SELECTION_KEYS:
        name = report.get(key)
        if name and str(name) in buildable:
            return str(name)
    return None


def _fetch_run(algorithm: str | None, *, competition: str = DEFAULT_COMPETITION) -> tuple | None:
    """Newest model_runs row with an artifact for this competition."""
    competition_id = _competition_id(competition)
    with connect() as conn:
        with conn.cursor() as cur:
            if competition_id is None:
                return None
            if algorithm:
                cur.execute(
                    """
                    SELECT id, algorithm, artifact_path, feature_version, trained_at
                    FROM model_runs
                    WHERE algorithm = %s
                      AND competition_id = %s
                      AND artifact_path IS NOT NULL
                    ORDER BY trained_at DESC, id DESC
                    LIMIT 1
                    """,
                    (algorithm, competition_id),
                )
            else:
                cur.execute(
                    """
                    SELECT id, algorithm, artifact_path, feature_version, trained_at
                    FROM model_runs
                    WHERE competition_id = %s AND artifact_path IS NOT NULL
                    ORDER BY trained_at DESC, id DESC
                    LIMIT 1
                    """,
                    (competition_id,),
                )
            return cur.fetchone()


def production_model(*, competition: str = DEFAULT_COMPETITION) -> ProductionModel:
    """Resolve the canonical production model for one competition, or raise."""
    preferred = reported_algorithm(competition=competition)
    row = _fetch_run(preferred, competition=competition)
    if row is None and preferred is not None:
        logger.warning(
            "Report names %s for %s but no model_runs row has an artifact for it; "
            "falling back to the newest trained run in that league.",
            preferred,
            competition,
        )
        row = _fetch_run(None, competition=competition)
    if row is None:
        raise ProductionModelUnavailable(
            f"No trained production model for {competition}. "
            "Run python -m football_pipeline.train first."
        )
    run_id, algorithm, artifact_path, feature_version, trained_at = row
    return ProductionModel(
        algorithm=algorithm,
        model_run_id=int(run_id),
        artifact_path=artifact_path,
        feature_version=feature_version,
        trained_at=trained_at,
    )


def selected_algorithm(*, competition: str = DEFAULT_COMPETITION) -> str:
    """Name of the production algorithm for one competition."""
    return production_model(competition=competition).algorithm


def load_estimator(model: ProductionModel | None = None, *, competition: str = DEFAULT_COMPETITION):
    """Load the fitted estimator for the production run.

    Reads the path recorded on the run itself. Falls back to the legacy
    algorithm-named artifact only when the recorded path is missing from disk,
    so runs trained before artifacts were versioned keep working.
    """
    import joblib

    model = model or production_model(competition=competition)
    candidates: list[Path] = []
    if model.artifact is not None:
        candidates.append(model.artifact)
    legacy = MODEL_DIR / f"{model.algorithm}.joblib"
    if competition == DEFAULT_COMPETITION and legacy not in candidates:
        candidates.append(legacy)
    for path in candidates:
        if path.is_file():
            if model.artifact is not None and path != model.artifact:
                logger.warning(
                    "Recorded artifact %s is missing; loaded legacy %s instead.",
                    model.artifact,
                    path,
                )
            return joblib.load(path)
    raise ProductionModelUnavailable(
        f"No artifact on disk for {model.algorithm} (run {model.model_run_id}). "
        f"Looked in: {', '.join(str(p) for p in candidates)}. Train first."
    )


def artifact_path_for_run(
    algorithm: str,
    model_run_id: int,
    *,
    competition: str = DEFAULT_COMPETITION,
) -> Path:
    """Stable, unique artifact path for a run. Never overwritten by a later run."""
    folder = MODEL_DIR if competition == DEFAULT_COMPETITION else MODEL_DIR / competition
    return folder / f"{algorithm}-run{int(model_run_id):05d}.joblib"
