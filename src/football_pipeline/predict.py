"""Probabilities for unplayed matches using a model fit only on past training rows."""

from __future__ import annotations

import argparse
import csv
import json
import logging

import joblib
import pandas as pd

from football_pipeline.config import ROOT
from football_pipeline.dataset import FEATURE_COLUMNS
from football_pipeline.db import connect
from football_pipeline.models import SKLEARN_MODELS, feature_matrix, predict_proba_3way
from football_pipeline.train import MODEL_DIR, PRODUCTION_MODELS, REPORT_PATH

logger = logging.getLogger(__name__)

CSV_PATH = ROOT / "data" / "processed" / "upcoming_predictions.csv"


def format_prediction(home: str, away: str, p_away: float, p_draw: float, p_home: float) -> str:
    return (
        f"{home} vs {away}\n"
        f"  Home Win: {p_home:.0%}\n"
        f"  Draw: {p_draw:.0%}\n"
        f"  Away Win: {p_away:.0%}"
    )


def _selected_algorithm() -> str:
    if REPORT_PATH.is_file():
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        name = report.get("selected_by_walkforward_log_loss") or report.get("selected_by_valid_log_loss")
        if name in PRODUCTION_MODELS:
            return name
        name = report.get("selected_sklearn_by_valid_log_loss")
        if name in SKLEARN_MODELS:
            return name
    with connect() as conn:
        with conn.cursor() as cur:
            algorithms = tuple(PRODUCTION_MODELS)
            placeholders = ", ".join(["%s"] * len(algorithms))
            cur.execute(
                f"""
                SELECT algorithm
                FROM model_runs
                WHERE algorithm IN ({placeholders}) AND artifact_path IS NOT NULL
                ORDER BY trained_at DESC
                LIMIT 1
                """,
                algorithms,
            )
            row = cur.fetchone()
    if not row:
        raise RuntimeError("No trained production model. Run python -m football_pipeline.train first.")
    return row[0]


def _latest_run_id(algorithm: str) -> int:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM model_runs
                WHERE algorithm = %s
                ORDER BY trained_at DESC
                LIMIT 1
                """,
                (algorithm,),
            )
            row = cur.fetchone()
    if not row:
        raise RuntimeError(f"No model_runs row for {algorithm}")
    return int(row[0])


def load_upcoming_frame() -> pd.DataFrame:
    feature_sql = ", ".join(f"f.{name}" for name in FEATURE_COLUMNS)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT m.id AS match_id,
                       m.match_date,
                       m.home_team_id,
                       m.away_team_id,
                       home.canonical_name AS home_team,
                       away.canonical_name AS away_team,
                       {feature_sql}
                FROM matches AS m
                JOIN match_features AS f ON f.match_id = m.id
                JOIN teams AS home ON home.id = m.home_team_id
                JOIN teams AS away ON away.id = m.away_team_id
                WHERE m.is_played = FALSE
                ORDER BY m.match_date, m.kickoff_time, m.id
                """
            )
            columns = [col.name for col in cur.description]
            rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _predict_proba(algorithm: str, model, frame: pd.DataFrame):
    if algorithm == "poisson_dixon_coles":
        return model.predict_proba(frame["home_team_id"], frame["away_team_id"])
    return predict_proba_3way(model, feature_matrix(frame))


def predict_upcoming() -> list[dict]:
    algorithm = _selected_algorithm()
    artifact = MODEL_DIR / f"{algorithm}.joblib"
    if not artifact.is_file():
        raise RuntimeError(f"Missing {artifact}. Train first.")
    model = joblib.load(artifact)
    frame = load_upcoming_frame()
    if frame.empty:
        logger.info("No unplayed matches with features; skipping upcoming predictions.")
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "match_id",
                    "home_team",
                    "away_team",
                    "match_date",
                    "algorithm",
                    "p_away",
                    "p_draw",
                    "p_home",
                    "predicted_class",
                ],
            )
            writer.writeheader()
        return []
    proba = _predict_proba(algorithm, model, frame)
    run_id = _latest_run_id(algorithm)
    records = []
    insert_rows = []
    for row, probs in zip(frame.itertuples(index=False), proba, strict=True):
        predicted = int(probs.argmax())
        record = {
            "match_id": int(row.match_id),
            "home_team": row.home_team,
            "away_team": row.away_team,
            "match_date": str(row.match_date),
            "algorithm": algorithm,
            "p_away": float(probs[0]),
            "p_draw": float(probs[1]),
            "p_home": float(probs[2]),
            "predicted_class": predicted,
        }
        records.append(record)
        insert_rows.append(
            (int(row.match_id), run_id, float(probs[0]), float(probs[1]), float(probs[2]), predicted)
        )
        logger.info("\n%s", format_prediction(row.home_team, row.away_team, *probs))

    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO predictions (
                    match_id, model_run_id, p_away, p_draw, p_home, predicted_class
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (match_id, model_run_id) DO UPDATE SET
                    p_away = EXCLUDED.p_away,
                    p_draw = EXCLUDED.p_draw,
                    p_home = EXCLUDED.p_home,
                    predicted_class = EXCLUDED.predicted_class
                WHERE NOT EXISTS (
                    SELECT 1 FROM matches AS played
                    WHERE played.id = predictions.match_id AND played.is_played
                )
                """,
                insert_rows,
            )
        conn.commit()

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    logger.info("Wrote %s using %s (model_run_id=%s)", CSV_PATH, algorithm, run_id)
    return records


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    argparse.ArgumentParser(description="Predict Home/Draw/Away probabilities for unplayed matches.").parse_args()
    predict_upcoming()


if __name__ == "__main__":
    main()
