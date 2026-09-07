"""Walk-forward selection, then one production retrain and one 2025/26 test."""

from __future__ import annotations

import argparse
import csv
import json
import logging

import joblib
import numpy as np
import pandas as pd
from psycopg.types.json import Jsonb

from football_pipeline.config import ROOT
from football_pipeline.constants import FEATURE_VERSION
from football_pipeline.dataset import FEATURE_COLUMNS
from football_pipeline.db import connect
from football_pipeline.metrics import CLASS_NAMES, evaluate_split
from football_pipeline.models import SKLEARN_MODELS, feature_matrix, predict_proba_3way, target_vector
from football_pipeline.walkforward import (
    PRODUCTION_TRAIN_END,
    TEST_SEASON_END,
    TEST_SEASON_NAME,
    WALKFORWARD_FOLDS,
    WalkForwardError,
    aggregate_scalar_metrics,
    assert_folds_exclude_test_season,
    flatten_metrics,
    fold_frames,
    select_by_walkforward,
    slice_season,
    slice_through,
)

logger = logging.getLogger(__name__)

REPORT_PATH = ROOT / "data" / "processed" / "model_metrics.json"
WALKFORWARD_JSON = ROOT / "data" / "processed" / "walkforward_metrics.json"
WALKFORWARD_FOLDS_CSV = ROOT / "data" / "processed" / "walkforward_folds.csv"
WALKFORWARD_SUMMARY_CSV = ROOT / "data" / "processed" / "walkforward_summary.csv"
FINAL_TEST_JSON = ROOT / "data" / "processed" / "final_test_metrics.json"
MODEL_DIR = ROOT / "data" / "processed" / "models"
PRODUCTION_MODELS = set(SKLEARN_MODELS)
CANDIDATE_MODELS = {
    "logistic_regression": SKLEARN_MODELS["logistic_regression"],
    "logistic_regression_balanced": SKLEARN_MODELS["logistic_regression_balanced"],
    "random_forest": SKLEARN_MODELS["random_forest"],
    "xgboost": SKLEARN_MODELS["xgboost"],
}


def load_training_frame() -> pd.DataFrame:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT tr.match_id, tr.match_date, tr.split, tr.result_code,
                       m.home_team_id, m.away_team_id, m.home_goals, m.away_goals,
                       {", ".join(f"tr.{name}" for name in FEATURE_COLUMNS)}
                FROM training_rows AS tr
                JOIN matches AS m ON m.id = tr.match_id
                ORDER BY tr.match_date, tr.match_id
                """
            )
            columns = [col.name for col in cur.description]
            rows = cur.fetchall()
    if not rows:
        raise RuntimeError("training_rows is empty. Run python -m football_pipeline.dataset first.")
    return pd.DataFrame(rows, columns=columns)


def _date_bounds(frame: pd.DataFrame) -> tuple:
    if frame is None or frame.empty:
        return None, None
    dates = frame["match_date"]
    return dates.min(), dates.max()


def _split_summary(frame: pd.DataFrame) -> dict:
    return {
        "n": int(len(frame)),
        "start": str(frame["match_date"].min()) if len(frame) else None,
        "end": str(frame["match_date"].max()) if len(frame) else None,
        "away": int((frame["result_code"] == 0).sum()),
        "draw": int((frame["result_code"] == 1).sum()),
        "home": int((frame["result_code"] == 2).sum()),
    }


def _insert_run(
    *,
    algorithm: str,
    metrics: dict,
    train: pd.DataFrame,
    valid: pd.DataFrame | None,
    test: pd.DataFrame | None,
    artifact_path: str | None,
) -> int:
    train_start, train_end = _date_bounds(train)
    valid_start, valid_end = _date_bounds(valid if valid is not None else pd.DataFrame())
    test_start, test_end = _date_bounds(test if test is not None else pd.DataFrame())
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_runs (
                    train_start, train_end, valid_start, valid_end,
                    test_start, test_end, feature_version, algorithm,
                    metrics, artifact_path
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    train_start,
                    train_end,
                    valid_start,
                    valid_end,
                    test_start,
                    test_end,
                    FEATURE_VERSION,
                    algorithm,
                    Jsonb(metrics),
                    artifact_path,
                ),
            )
            run_id = cur.fetchone()[0]
        conn.commit()
    return run_id


def _store_predictions(run_id: int, frame: pd.DataFrame, proba: np.ndarray) -> None:
    predicted = proba.argmax(axis=1)
    records = [
        (int(row.match_id), run_id, float(probs[0]), float(probs[1]), float(probs[2]), int(cls))
        for row, probs, cls in zip(frame.itertuples(index=False), proba, predicted, strict=True)
    ]
    if not records:
        return
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
                """,
                records,
            )
        conn.commit()


def _fit_and_evaluate(factory, train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    pipe = factory()
    pipe.fit(feature_matrix(train), target_vector(train))
    proba = predict_proba_3way(pipe, feature_matrix(valid))
    return evaluate_split(target_vector(valid), proba)


def _write_fold_csv(rows: list[dict]) -> None:
    WALKFORWARD_FOLDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm",
        "fold",
        "train_end",
        "valid_end",
        "train_n",
        "valid_n",
        "log_loss",
        "accuracy",
        "f1_macro",
        "precision_away",
        "precision_draw",
        "precision_home",
        "recall_away",
        "recall_draw",
        "recall_home",
        "f1_away",
        "f1_draw",
        "f1_home",
        "draw_recall",
        "draw_f1",
        "confusion_matrix",
    ]
    with WALKFORWARD_FOLDS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_summary_csv(summaries: dict[str, dict]) -> None:
    fieldnames = ["algorithm", "stat", *list(next(iter(summaries.values()))["mean"].keys())]
    with WALKFORWARD_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for algorithm, block in summaries.items():
            for stat in ("mean", "median", "std", "min", "max"):
                writer.writerow({"algorithm": algorithm, "stat": stat, **block[stat]})


def train_and_evaluate() -> dict:
    frame = load_training_frame()
    assert_folds_exclude_test_season(frame)
    production_train = slice_through(frame, PRODUCTION_TRAIN_END)
    test = slice_season(frame, after=PRODUCTION_TRAIN_END, through=TEST_SEASON_END)
    if production_train.empty:
        raise WalkForwardError(f"No rows through {PRODUCTION_TRAIN_END}")
    if test.empty:
        raise WalkForwardError(f"No {TEST_SEASON_NAME} test rows")

    logger.info(
        "Walk-forward on %s candidate models, %s folds; production train n=%s through %s; test n=%s (%s)",
        len(CANDIDATE_MODELS),
        len(WALKFORWARD_FOLDS),
        len(production_train),
        PRODUCTION_TRAIN_END,
        len(test),
        TEST_SEASON_NAME,
    )

    fold_rows: list[dict] = []
    by_model: dict[str, dict] = {}
    for algorithm, factory in CANDIDATE_MODELS.items():
        fold_metrics = []
        fold_flat = []
        for fold in WALKFORWARD_FOLDS:
            train, valid = fold_frames(frame, fold)
            metrics = _fit_and_evaluate(factory, train, valid)
            flat = flatten_metrics(metrics)
            fold_flat.append(flat)
            fold_metrics.append(
                {
                    "fold": fold.name,
                    "train_end": fold.train_end.isoformat(),
                    "valid_end": fold.valid_end.isoformat(),
                    "train": _split_summary(train),
                    "valid": _split_summary(valid),
                    "metrics": metrics,
                }
            )
            fold_rows.append(
                {
                    "algorithm": algorithm,
                    "fold": fold.name,
                    "train_end": fold.train_end.isoformat(),
                    "valid_end": fold.valid_end.isoformat(),
                    "train_n": int(len(train)),
                    "valid_n": int(len(valid)),
                    "log_loss": flat["log_loss"],
                    "accuracy": flat["accuracy"],
                    "f1_macro": flat["f1_macro"],
                    "precision_away": flat["precision_away"],
                    "precision_draw": flat["precision_draw"],
                    "precision_home": flat["precision_home"],
                    "recall_away": flat["recall_away"],
                    "recall_draw": flat["recall_draw"],
                    "recall_home": flat["recall_home"],
                    "f1_away": flat["f1_away"],
                    "f1_draw": flat["f1_draw"],
                    "f1_home": flat["f1_home"],
                    "draw_recall": flat["draw_recall"],
                    "draw_f1": flat["draw_f1"],
                    "confusion_matrix": json.dumps(flat["confusion_matrix"]),
                }
            )
            logger.info(
                "%s %s train_n=%s valid_n=%s log_loss=%.4f acc=%.3f macro_f1=%.3f draw_f1=%.3f draw_recall=%.3f",
                algorithm,
                fold.name,
                len(train),
                len(valid),
                flat["log_loss"],
                flat["accuracy"],
                flat["f1_macro"],
                flat["draw_f1"],
                flat["draw_recall"],
            )
        by_model[algorithm] = {
            "folds": fold_metrics,
            "summary": aggregate_scalar_metrics(fold_flat),
        }

    summaries = {name: payload["summary"] for name, payload in by_model.items()}
    selected, selection_reason = select_by_walkforward(summaries)
    logger.info("%s", selection_reason)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    factory = CANDIDATE_MODELS[selected]
    production = factory()
    production.fit(feature_matrix(production_train), target_vector(production_train))
    artifact = MODEL_DIR / f"{selected}.joblib"
    joblib.dump(production, artifact)

    test_proba = predict_proba_3way(production, feature_matrix(test))
    test_metrics = evaluate_split(target_vector(test), test_proba)
    test_flat = flatten_metrics(test_metrics)
    logger.info(
        "Final %s test %s n=%s log_loss=%.4f acc=%.3f macro_f1=%.3f draw_f1=%.3f draw_recall=%.3f (unused for selection)",
        selected,
        TEST_SEASON_NAME,
        test_flat["n"],
        test_flat["log_loss"],
        test_flat["accuracy"],
        test_flat["f1_macro"],
        test_flat["draw_f1"],
        test_flat["draw_recall"],
    )

    run_metrics = {
        "walkforward_mean_log_loss": summaries[selected]["mean"]["log_loss"],
        "test": test_metrics,
        "selection": selection_reason,
        "note": (
            f"Retrained on all permitted history through {PRODUCTION_TRAIN_END} (2024/25). "
            f"{TEST_SEASON_NAME} evaluated once and was not used for selection."
        ),
    }
    run_id = _insert_run(
        algorithm=selected,
        metrics=run_metrics,
        train=production_train,
        valid=None,
        test=test,
        artifact_path=str(artifact),
    )
    _store_predictions(run_id, test, test_proba)

    walkforward_report = {
        "feature_version": FEATURE_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "class_encoding": CLASS_NAMES,
        "methodology": (
            "Expanding-window walk-forward: each fold trains on all earlier seasons "
            "and validates on the next full season. 2025/26 is held out. 2026/27 is live-only."
        ),
        "folds": [
            {
                "name": fold.name,
                "train_through": f"{fold.train_through_start_year}/{str(fold.train_through_start_year + 1)[2:]}",
                "valid_season": fold.name,
                "train_end": fold.train_end.isoformat(),
                "valid_end": fold.valid_end.isoformat(),
            }
            for fold in WALKFORWARD_FOLDS
        ],
        "production_train_end": PRODUCTION_TRAIN_END.isoformat(),
        "test_season": TEST_SEASON_NAME,
        "test_end": TEST_SEASON_END.isoformat(),
        "candidates": list(CANDIDATE_MODELS),
        "models": by_model,
        "selected_algorithm": selected,
        "selection_reason": selection_reason,
    }
    final_test_report = {
        "algorithm": selected,
        "model_run_id": run_id,
        "artifact_path": str(artifact),
        "trained_through": PRODUCTION_TRAIN_END.isoformat(),
        "train": _split_summary(production_train),
        "test_season": TEST_SEASON_NAME,
        "test": {**_split_summary(test), **test_flat, "metrics": test_metrics},
        "note": (
            f"{TEST_SEASON_NAME} was not used for walk-forward selection, feature choices, "
            "hyperparameters, class weights, or thresholds. 2026/27 is live-only."
        ),
    }
    report = {
        "feature_version": FEATURE_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "class_encoding": CLASS_NAMES,
        "selected_by_walkforward_log_loss": selected,
        "selected_by_valid_log_loss": selected,
        "selected_sklearn_by_valid_log_loss": selected,
        "selection_reason": selection_reason,
        "walkforward": {
            "folds": walkforward_report["folds"],
            "summary": summaries,
        },
        "production_train": _split_summary(production_train),
        "test_holdout": final_test_report["test"],
        "note": walkforward_report["methodology"],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    WALKFORWARD_JSON.write_text(json.dumps(walkforward_report, indent=2, default=str), encoding="utf-8")
    FINAL_TEST_JSON.write_text(json.dumps(final_test_report, indent=2, default=str), encoding="utf-8")
    _write_fold_csv(fold_rows)
    _write_summary_csv(summaries)
    logger.info(
        "Wrote %s, %s, %s; production model %s (run_id=%s)",
        WALKFORWARD_JSON,
        FINAL_TEST_JSON,
        REPORT_PATH,
        selected,
        run_id,
    )
    return report


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    argparse.ArgumentParser(
        description="Walk-forward model selection, then one 2025/26 test evaluation."
    ).parse_args()
    train_and_evaluate()


if __name__ == "__main__":
    main()
