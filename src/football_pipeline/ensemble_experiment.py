"""Walk-forward experiment: individual models and probability ensembles.

Does not write production artifacts (model_metrics.json, joblib, model_runs,
or predictions). 2025/26 is scored only if an ensemble beats logistic regression
on mean walk-forward log loss.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from football_pipeline.calibration import class_summary
from football_pipeline.config import ROOT
from football_pipeline.constants import FEATURE_VERSION
from football_pipeline.metrics import evaluate_split
from football_pipeline.models import SKLEARN_MODELS, feature_matrix, predict_proba_3way, target_vector
from football_pipeline.poisson import DixonColesPoisson
from football_pipeline.train import load_training_frame
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

EXPERIMENT_DIR = ROOT / "data" / "processed" / "experiments"
REPORT_PATH = EXPERIMENT_DIR / "ensemble_walkforward.json"
PRODUCTION_REPORT = ROOT / "data" / "processed" / "model_metrics.json"
PRODUCTION_MODEL = "logistic_regression"

SKLEARN_CANDIDATES = (
    "logistic_regression",
    "logistic_regression_balanced",
    "random_forest",
    "xgboost",
)
POISSON = "poisson_dixon_coles"
BASE_MODELS = (*SKLEARN_CANDIDATES, POISSON)

# Pre-specified weights. Not estimated on 2025/26.
FIXED_ENSEMBLES = {
    "ensemble_lr_xgb_50_50": {"logistic_regression": 0.5, "xgboost": 0.5},
    "ensemble_lr_xgb_60_40": {"logistic_regression": 0.6, "xgboost": 0.4},
    "ensemble_lr_xgb_70_30": {"logistic_regression": 0.7, "xgboost": 0.3},
    "ensemble_lr_xgb_poisson_equal": {
        "logistic_regression": 1.0,
        "xgboost": 1.0,
        "poisson_dixon_coles": 1.0,
    },
    "ensemble_lr_xgb_poisson_50_30_20": {
        "logistic_regression": 0.5,
        "xgboost": 0.3,
        "poisson_dixon_coles": 0.2,
    },
}
INVLL_MEMBERS = ("logistic_regression", "xgboost", "poisson_dixon_coles")
INVLL_NAME = "ensemble_lr_xgb_poisson_invll"


def mix_proba(parts: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted average of 1X2 probability matrices. Rows are renormalized to 1."""
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value.")
    stacked = None
    for name, weight in weights.items():
        contrib = (float(weight) / total) * np.asarray(parts[name], dtype=float)
        stacked = contrib if stacked is None else stacked + contrib
    stacked = np.clip(stacked, 0.0, None)
    denom = stacked.sum(axis=1, keepdims=True)
    missing = denom.ravel() <= 0
    stacked[missing] = 1.0 / 3.0
    denom = stacked.sum(axis=1, keepdims=True)
    return stacked / denom


def inverse_logloss_weights(fold_log_loss: dict[str, list[float]], fold_index: int, members: tuple[str, ...]) -> dict[str, float]:
    """Weights from other folds only. Never uses 2025/26."""
    scores: dict[str, float] = {}
    for name in members:
        others = [loss for i, loss in enumerate(fold_log_loss[name]) if i != fold_index]
        mean_loss = float(np.mean(others)) if others else 1.0
        scores[name] = 1.0 / max(mean_loss, 1e-9)
    total = sum(scores.values())
    return {name: scores[name] / total for name in members}


def all_fold_inverse_logloss_weights(fold_log_loss: dict[str, list[float]], members: tuple[str, ...]) -> dict[str, float]:
    """Weights from every walk-forward fold. Still unused on 2025/26 until after selection."""
    scores = {
        name: 1.0 / max(float(np.mean(fold_log_loss[name])), 1e-9) for name in members
    }
    total = sum(scores.values())
    return {name: scores[name] / total for name in members}


def _fit_sklearn(algorithm: str, train: pd.DataFrame, scored: pd.DataFrame) -> np.ndarray:
    pipe = SKLEARN_MODELS[algorithm]()
    pipe.fit(feature_matrix(train), target_vector(train))
    return predict_proba_3way(pipe, feature_matrix(scored))


def _played(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(subset=["home_goals", "away_goals"]).reset_index(drop=True)


def _fit_poisson(train: pd.DataFrame, scored: pd.DataFrame) -> np.ndarray:
    model = DixonColesPoisson()
    train_played = _played(train)
    if train_played.empty:
        raise WalkForwardError("Poisson train split has no scored matches.")
    model.fit(
        train_played["home_team_id"].to_numpy(),
        train_played["away_team_id"].to_numpy(),
        train_played["home_goals"].to_numpy(),
        train_played["away_goals"].to_numpy(),
    )
    return model.predict_proba(scored["home_team_id"].to_numpy(), scored["away_team_id"].to_numpy())


def _predict_base(algorithm: str, train: pd.DataFrame, scored: pd.DataFrame) -> np.ndarray:
    if algorithm == POISSON:
        return _fit_poisson(train, scored)
    return _fit_sklearn(algorithm, train, scored)


def _enrich(metrics: dict) -> dict:
    flat = flatten_metrics(metrics)
    brier_cls = metrics.get("brier_by_class") or {}
    flat["brier"] = metrics.get("brier")
    flat["brier_away"] = brier_cls.get("away")
    flat["brier_draw"] = brier_cls.get("draw")
    flat["brier_home"] = brier_cls.get("home")
    return flat


def _calibration(y: np.ndarray, proba: np.ndarray) -> dict:
    return {name: class_summary(y, proba, name) for name in ("away", "draw", "home")}


def _folds_won(fold_losses: dict[str, list[float]]) -> dict[str, int]:
    names = list(fold_losses)
    n_folds = len(next(iter(fold_losses.values())))
    wins = {name: 0 for name in names}
    for i in range(n_folds):
        best = min(names, key=lambda name: fold_losses[name][i])
        wins[best] += 1
    return wins


def _candidate_block(fold_flat: list[dict], pooled_y: np.ndarray, pooled_p: np.ndarray) -> dict:
    summary = aggregate_scalar_metrics(fold_flat)
    extra_keys = ("brier", "brier_away", "brier_draw", "brier_home")
    for key in extra_keys:
        values = np.array([row[key] for row in fold_flat], dtype=float)
        summary["mean"][key] = float(np.mean(values))
        summary["median"][key] = float(np.median(values))
    return {
        "summary": summary,
        "folds": fold_flat,
        "pooled_calibration": _calibration(pooled_y, pooled_p),
        "pooled_n": int(len(pooled_y)),
    }


def collect_base_fold_predictions(frame: pd.DataFrame) -> dict:
    """Fit each base model on each walk-forward train split; score that fold's valid season."""
    assert_folds_exclude_test_season(frame)
    by_model: dict[str, dict] = {name: {"y": [], "proba": []} for name in BASE_MODELS}
    fold_log_loss = {name: [] for name in BASE_MODELS}
    fold_details: dict[str, list[dict]] = {name: [] for name in BASE_MODELS}
    for fold in WALKFORWARD_FOLDS:
        train, valid = fold_frames(frame, fold)
        y = target_vector(valid)
        logger.info("Fold %s train_n=%s valid_n=%s", fold.name, len(train), len(valid))
        for algorithm in BASE_MODELS:
            proba = _predict_base(algorithm, train, valid)
            metrics = evaluate_split(y, proba)
            flat = _enrich(metrics)
            fold_log_loss[algorithm].append(flat["log_loss"])
            by_model[algorithm]["y"].append(y)
            by_model[algorithm]["proba"].append(proba)
            fold_details[algorithm].append(
                {
                    "fold": fold.name,
                    "train_n": int(len(train)),
                    "valid_n": int(len(valid)),
                    **flat,
                }
            )
            logger.info(
                "%s %s log_loss=%.4f acc=%.3f draw_f1=%.3f",
                algorithm,
                fold.name,
                flat["log_loss"],
                flat["accuracy"],
                flat["draw_f1"],
            )
    return {
        "by_model": by_model,
        "fold_log_loss": fold_log_loss,
        "fold_details": fold_details,
    }


def _pool(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts, axis=0)


def build_candidates(collected: dict) -> dict[str, dict]:
    by_model = collected["by_model"]
    fold_log_loss = dict(collected["fold_log_loss"])
    fold_details = dict(collected["fold_details"])
    n_folds = len(WALKFORWARD_FOLDS)

    candidates: dict[str, dict] = {}
    for name in BASE_MODELS:
        y = _pool(by_model[name]["y"])
        p = _pool(by_model[name]["proba"])
        candidates[name] = _candidate_block(fold_details[name], y, p)

    for ens_name, weights in FIXED_ENSEMBLES.items():
        details = []
        ys, ps = [], []
        fold_log_loss[ens_name] = []
        for i in range(n_folds):
            parts = {member: by_model[member]["proba"][i] for member in weights}
            mixed = mix_proba(parts, weights)
            y = by_model[PRODUCTION_MODEL]["y"][i]
            metrics = evaluate_split(y, mixed)
            flat = _enrich(metrics)
            details.append({"fold": WALKFORWARD_FOLDS[i].name, **flat})
            fold_log_loss[ens_name].append(flat["log_loss"])
            ys.append(y)
            ps.append(mixed)
        candidates[ens_name] = {
            **_candidate_block(details, _pool(ys), _pool(ps)),
            "weights": {k: float(v / sum(weights.values())) for k, v in weights.items()},
            "weight_source": "pre_specified",
        }

    details = []
    ys, ps = [], []
    fold_log_loss[INVLL_NAME] = []
    fold_weights = []
    for i in range(n_folds):
        weights = inverse_logloss_weights(collected["fold_log_loss"], i, INVLL_MEMBERS)
        parts = {member: by_model[member]["proba"][i] for member in INVLL_MEMBERS}
        mixed = mix_proba(parts, weights)
        y = by_model[PRODUCTION_MODEL]["y"][i]
        metrics = evaluate_split(y, mixed)
        flat = _enrich(metrics)
        details.append({"fold": WALKFORWARD_FOLDS[i].name, "weights": weights, **flat})
        fold_log_loss[INVLL_NAME].append(flat["log_loss"])
        fold_weights.append(weights)
        ys.append(y)
        ps.append(mixed)
    all_fold_weights = all_fold_inverse_logloss_weights(collected["fold_log_loss"], INVLL_MEMBERS)
    candidates[INVLL_NAME] = {
        **_candidate_block(details, _pool(ys), _pool(ps)),
        "weights": all_fold_weights,
        "fold_weights": fold_weights,
        "weight_source": "inverse_log_loss_other_folds",
    }
    wins = _folds_won(fold_log_loss)
    for name, block in candidates.items():
        block["folds_won"] = wins[name]
        block["kind"] = "ensemble" if name.startswith("ensemble_") else "individual"
    return candidates


def _mean_log_loss(candidates: dict[str, dict], name: str) -> float:
    return float(candidates[name]["summary"]["mean"]["log_loss"])


def score_test_blend(
    frame: pd.DataFrame,
    *,
    weights: dict[str, float],
) -> dict:
    """One 2025/26 evaluation. Fits on history through 2024/25 only. Does not save models."""
    production_train = slice_through(frame, PRODUCTION_TRAIN_END)
    test = slice_season(frame, after=PRODUCTION_TRAIN_END, through=TEST_SEASON_END)
    if test.empty:
        raise WalkForwardError(f"No {TEST_SEASON_NAME} test rows")
    parts = {}
    for algorithm in weights:
        parts[algorithm] = _predict_base(algorithm, production_train, test)
        logger.info("Test fit %s n_train=%s n_test=%s", algorithm, len(production_train), len(test))
    y = target_vector(test)
    blended = mix_proba(parts, weights)
    lr = parts[PRODUCTION_MODEL]
    return {
        "test_season": TEST_SEASON_NAME,
        "n": int(len(test)),
        "weights": {k: float(v / sum(weights.values())) for k, v in weights.items()},
        "ensemble": {**_enrich(evaluate_split(y, blended)), "calibration": _calibration(y, blended)},
        "logistic_regression": {
            **_enrich(evaluate_split(y, lr)),
            "calibration": _calibration(y, lr),
        },
        "note": (
            f"{TEST_SEASON_NAME} was not used for walk-forward selection or ensemble weights. "
            "This evaluation does not write production artifacts."
        ),
    }


def run_experiment(*, evaluate_test_if_ensemble_wins: bool = True) -> dict:
    frame = load_training_frame()
    collected = collect_base_fold_predictions(frame)
    candidates = build_candidates(collected)
    summaries = {name: block["summary"] for name, block in candidates.items()}
    winner, reason = select_by_walkforward(summaries)
    lr_mean = _mean_log_loss(candidates, PRODUCTION_MODEL)
    winner_mean = _mean_log_loss(candidates, winner)
    ensembles = {name: block for name, block in candidates.items() if block["kind"] == "ensemble"}
    best_ensemble = min(ensembles, key=lambda name: _mean_log_loss(candidates, name))
    best_ensemble_mean = _mean_log_loss(candidates, best_ensemble)
    ensemble_beats_lr = best_ensemble_mean < lr_mean - 1e-12

    test_report = None
    if evaluate_test_if_ensemble_wins and ensemble_beats_lr:
        weights = candidates[best_ensemble].get("weights") or FIXED_ENSEMBLES.get(best_ensemble)
        logger.info("Ensemble %s beats logistic (%.4f vs %.4f). Scoring %s once.", best_ensemble, best_ensemble_mean, lr_mean, TEST_SEASON_NAME)
        test_report = score_test_blend(frame, weights=weights)
    else:
        logger.info(
            "No 2025/26 look. Best walk-forward is %s at %.4f; logistic is %.4f; best ensemble %s at %.4f.",
            winner,
            winner_mean,
            lr_mean,
            best_ensemble,
            best_ensemble_mean,
        )

    report = {
        "feature_version": FEATURE_VERSION,
        "production_model": PRODUCTION_MODEL,
        "methodology": (
            "Same expanding-window folds as production selection. Ensembles average "
            "1X2 probabilities from models trained on that fold's train split only. "
            "Inverse-log-loss weights for each fold use the other folds. 2025/26 is unused "
            "for weights. Production artifacts are not written."
        ),
        "folds": [
            {
                "name": fold.name,
                "train_end": fold.train_end.isoformat(),
                "valid_end": fold.valid_end.isoformat(),
            }
            for fold in WALKFORWARD_FOLDS
        ],
        "candidates": candidates,
        "selected_by_walkforward_log_loss": winner,
        "selection_reason": reason,
        "best_ensemble": best_ensemble,
        "ensemble_beats_logistic": ensemble_beats_lr,
        "test_holdout": test_report,
        "wrote_production_artifacts": False,
    }
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", REPORT_PATH)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Walk-forward probability ensembles. Does not change production.")
    parser.add_argument("--skip-test", action="store_true", help="Never score 2025/26, even if an ensemble wins.")
    args = parser.parse_args()
    run_experiment(evaluate_test_if_ensemble_wins=not args.skip_test)


if __name__ == "__main__":
    main()
