"""Walk-forward experiment: individual models and probability ensembles.

Research only. Does not write production artifacts (model_metrics.json, joblib,
model_runs, predictions, or season_forecast.json). Each supported league is
scored independently from its own training_rows.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from football_pipeline.calibration import class_summary
from football_pipeline.competitions import (
    DEFAULT_COMPETITION,
    SUPPORTED_CODES,
    parse_competition,
    parse_competition_list,
)
from football_pipeline.config import ROOT
from football_pipeline.constants import FEATURE_VERSION
from football_pipeline.metrics import evaluate_split
from football_pipeline.models import SKLEARN_MODELS, feature_matrix, predict_proba_3way, target_vector
from football_pipeline.poisson import DixonColesPoisson
from football_pipeline.registry import report_path_for as production_report_path
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
COMBINED_REPORT = EXPERIMENT_DIR / "top5_model_research.json"
PRODUCTION_REPORT = ROOT / "data" / "processed" / "model_metrics.json"
DEFAULT_PRODUCTION_MODEL = "logistic_regression"

SKLEARN_CANDIDATES = (
    "logistic_regression",
    "logistic_regression_balanced",
    "random_forest",
    "xgboost",
)
POISSON = "poisson_dixon_coles"
BASE_MODELS = (*SKLEARN_CANDIDATES, POISSON)

# Pre-specified weights. Not estimated on the holdout season.
FIXED_ENSEMBLES = {
    "ensemble_lr_xgb_50_50": {"logistic_regression": 0.5, "xgboost": 0.5},
    "ensemble_lr_xgb_60_40": {"logistic_regression": 0.6, "xgboost": 0.4},
    "ensemble_lr_xgb_70_30": {"logistic_regression": 0.7, "xgboost": 0.3},
    "ensemble_lr_poisson_50_50": {"logistic_regression": 0.5, "poisson_dixon_coles": 0.5},
    "ensemble_lr_poisson_70_30": {"logistic_regression": 0.7, "poisson_dixon_coles": 0.3},
    "ensemble_xgb_poisson_50_50": {"xgboost": 0.5, "poisson_dixon_coles": 0.5},
    "ensemble_xgb_poisson_70_30": {"xgboost": 0.7, "poisson_dixon_coles": 0.3},
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

# Conservative switch thresholds. Tiny walk-forward gaps are treated as noise.
MIN_WF_LOG_LOSS = 0.005
MIN_HOLDOUT_LOG_LOSS = 0.005
MIN_COMPLEX_WF = 0.010
MIN_COMPLEX_HOLDOUT = 0.010
RATE_KEYS = (
    "mean_pred_away",
    "mean_pred_draw",
    "mean_pred_home",
    "actual_away",
    "actual_draw",
    "actual_home",
    "argmax_away",
    "argmax_draw",
    "argmax_home",
    "brier",
    "brier_away",
    "brier_draw",
    "brier_home",
)


def report_path_for(competition: str = DEFAULT_COMPETITION) -> Path:
    return EXPERIMENT_DIR / parse_competition(competition) / "ensemble_walkforward.json"


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


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
    """Weights from other folds only. Never uses the holdout season."""
    scores: dict[str, float] = {}
    for name in members:
        others = [loss for i, loss in enumerate(fold_log_loss[name]) if i != fold_index]
        mean_loss = float(np.mean(others)) if others else 1.0
        scores[name] = 1.0 / max(mean_loss, 1e-9)
    total = sum(scores.values())
    return {name: scores[name] / total for name in members}


def all_fold_inverse_logloss_weights(fold_log_loss: dict[str, list[float]], members: tuple[str, ...]) -> dict[str, float]:
    """Weights from every walk-forward fold. Still unused on the holdout until after selection."""
    scores = {
        name: 1.0 / max(float(np.mean(fold_log_loss[name])), 1e-9) for name in members
    }
    total = sum(scores.values())
    return {name: scores[name] / total for name in members}


def current_production_algorithm(competition: str = DEFAULT_COMPETITION) -> str:
    """Read-only: the algorithm named by that league's existing training report."""
    path = production_report_path(competition)
    if not path.is_file():
        return DEFAULT_PRODUCTION_MODEL
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return DEFAULT_PRODUCTION_MODEL
    for key in (
        "selected_by_walkforward_log_loss",
        "selected_by_valid_log_loss",
        "selected_sklearn_by_valid_log_loss",
    ):
        name = report.get(key)
        if name:
            return str(name)
    return DEFAULT_PRODUCTION_MODEL


def is_complex_candidate(name: str) -> bool:
    return name.startswith("ensemble_") or name == POISSON


def recommend_action(
    *,
    production: str,
    winner: str,
    wf_production: float,
    wf_winner: float,
    holdout_production: float | None,
    holdout_winner: float | None,
) -> dict:
    """Conservative keep / switch / insufficient decision. Pure: no I/O."""
    wf_gain = float(wf_production) - float(wf_winner)
    holdout_gain = (
        None
        if holdout_production is None or holdout_winner is None
        else float(holdout_production) - float(holdout_winner)
    )
    complex_winner = is_complex_candidate(winner) and not is_complex_candidate(production)
    wf_need = MIN_COMPLEX_WF if complex_winner else MIN_WF_LOG_LOSS
    ho_need = MIN_COMPLEX_HOLDOUT if complex_winner else MIN_HOLDOUT_LOG_LOSS

    if winner == production:
        return {
            "action": "KEEP CURRENT PRODUCTION",
            "reason": (
                f"Walk-forward still selects the current production model ({production}). "
                "No change."
            ),
            "wf_log_loss_gain": wf_gain,
            "holdout_log_loss_gain": holdout_gain,
        }
    if wf_gain < wf_need:
        return {
            "action": "INSUFFICIENT IMPROVEMENT",
            "reason": (
                f"{winner} beats {production} by only {wf_gain:.4f} mean walk-forward log loss, "
                f"below the {wf_need:.3f} bar used to ignore fold noise"
                f"{' for a more complex model' if complex_winner else ''}."
            ),
            "wf_log_loss_gain": wf_gain,
            "holdout_log_loss_gain": holdout_gain,
        }
    if holdout_gain is None:
        return {
            "action": "INSUFFICIENT IMPROVEMENT",
            "reason": (
                f"{winner} wins walk-forward by {wf_gain:.4f} but the holdout season was not scored."
            ),
            "wf_log_loss_gain": wf_gain,
            "holdout_log_loss_gain": None,
        }
    if holdout_gain < ho_need:
        return {
            "action": "INSUFFICIENT IMPROVEMENT",
            "reason": (
                f"{winner} wins walk-forward by {wf_gain:.4f}, but the untouched holdout "
                f"{'worsens' if holdout_gain < 0 else 'improves'} log loss by "
                f"{holdout_gain:.4f} versus {production} (need {ho_need:.3f})."
            ),
            "wf_log_loss_gain": wf_gain,
            "holdout_log_loss_gain": holdout_gain,
        }
    return {
        "action": "SWITCH TO NEW MODEL",
        "reason": (
            f"{winner} improves mean walk-forward log loss by {wf_gain:.4f} and holdout "
            f"log loss by {holdout_gain:.4f} versus {production}."
        ),
        "wf_log_loss_gain": wf_gain,
        "holdout_log_loss_gain": holdout_gain,
    }


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


def _rate_stats(y: np.ndarray, proba: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    pred = proba.argmax(axis=1)
    names = ("away", "draw", "home")
    out: dict[str, float] = {}
    for index, name in enumerate(names):
        out[f"mean_pred_{name}"] = float(proba[:, index].mean())
        out[f"actual_{name}"] = float((y == index).mean())
        out[f"argmax_{name}"] = float((pred == index).mean())
    return out


def _enrich(metrics: dict, y: np.ndarray, proba: np.ndarray) -> dict:
    flat = flatten_metrics(metrics)
    brier_cls = metrics.get("brier_by_class") or {}
    flat["brier"] = metrics.get("brier")
    flat["brier_away"] = brier_cls.get("away")
    flat["brier_draw"] = brier_cls.get("draw")
    flat["brier_home"] = brier_cls.get("home")
    flat.update(_rate_stats(y, proba))
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
    for key in RATE_KEYS:
        values = np.array([row[key] for row in fold_flat], dtype=float)
        summary["mean"][key] = float(np.mean(values))
        summary["median"][key] = float(np.median(values))
        summary["std"][key] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "summary": summary,
        "folds": fold_flat,
        "pooled_calibration": _calibration(pooled_y, pooled_p),
        "pooled_rates": _rate_stats(pooled_y, pooled_p),
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
            flat = _enrich(metrics, y, proba)
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
    y_ref = by_model[DEFAULT_PRODUCTION_MODEL]["y"]

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
            y = y_ref[i]
            metrics = evaluate_split(y, mixed)
            flat = _enrich(metrics, y, mixed)
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
        y = y_ref[i]
        metrics = evaluate_split(y, mixed)
        flat = _enrich(metrics, y, mixed)
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


def _candidate_weights(candidates: dict[str, dict], name: str) -> dict[str, float]:
    if name in BASE_MODELS:
        return {name: 1.0}
    weights = candidates[name].get("weights")
    if weights:
        return {str(k): float(v) for k, v in weights.items()}
    if name in FIXED_ENSEMBLES:
        return dict(FIXED_ENSEMBLES[name])
    raise WalkForwardError(f"No blend weights for {name}")


def score_holdout(frame: pd.DataFrame, *, weights: dict[str, float]) -> dict:
    """One holdout evaluation. Fits on history through the season before the holdout only."""
    production_train = slice_through(frame, PRODUCTION_TRAIN_END)
    test = slice_season(frame, after=PRODUCTION_TRAIN_END, through=TEST_SEASON_END)
    if test.empty:
        raise WalkForwardError(f"No {TEST_SEASON_NAME} test rows")
    parts = {}
    for algorithm in weights:
        parts[algorithm] = _predict_base(algorithm, production_train, test)
        logger.info("Holdout fit %s n_train=%s n_test=%s", algorithm, len(production_train), len(test))
    y = target_vector(test)
    blended = mix_proba(parts, weights) if len(weights) > 1 else next(iter(parts.values()))
    metrics = evaluate_split(y, blended)
    return {
        "test_season": TEST_SEASON_NAME,
        "n": int(len(test)),
        "weights": {k: float(v / sum(weights.values())) for k, v in weights.items()},
        **_enrich(metrics, y, blended),
        "calibration": _calibration(y, blended),
        "note": (
            f"{TEST_SEASON_NAME} was not used for walk-forward selection or ensemble weights. "
            "This evaluation does not write production artifacts."
        ),
    }


def score_test_blend(
    frame: pd.DataFrame,
    *,
    weights: dict[str, float],
) -> dict:
    """Backward-compatible holdout blend scorer used by the original Premier League experiment."""
    ensemble = score_holdout(frame, weights=weights)
    lr = score_holdout(frame, weights={DEFAULT_PRODUCTION_MODEL: 1.0})
    return {
        "test_season": TEST_SEASON_NAME,
        "n": ensemble["n"],
        "weights": ensemble["weights"],
        "ensemble": ensemble,
        "logistic_regression": lr,
        "note": ensemble["note"],
    }


def _comparison_row(
    *,
    league: str,
    name: str,
    block: dict,
    holdout: dict | None,
) -> dict:
    mean = block["summary"]["mean"]
    median = block["summary"]["median"]
    std = block["summary"]["std"]
    return {
        "league": league,
        "candidate": name,
        "kind": block.get("kind"),
        "wf_log_loss": mean["log_loss"],
        "wf_log_loss_median": median["log_loss"],
        "wf_log_loss_std": std["log_loss"],
        "wf_accuracy": mean["accuracy"],
        "wf_macro_f1": mean["f1_macro"],
        "wf_draw_f1": mean["draw_f1"],
        "wf_draw_recall": mean["draw_recall"],
        "wf_home_f1": mean["f1_home"],
        "wf_away_f1": mean["f1_away"],
        "wf_brier_draw": mean.get("brier_draw"),
        "holdout_log_loss": None if holdout is None else holdout.get("log_loss"),
        "holdout_accuracy": None if holdout is None else holdout.get("accuracy"),
        "holdout_draw_f1": None if holdout is None else holdout.get("draw_f1"),
        "holdout_draw_recall": None if holdout is None else holdout.get("draw_recall"),
    }


def run_experiment(
    *,
    competition: str = DEFAULT_COMPETITION,
    evaluate_holdout: bool = True,
    evaluate_test_if_ensemble_wins: bool | None = None,
) -> dict:
    code = parse_competition(competition)
    if evaluate_test_if_ensemble_wins is not None:
        evaluate_holdout = bool(evaluate_test_if_ensemble_wins) or evaluate_holdout
    production = current_production_algorithm(code)
    frame = load_training_frame(competition=code)
    collected = collect_base_fold_predictions(frame)
    candidates = build_candidates(collected)
    if production not in candidates:
        logger.warning(
            "Production algorithm %s is not in the research set for %s; comparing to %s",
            production,
            code,
            DEFAULT_PRODUCTION_MODEL,
        )
        production = DEFAULT_PRODUCTION_MODEL
    summaries = {name: block["summary"] for name, block in candidates.items()}
    winner, reason = select_by_walkforward(summaries)
    winner_mean = _mean_log_loss(candidates, winner)
    production_mean = _mean_log_loss(candidates, production)
    ensembles = {name: block for name, block in candidates.items() if block["kind"] == "ensemble"}
    best_ensemble = min(ensembles, key=lambda name: _mean_log_loss(candidates, name))
    best_poisson = _mean_log_loss(candidates, POISSON)

    holdout_by_name: dict[str, dict] = {}
    if evaluate_holdout:
        for name in dict.fromkeys((winner, production)):
            logger.info("Scoring %s holdout for %s (%s)", TEST_SEASON_NAME, code, name)
            holdout_by_name[name] = score_holdout(frame, weights=_candidate_weights(candidates, name))

    decision = recommend_action(
        production=production,
        winner=winner,
        wf_production=production_mean,
        wf_winner=winner_mean,
        holdout_production=(holdout_by_name.get(production) or {}).get("log_loss"),
        holdout_winner=(holdout_by_name.get(winner) or {}).get("log_loss"),
    )

    rows = [
        _comparison_row(
            league=code,
            name=name,
            block=block,
            holdout=holdout_by_name.get(name),
        )
        for name, block in candidates.items()
    ]
    rows.sort(key=lambda row: row["wf_log_loss"])

    report = {
        "competition": code,
        "feature_version": FEATURE_VERSION,
        "production_model": production,
        "methodology": (
            "Same expanding-window folds as production selection. Each league is trained "
            "and scored in isolation. Ensembles average 1X2 probabilities from models "
            "trained on that fold's train split only. Inverse-log-loss weights for each "
            f"fold use the other folds. {TEST_SEASON_NAME} is unused for selection or "
            "weights. Production artifacts are not written."
        ),
        "champion_backtest": {
            "ran": False,
            "reason": (
                "The existing historical Season Forecast backtest is Premier League-only, "
                "hardcoded to 20 clubs and unweighted logistic regression, and rebuilds "
                "pre-kickoff features at every checkpoint from the database. Extending it "
                "to five leagues, 18-club seasons, Poisson, and ensembles is a separate "
                "experiment, not this walk-forward script."
            ),
        },
        "folds": [
            {
                "name": fold.name,
                "train_end": fold.train_end.isoformat(),
                "valid_end": fold.valid_end.isoformat(),
            }
            for fold in WALKFORWARD_FOLDS
        ],
        "candidates": candidates,
        "comparison_table": rows,
        "selected_by_walkforward_log_loss": winner,
        "selection_reason": reason,
        "best_ensemble": best_ensemble,
        "poisson_walkforward_log_loss": best_poisson,
        "recommendation": decision,
        "test_holdout": {
            "winner": holdout_by_name.get(winner),
            "production": holdout_by_name.get(production),
        },
        "wrote_production_artifacts": False,
    }
    dest = report_path_for(code)
    prod_report = production_report_path(code)
    if dest.resolve() == prod_report.resolve() or "experiments" not in dest.parts:
        raise WalkForwardError(f"Refusing to write research output over production report {prod_report}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    logger.info("Wrote %s", dest)
    return report


def run_all(
    competitions: tuple[str, ...] = SUPPORTED_CODES,
    *,
    evaluate_holdout: bool = True,
) -> dict:
    leagues = []
    for code in competitions:
        logger.info("=== Research experiment %s ===", code)
        leagues.append(run_experiment(competition=code, evaluate_holdout=evaluate_holdout))
    combined = {
        "feature_version": FEATURE_VERSION,
        "holdout_season": TEST_SEASON_NAME,
        "wrote_production_artifacts": False,
        "champion_backtest_ran": False,
        "leagues": {row["competition"]: row for row in leagues},
        "comparison_table": [item for row in leagues for item in row["comparison_table"]],
        "recommendations": {
            row["competition"]: {
                "production_model": row["production_model"],
                "walkforward_winner": row["selected_by_walkforward_log_loss"],
                **row["recommendation"],
            }
            for row in leagues
        },
    }
    slim = {
        "feature_version": FEATURE_VERSION,
        "holdout_season": TEST_SEASON_NAME,
        "wrote_production_artifacts": False,
        "champion_backtest_ran": False,
        "comparison_table": combined["comparison_table"],
        "recommendations": combined["recommendations"],
        "per_league": {
            code: {
                "production_model": row["production_model"],
                "walkforward_winner": row["selected_by_walkforward_log_loss"],
                "selection_reason": row["selection_reason"],
                "best_ensemble": row["best_ensemble"],
                "poisson_walkforward_log_loss": row["poisson_walkforward_log_loss"],
                "recommendation": row["recommendation"],
                "test_holdout": row["test_holdout"],
                "champion_backtest": row["champion_backtest"],
            }
            for code, row in combined["leagues"].items()
        },
    }
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_REPORT.write_text(json.dumps(combined, indent=2, default=_json_default), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(combined, indent=2, default=_json_default), encoding="utf-8")
    summary_path = EXPERIMENT_DIR / "top5_model_research_summary.json"
    summary_path.write_text(json.dumps(slim, indent=2, default=_json_default), encoding="utf-8")
    logger.info("Wrote %s", COMBINED_REPORT)
    logger.info("Wrote %s", summary_path)
    return combined


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Walk-forward model research for one or all Top 5 leagues. Does not change production."
    )
    parser.add_argument(
        "--competition",
        default="all",
        help="E0/SP1/D1/I1/F1, or all (default).",
    )
    parser.add_argument("--skip-test", action="store_true", help="Never score the holdout season.")
    args = parser.parse_args()
    codes = parse_competition_list(args.competition)
    if len(codes) == 1:
        run_experiment(competition=codes[0], evaluate_holdout=not args.skip_test)
    else:
        run_all(codes, evaluate_holdout=not args.skip_test)


if __name__ == "__main__":
    main()
