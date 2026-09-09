"""Out-of-sample probability calibration for the production logistic regression.

Diagnosis only. Does not write model_runs, predictions, or artifacts used by
the dashboard. Walk-forward folds are refit with the same unweighted logreg
pipeline. The holdout season uses stored production probabilities when present,
otherwise the same production retrain procedure without saving the model.

Season labels are derived from football_pipeline.seasons, not hardcoded.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging

import numpy as np
import pandas as pd

from football_pipeline.config import ROOT
from football_pipeline.dataset import FEATURE_COLUMNS
from football_pipeline.db import connect
from football_pipeline.models import feature_matrix, logreg_pipeline, predict_proba_3way, target_vector
from football_pipeline.train import load_training_frame
from football_pipeline.walkforward import (
    PRODUCTION_TRAIN_END,
    TEST_SEASON_END,
    TEST_SEASON_NAME,
    WALKFORWARD_FOLDS,
    fold_frames,
    slice_season,
    slice_through,
)

logger = logging.getLogger(__name__)

OUT_DIR = ROOT / "data" / "processed"
PREDICTIONS_CSV = OUT_DIR / "calibration_oos_predictions.csv"
BINS_CSV = OUT_DIR / "calibration_bins.csv"
SUMMARY_JSON = OUT_DIR / "calibration_summary.json"
RELIABILITY_SVG = OUT_DIR / "draw_reliability.svg"

CLASS_INDEX = {"away": 0, "draw": 1, "home": 2}
CLASS_NAMES = ("away", "draw", "home")
MIN_BIN_N = 30
BIN_WIDTH = 0.05
# Report key for the holdout block. Derived so it tracks the rolling holdout season.
TEST_SPLIT_KEY = f"test_{TEST_SEASON_NAME.replace('/', '_')}"


def class_brier(y_true: np.ndarray, p_class: np.ndarray, cls: int) -> float:
    indicator = (np.asarray(y_true, dtype=int) == int(cls)).astype(float)
    return float(np.mean((np.asarray(p_class, dtype=float) - indicator) ** 2))


def _combine_bin(left: dict, right: dict) -> dict:
    p = np.concatenate([left["_p"], right["_p"]])
    y = np.concatenate([left["_y"], right["_y"]])
    return {
        "lo": float(left["lo"]),
        "hi": float(right["hi"]),
        "n": int(len(p)),
        "mean_predicted": float(p.mean()),
        "mean_observed": float(y.mean()),
        "_p": p,
        "_y": y,
    }


def reliability_table(
    p: np.ndarray,
    y: np.ndarray,
    *,
    width: float = BIN_WIDTH,
    min_n: int = MIN_BIN_N,
) -> list[dict]:
    """Equal-width bins, adjacent bins merged until each has at least `min_n` rows."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return []
    lo = float(np.floor(p.min() / width) * width)
    hi = float(np.ceil(p.max() / width) * width)
    if hi <= lo:
        hi = lo + width
    edges = np.round(np.arange(lo, hi + width * 0.5, width), 10)
    raw: list[dict] = []
    for i in range(len(edges) - 1):
        left, right = float(edges[i]), float(edges[i + 1])
        last = i == len(edges) - 2
        mask = (p >= left) & ((p <= right) if last else (p < right))
        if not mask.any():
            continue
        raw.append(
            {
                "lo": left,
                "hi": right,
                "n": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()),
                "mean_observed": float(y[mask].mean()),
                "_p": p[mask],
                "_y": y[mask],
            }
        )
    merged: list[dict] = []
    acc: dict | None = None
    for bucket in raw:
        acc = bucket if acc is None else _combine_bin(acc, bucket)
        if acc["n"] >= min_n:
            merged.append(acc)
            acc = None
    if acc is not None:
        if merged:
            merged[-1] = _combine_bin(merged[-1], acc)
        else:
            merged.append(acc)
    out = []
    for bucket in merged:
        pred = float(bucket["mean_predicted"])
        obs = float(bucket["mean_observed"])
        out.append(
            {
                "lo": float(bucket["lo"]),
                "hi": float(bucket["hi"]),
                "n": int(bucket["n"]),
                "mean_predicted": pred,
                "mean_observed": obs,
                "actual_pct": obs,
                "difference": obs - pred,
            }
        )
    return out


def class_summary(y_true: np.ndarray, proba: np.ndarray, cls_name: str) -> dict:
    idx = CLASS_INDEX[cls_name]
    p = proba[:, idx]
    y = (np.asarray(y_true, dtype=int) == idx).astype(float)
    pred_mean = float(p.mean())
    actual = float(y.mean())
    return {
        "class": cls_name,
        "n": int(len(y_true)),
        "mean_predicted": pred_mean,
        "actual_frequency": actual,
        "difference": actual - pred_mean,
        "brier": class_brier(y_true, p, idx),
        "p_min": float(p.min()),
        "p_max": float(p.max()),
        "pct_argmax": float((proba.argmax(axis=1) == idx).mean()),
        "bins": reliability_table(p, y),
    }


def _score_logreg(train: pd.DataFrame, scored: pd.DataFrame) -> np.ndarray:
    pipe = logreg_pipeline()
    pipe.fit(feature_matrix(train), target_vector(train))
    return predict_proba_3way(pipe, feature_matrix(scored))


def _load_production_test_proba(test: pd.DataFrame) -> np.ndarray | None:
    ids = [int(x) for x in test["match_id"].tolist()]
    if not ids:
        return None
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.match_id, p.p_away, p.p_draw, p.p_home
                FROM predictions AS p
                JOIN model_runs AS r ON r.id = p.model_run_id
                WHERE r.algorithm = 'logistic_regression'
                  AND r.artifact_path IS NOT NULL
                  AND p.match_id = ANY(%s)
                ORDER BY r.trained_at DESC
                """,
                (ids,),
            )
            rows = cur.fetchall()
    by_id: dict[int, tuple[float, float, float]] = {}
    for match_id, p_away, p_draw, p_home in rows:
        if int(match_id) not in by_id:
            by_id[int(match_id)] = (float(p_away), float(p_draw), float(p_home))
    if len(by_id) != len(ids):
        return None
    return np.array([by_id[int(mid)] for mid in ids], dtype=float)


def collect_oos_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for fold in WALKFORWARD_FOLDS:
        train, valid = fold_frames(frame, fold)
        proba = _score_logreg(train, valid)
        y = target_vector(valid)
        for row, probs, result in zip(valid.itertuples(index=False), proba, y, strict=True):
            records.append(
                {
                    "split": "walkforward",
                    "fold": fold.name,
                    "match_id": int(row.match_id),
                    "match_date": str(row.match_date),
                    "result_code": int(result),
                    "p_away": float(probs[0]),
                    "p_draw": float(probs[1]),
                    "p_home": float(probs[2]),
                    "predicted_class": int(probs.argmax()),
                }
            )
        logger.info("Walk-forward %s valid n=%s", fold.name, len(valid))

    production_train = slice_through(frame, PRODUCTION_TRAIN_END)
    test = slice_season(frame, after=PRODUCTION_TRAIN_END, through=TEST_SEASON_END)
    stored = _load_production_test_proba(test)
    if stored is not None:
        test_proba = stored
        test_source = "stored_production_predictions"
        logger.info("%s test n=%s from stored production predictions", TEST_SEASON_NAME, len(test))
    else:
        test_proba = _score_logreg(production_train, test)
        test_source = "refit_production_procedure"
        logger.info(
            "%s test n=%s from refit production procedure (no stored rows)",
            TEST_SEASON_NAME,
            len(test),
        )
    y_test = target_vector(test)
    for row, probs, result in zip(test.itertuples(index=False), test_proba, y_test, strict=True):
        records.append(
            {
                "split": TEST_SPLIT_KEY,
                "fold": TEST_SEASON_NAME,
                "match_id": int(row.match_id),
                "match_date": str(row.match_date),
                "result_code": int(result),
                "p_away": float(probs[0]),
                "p_draw": float(probs[1]),
                "p_home": float(probs[2]),
                "predicted_class": int(probs.argmax()),
            }
        )
    out = pd.DataFrame.from_records(records)
    out.attrs["test_source"] = test_source
    return out


def summarize_block(label: str, block: pd.DataFrame) -> dict:
    y = block["result_code"].to_numpy(dtype=int)
    proba = block[["p_away", "p_draw", "p_home"]].to_numpy(dtype=float)
    classes = {name: class_summary(y, proba, name) for name in CLASS_NAMES}
    return {
        "label": label,
        "n": int(len(block)),
        "draw_rate": float((y == 1).mean()),
        "home_rate": float((y == 2).mean()),
        "away_rate": float((y == 0).mean()),
        "pct_argmax_draw": float((block["predicted_class"] == 1).mean()),
        "classes": classes,
    }


def _verdict(draw: dict) -> dict:
    gap = draw["difference"]
    if abs(gap) <= 0.015:
        overall = "well_calibrated"
        overall_text = (
            "Overall Draw probabilities are well calibrated: the mean predicted "
            "Draw chance is within 1.5 percentage points of the actual Draw rate."
        )
    elif gap > 0:
        overall = "underpredicted"
        overall_text = (
            "Draws are systematically underpredicted: actual Draws happen more "
            "often than the mean predicted Draw probability."
        )
    else:
        overall = "overpredicted"
        overall_text = (
            "Draws are systematically overpredicted: the model’s mean Draw "
            "probability is higher than the actual Draw rate."
        )
    return {"label": overall, "text": overall_text, "mean_gap": gap}


def build_report(preds: pd.DataFrame) -> dict:
    wf = preds[preds["split"] == "walkforward"]
    test = preds[preds["split"] == TEST_SPLIT_KEY]
    fold_span = f"{WALKFORWARD_FOLDS[0].name}–{WALKFORWARD_FOLDS[-1].name}"
    walkforward = summarize_block(f"walk-forward validation folds ({fold_span})", wf)
    test_block = summarize_block(f"untouched {TEST_SEASON_NAME} test", test)
    pooled = summarize_block(
        f"pooled out-of-sample (walk-forward + {TEST_SEASON_NAME} test)", preds
    )
    return {
        "model": "logistic_regression",
        "model_label": "Unweighted logistic regression (production)",
        "feature_columns": list(FEATURE_COLUMNS),
        "bin_width": BIN_WIDTH,
        "min_bin_n": MIN_BIN_N,
        "difference_meaning": "actual_frequency minus mean_predicted; positive means more events than predicted (underpredicted)",
        "test_probability_source": preds.attrs.get("test_source"),
        "walkforward": walkforward,
        TEST_SPLIT_KEY: test_block,
        "pooled_oos": pooled,
        "verdict": {
            "walkforward": _verdict(walkforward["classes"]["draw"]),
            TEST_SPLIT_KEY: _verdict(test_block["classes"]["draw"]),
            "pooled_oos": _verdict(pooled["classes"]["draw"]),
        },
        "argmax_note": (
            "A class can be well calibrated without ever being the argmax. "
            "Calibration asks whether P(Draw)=0.22 matches a 22% Draw rate among "
            "those matches. Argmax asks whether Draw is the single most likely "
            "of three outcomes. If Home is 0.48, Draw 0.27, Away 0.25, Draw is "
            "well scored but never selected."
        ),
    }


def _write_predictions_csv(preds: pd.DataFrame) -> None:
    preds.to_csv(PREDICTIONS_CSV, index=False)


def _write_bins_csv(report: dict) -> None:
    rows = []
    for block_key in ("walkforward", TEST_SPLIT_KEY, "pooled_oos"):
        block = report[block_key]
        for cls_name, payload in block["classes"].items():
            for bucket in payload["bins"]:
                rows.append(
                    {
                        "block": block_key,
                        "class": cls_name,
                        "lo": bucket["lo"],
                        "hi": bucket["hi"],
                        "n": bucket["n"],
                        "mean_predicted": bucket["mean_predicted"],
                        "actual_frequency": bucket["mean_observed"],
                        "difference": bucket["difference"],
                    }
                )
    with BINS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _svg_reliability(draw_bins: list[dict], title: str, subtitle: str) -> str:
    width, height = 720, 420
    pad_l, pad_r, pad_t, pad_b = 56, 24, 48, 52
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    xs = [b["mean_predicted"] for b in draw_bins]
    ys = [b["mean_observed"] for b in draw_bins]
    lo = 0.0
    hi = max(0.40, max(xs + ys) + 0.04)

    def sx(v: float) -> float:
        return pad_l + (v - lo) / (hi - lo) * plot_w

    def sy(v: float) -> float:
        return pad_t + (1 - (v - lo) / (hi - lo)) * plot_h

    ticks = [0.0, 0.1, 0.2, 0.3, 0.4]
    grid = []
    for t in ticks:
        x, y = sx(t), sy(t)
        grid.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#d9d9d9" stroke-width="1"/>')
        grid.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{height - pad_b}" stroke="#d9d9d9" stroke-width="1"/>')
        grid.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#555">{t:.0%}</text>')
        grid.append(f'<text x="{x:.1f}" y="{height - pad_b + 18}" text-anchor="middle" font-size="11" fill="#555">{t:.0%}</text>')
    diag = f'<line x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" y2="{sy(hi):.1f}" stroke="#888" stroke-dasharray="5 4" stroke-width="1.5"/>'
    poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
    points = []
    for b in draw_bins:
        points.append(
            f'<circle cx="{sx(b["mean_predicted"]):.1f}" cy="{sy(b["mean_observed"]):.1f}" r="5" fill="#1f4e79"/>'
            f'<text x="{sx(b["mean_predicted"]):.1f}" y="{sy(b["mean_observed"]) - 10:.1f}" text-anchor="middle" font-size="10" fill="#333">n={b["n"]}</text>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{width / 2}" y="22" text-anchor="middle" font-size="16" font-family="system-ui, sans-serif" fill="#111">{title}</text>
  <text x="{width / 2}" y="40" text-anchor="middle" font-size="11" font-family="system-ui, sans-serif" fill="#555">{subtitle}</text>
  {''.join(grid)}
  {diag}
  <polyline fill="none" stroke="#1f4e79" stroke-width="2" points="{poly}"/>
  {''.join(points)}
  <text x="{width / 2}" y="{height - 8}" text-anchor="middle" font-size="12" fill="#333">Mean predicted Draw probability</text>
  <text x="16" y="{height / 2}" text-anchor="middle" font-size="12" fill="#333" transform="rotate(-90 16 {height / 2})">Actual Draw frequency</text>
</svg>
"""


def write_artifacts(preds: pd.DataFrame, report: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_predictions_csv(preds)
    _write_bins_csv(report)
    SUMMARY_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pooled_draw = report["pooled_oos"]["classes"]["draw"]["bins"]
    RELIABILITY_SVG.write_text(
        _svg_reliability(
            pooled_draw,
            "Draw reliability — unweighted logistic regression",
            f"Walk-forward {WALKFORWARD_FOLDS[0].name}–{WALKFORWARD_FOLDS[-1].name} plus untouched "
            f"{TEST_SEASON_NAME} test. Dashed line is perfect calibration.",
        ),
        encoding="utf-8",
    )


def run() -> dict:
    frame = load_training_frame()
    preds = collect_oos_predictions(frame)
    report = build_report(preds)
    write_artifacts(preds, report)
    logger.info("Wrote %s, %s, %s, %s", PREDICTIONS_CSV, BINS_CSV, SUMMARY_JSON, RELIABILITY_SVG)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="OOS calibration diagnostics for production logistic regression.")
    parser.parse_args()
    report = run()
    draw = report["pooled_oos"]["classes"]["draw"]
    print(
        json.dumps(
            {
                "pooled_n": report["pooled_oos"]["n"],
                "mean_predicted_draw": draw["mean_predicted"],
                "actual_draw_rate": draw["actual_frequency"],
                "difference": draw["difference"],
                "draw_brier": draw["brier"],
                "pct_argmax_draw": report["pooled_oos"]["pct_argmax_draw"],
                "verdict": report["verdict"]["pooled_oos"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
