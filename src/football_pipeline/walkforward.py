"""Expanding-window walk-forward folds for out-of-time model selection.

Season boundaries use 31 July and are derived from today's date by
football_pipeline.seasons, so the windows roll forward on 1 August each year
instead of being pinned to one set of seasons.

The invariant that matters is unchanged: the holdout season is never a
validation fold, and the live season is never in training_rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from football_pipeline import seasons
from football_pipeline.config import INGEST_START_YEAR
from football_pipeline.metrics import CLASS_NAMES


class WalkForwardError(Exception):
    """Invalid fold dates or empty train/valid slice."""


def season_end(start_year: int) -> date:
    """Inclusive last date of a season that starts in `start_year` (e.g. 2020 → 2021-07-31)."""
    return seasons.season_end(start_year)


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_through_start_year: int
    valid_start_year: int

    @property
    def train_end(self) -> date:
        return season_end(self.train_through_start_year)

    @property
    def valid_end(self) -> date:
        return season_end(self.valid_start_year)

    def __post_init__(self) -> None:
        if self.valid_start_year != self.train_through_start_year + 1:
            raise WalkForwardError(
                f"{self.name}: valid season must be the next season after train_through"
            )
        if self.train_end >= self.valid_end:
            raise WalkForwardError(f"{self.name}: train_end must precede valid_end")


def build_folds(
    today: date | None = None,
    *,
    n_folds: int = seasons.DEFAULT_WALKFORWARD_FOLDS,
    ingest_start_year: int = INGEST_START_YEAR,
) -> tuple[WalkForwardFold, ...]:
    """Expanding-window folds ending immediately before the holdout season.

    Train through 2020/21 → valid 2021/22, …, train through 2023/24 → valid
    2024/25, when the holdout is 2025/26. The whole sequence shifts by one
    season on 1 August each year.
    """
    valid_years = seasons.walkforward_valid_years(
        today, n_folds=n_folds, ingest_start_year=ingest_start_year
    )
    return tuple(
        WalkForwardFold(
            seasons.short_season_name(year),
            train_through_start_year=year - 1,
            valid_start_year=year,
        )
        for year in valid_years
    )


# Derived at import from today's date. Retrain on everything through the season
# before the holdout; the holdout itself is scored once, after selection.
WALKFORWARD_FOLDS = build_folds()
PRODUCTION_TRAIN_END = seasons.production_train_end()
TEST_SEASON_END = seasons.holdout_season_end()
TEST_SEASON_NAME = seasons.holdout_season_name()

SCALAR_METRICS = (
    "log_loss",
    "accuracy",
    "f1_macro",
    "precision_macro",
    "recall_macro",
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
)


def flatten_metrics(metrics: dict) -> dict:
    """Scalar metrics plus confusion matrix for JSON/CSV. Nested evaluate_split payload."""
    precision = metrics["precision_by_class"]
    recall = metrics["recall_by_class"]
    f1 = metrics["f1_by_class"]
    return {
        "n": metrics["n"],
        "log_loss": metrics["log_loss"],
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "precision_away": precision["away"],
        "precision_draw": precision["draw"],
        "precision_home": precision["home"],
        "recall_away": recall["away"],
        "recall_draw": recall["draw"],
        "recall_home": recall["home"],
        "f1_away": f1["away"],
        "f1_draw": f1["draw"],
        "f1_home": f1["home"],
        "draw_recall": recall["draw"],
        "draw_f1": f1["draw"],
        "confusion_matrix": metrics["confusion_matrix"],
        "confusion_matrix_labels": metrics.get("confusion_matrix_labels", list(CLASS_NAMES.values())),
    }


def slice_through(frame: pd.DataFrame, end: date) -> pd.DataFrame:
    dates = pd.to_datetime(frame["match_date"]).dt.date
    return frame.loc[dates <= end].reset_index(drop=True)


def slice_season(frame: pd.DataFrame, *, after: date, through: date) -> pd.DataFrame:
    dates = pd.to_datetime(frame["match_date"]).dt.date
    return frame.loc[(dates > after) & (dates <= through)].reset_index(drop=True)


def fold_frames(frame: pd.DataFrame, fold: WalkForwardFold) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = slice_through(frame, fold.train_end)
    valid = slice_season(frame, after=fold.train_end, through=fold.valid_end)
    if train.empty:
        raise WalkForwardError(f"{fold.name}: no train rows through {fold.train_end}")
    if valid.empty:
        raise WalkForwardError(f"{fold.name}: no valid rows {fold.train_end} < date <= {fold.valid_end}")
    train_max = pd.to_datetime(train["match_date"]).max().date()
    valid_min = pd.to_datetime(valid["match_date"]).min().date()
    if train_max > valid_min:
        raise WalkForwardError(f"{fold.name}: train overlaps valid ({train_max} / {valid_min})")
    return train, valid


def assert_folds_exclude_test_season(frame: pd.DataFrame) -> None:
    """Walk-forward valid windows must not reach into the holdout season or later."""
    latest_valid = WALKFORWARD_FOLDS[-1].valid_end
    if latest_valid > PRODUCTION_TRAIN_END:
        raise WalkForwardError("Last walk-forward valid season reaches into the test holdout")
    dates = pd.to_datetime(frame["match_date"]).dt.date
    test_rows = frame.loc[(dates > PRODUCTION_TRAIN_END) & (dates <= TEST_SEASON_END)]
    for fold in WALKFORWARD_FOLDS:
        _, valid = fold_frames(frame, fold)
        overlap = set(valid["match_id"]).intersection(set(test_rows["match_id"]))
        if overlap:
            raise WalkForwardError(
                f"{fold.name} includes {len(overlap)} {TEST_SEASON_NAME} test matches"
            )


def aggregate_scalar_metrics(fold_flat: list[dict]) -> dict:
    summary: dict[str, dict[str, float]] = {"mean": {}, "median": {}, "std": {}, "min": {}, "max": {}}
    for key in SCALAR_METRICS:
        values = np.array([row[key] for row in fold_flat], dtype=float)
        summary["mean"][key] = float(np.mean(values))
        summary["median"][key] = float(np.median(values))
        summary["std"][key] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary["min"][key] = float(np.min(values))
        summary["max"][key] = float(np.max(values))
    summary["folds"] = len(fold_flat)
    return summary


def select_by_walkforward(summaries: dict[str, dict]) -> tuple[str, str]:
    """Lowest mean OOT log loss, with median/std as consistency tie-breakers.

    A single-season win does not override a worse mean log loss.
    """
    if not summaries:
        raise WalkForwardError("No walk-forward summaries to select from")

    def sort_key(name: str) -> tuple[float, float, float]:
        block = summaries[name]
        return (
            block["mean"]["log_loss"],
            block["median"]["log_loss"],
            block["std"]["log_loss"],
        )

    ranked = sorted(summaries, key=sort_key)
    winner = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    win_mean = summaries[winner]["mean"]["log_loss"]
    win_median = summaries[winner]["median"]["log_loss"]
    reason = (
        f"Selected {winner} for the lowest mean walk-forward log loss "
        f"({win_mean:.4f}; median {win_median:.4f}) across {summaries[winner]['folds']} "
        f"out-of-time seasons. Macro-F1 and draw F1/recall are diagnostics only."
    )
    if runner:
        other_mean = summaries[runner]["mean"]["log_loss"]
        reason += (
            f" Next was {runner} at mean log loss {other_mean:.4f}. "
            "Selection uses all folds, not the last validation season alone."
        )
    return winner, reason
