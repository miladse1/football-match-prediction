from datetime import date

import pandas as pd
import pytest

from football_pipeline import seasons
from football_pipeline.walkforward import (
    PRODUCTION_TRAIN_END,
    build_folds,
    TEST_SEASON_END,
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


def test_walkforward_folds_at_a_fixed_date_match_the_documented_windows():
    """Pinned to a date so it keeps asserting the same thing after the rollover."""
    folds = build_folds(date(2026, 9, 9), ingest_start_year=2018)
    assert len(folds) == 4
    assert [fold.name for fold in folds] == ["2021/22", "2022/23", "2023/24", "2024/25"]
    train_ends = [fold.train_end for fold in folds]
    assert train_ends == [
        date(2021, 7, 31),
        date(2022, 7, 31),
        date(2023, 7, 31),
        date(2024, 7, 31),
    ]
    assert train_ends == sorted(train_ends)
    assert folds[-1].valid_end == date(2025, 7, 31)
    assert folds[-1].valid_end == seasons.production_train_end(date(2026, 9, 9))
    assert seasons.holdout_season_end(date(2026, 9, 9)) == date(2026, 7, 31)


def test_walkforward_folds_are_expanding_and_stop_before_test():
    """Invariants that must hold for the live module constants at any date."""
    assert len(WALKFORWARD_FOLDS) >= 1
    train_ends = [fold.train_end for fold in WALKFORWARD_FOLDS]
    assert train_ends == sorted(train_ends)
    assert len(set(train_ends)) == len(train_ends)
    for fold in WALKFORWARD_FOLDS:
        assert fold.train_end < fold.valid_end
        assert fold.valid_end <= PRODUCTION_TRAIN_END
    assert WALKFORWARD_FOLDS[-1].valid_end == PRODUCTION_TRAIN_END
    assert PRODUCTION_TRAIN_END < TEST_SEASON_END
    assert TEST_SEASON_END == seasons.holdout_season_end()


def test_folds_shift_by_one_season_after_the_august_rollover():
    before = build_folds(date(2027, 7, 31), ingest_start_year=2018)
    after = build_folds(date(2027, 8, 1), ingest_start_year=2018)
    assert [f.name for f in before] == ["2021/22", "2022/23", "2023/24", "2024/25"]
    assert [f.name for f in after] == ["2022/23", "2023/24", "2024/25", "2025/26"]
    assert after[-1].valid_end == date(2026, 7, 31)


def test_fold_slice_is_strictly_after_train_end():
    frame = pd.DataFrame(
        [
            {"match_id": 1, "match_date": date(2021, 5, 23), "result_code": 2},
            {"match_id": 2, "match_date": date(2021, 7, 31), "result_code": 1},
            {"match_id": 3, "match_date": date(2021, 8, 14), "result_code": 0},
            {"match_id": 4, "match_date": date(2022, 5, 22), "result_code": 2},
            {"match_id": 5, "match_date": date(2022, 8, 6), "result_code": 2},
        ]
    )
    fold = WALKFORWARD_FOLDS[0]
    train, valid = fold_frames(frame, fold)
    assert set(train["match_id"]) == {1, 2}
    assert set(valid["match_id"]) == {3, 4}
    assert 5 not in set(valid["match_id"])
    assert pd.to_datetime(train["match_date"]).max().date() <= fold.train_end
    assert pd.to_datetime(valid["match_date"]).min().date() > fold.train_end


def test_2025_26_is_test_only_not_a_fold():
    rows = []
    match_id = 1
    for year in range(2020, 2027):
        rows.append(
            {
                "match_id": match_id,
                "match_date": date(year, 5, 15),
                "result_code": 2,
            }
        )
        match_id += 1
        rows.append(
            {
                "match_id": match_id,
                "match_date": date(year, 8, 20),
                "result_code": 1,
            }
        )
        match_id += 1
    frame = pd.DataFrame(rows)
    assert_folds_exclude_test_season(frame)
    test = slice_season(frame, after=PRODUCTION_TRAIN_END, through=TEST_SEASON_END)
    test_ids = set(test["match_id"])
    assert date(2026, 5, 15) in set(pd.to_datetime(test["match_date"]).dt.date)
    assert date(2026, 8, 20) not in set(pd.to_datetime(test["match_date"]).dt.date)
    for fold in WALKFORWARD_FOLDS:
        _, valid = fold_frames(frame, fold)
        assert test_ids.isdisjoint(set(valid["match_id"]))


def test_select_by_mean_log_loss_not_a_single_fold_win():
    summaries = {
        "xgboost": {
            "folds": 4,
            "mean": {"log_loss": 1.02},
            "median": {"log_loss": 0.99},
            "std": {"log_loss": 0.08},
        },
        "logistic_regression": {
            "folds": 4,
            "mean": {"log_loss": 1.00},
            "median": {"log_loss": 1.00},
            "std": {"log_loss": 0.01},
        },
    }
    selected, reason = select_by_walkforward(summaries)
    assert selected == "logistic_regression"
    assert "xgboost" in reason
    assert "mean" in reason.lower()


def test_aggregate_mean_and_median():
    folds = [
        {
            "log_loss": 1.0,
            "accuracy": 0.5,
            "f1_macro": 0.4,
            "precision_macro": 0.4,
            "recall_macro": 0.4,
            "precision_away": 0.5,
            "precision_draw": 0.0,
            "precision_home": 0.6,
            "recall_away": 0.5,
            "recall_draw": 0.0,
            "recall_home": 0.7,
            "f1_away": 0.5,
            "f1_draw": 0.0,
            "f1_home": 0.65,
            "draw_recall": 0.0,
            "draw_f1": 0.0,
        },
        {
            "log_loss": 1.2,
            "accuracy": 0.4,
            "f1_macro": 0.3,
            "precision_macro": 0.3,
            "recall_macro": 0.3,
            "precision_away": 0.4,
            "precision_draw": 0.2,
            "precision_home": 0.5,
            "recall_away": 0.4,
            "recall_draw": 0.2,
            "recall_home": 0.5,
            "f1_away": 0.4,
            "f1_draw": 0.2,
            "f1_home": 0.5,
            "draw_recall": 0.2,
            "draw_f1": 0.2,
        },
    ]
    summary = aggregate_scalar_metrics(folds)
    assert summary["mean"]["log_loss"] == pytest.approx(1.1)
    assert summary["median"]["log_loss"] == pytest.approx(1.1)
    assert summary["mean"]["draw_recall"] == pytest.approx(0.1)
    assert summary["folds"] == 2


def test_flatten_metrics_exposes_draw_and_confusion():
    metrics = {
        "n": 3,
        "log_loss": 1.0,
        "accuracy": 0.5,
        "f1_macro": 0.4,
        "precision_macro": 0.4,
        "recall_macro": 0.4,
        "precision_by_class": {"away": 0.1, "draw": 0.2, "home": 0.3},
        "recall_by_class": {"away": 0.4, "draw": 0.5, "home": 0.6},
        "f1_by_class": {"away": 0.7, "draw": 0.8, "home": 0.9},
        "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "confusion_matrix_labels": ["away", "draw", "home"],
    }
    flat = flatten_metrics(metrics)
    assert flat["draw_recall"] == 0.5
    assert flat["draw_f1"] == 0.8
    assert flat["confusion_matrix"][1][1] == 1


def test_empty_valid_slice_raises():
    frame = pd.DataFrame(
        [{"match_id": 1, "match_date": date(2020, 5, 1), "result_code": 2}]
    )
    with pytest.raises(WalkForwardError, match="no valid rows"):
        fold_frames(frame, WALKFORWARD_FOLDS[0])


def test_slice_through_includes_end_date():
    frame = pd.DataFrame(
        [
            {"match_id": 1, "match_date": date(2025, 7, 31), "result_code": 2},
            {"match_id": 2, "match_date": date(2025, 8, 1), "result_code": 2},
        ]
    )
    through = slice_through(frame, PRODUCTION_TRAIN_END)
    assert set(through["match_id"]) == {1}
