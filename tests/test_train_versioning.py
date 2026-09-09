"""Artifacts are versioned per run, and identical retrains reuse their run."""

from datetime import date
from pathlib import Path

import pandas as pd

from football_pipeline.registry import artifact_path_for_run
from football_pipeline.train import CANDIDATE_MODELS, PRODUCTION_MODELS, run_fingerprint


def _frame(n, start_day=1):
    return pd.DataFrame(
        [
            {
                "match_id": i,
                "match_date": date(2024, 8, min(start_day + i, 28)),
                "result_code": i % 3,
            }
            for i in range(n)
        ]
    )


def _fingerprint(**overrides):
    kwargs = {
        "algorithm": "logistic_regression",
        "feature_version": "v2-draw-aware",
        "train": _frame(10),
        "test": _frame(5),
        "walkforward_mean_log_loss": 0.973,
        "test_log_loss": 1.029,
    }
    kwargs.update(overrides)
    return run_fingerprint(**kwargs)


def test_identical_training_inputs_produce_the_same_fingerprint():
    assert _fingerprint() == _fingerprint()


def test_new_training_rows_change_the_fingerprint():
    """A new match must create a genuinely new run, not reuse the old one."""
    assert _fingerprint() != _fingerprint(train=_frame(11))


def test_new_test_rows_change_the_fingerprint():
    assert _fingerprint() != _fingerprint(test=_frame(6))


def test_a_different_algorithm_changes_the_fingerprint():
    assert _fingerprint() != _fingerprint(algorithm="xgboost")


def test_a_different_feature_version_changes_the_fingerprint():
    assert _fingerprint() != _fingerprint(feature_version="v3-something")


def test_changed_scores_change_the_fingerprint():
    assert _fingerprint() != _fingerprint(test_log_loss=1.030)
    assert _fingerprint() != _fingerprint(walkforward_mean_log_loss=0.974)


def test_shifted_dates_change_the_fingerprint():
    assert _fingerprint() != _fingerprint(train=_frame(10, start_day=2))


def test_artifact_paths_do_not_collide_across_runs():
    paths = {artifact_path_for_run("logistic_regression", i) for i in range(1, 50)}
    assert len(paths) == 49


def test_training_writes_a_run_scoped_artifact_not_a_shared_one():
    source = Path("src/football_pipeline/train.py").read_text(encoding="utf-8")
    assert 'MODEL_DIR / f"{selected}.joblib"' not in source
    assert "artifact_path_for_run(selected, run_id)" in source


def test_training_reuses_an_identical_run_instead_of_duplicating_rows():
    source = Path("src/football_pipeline/train.py").read_text(encoding="utf-8")
    assert "_find_reusable_run" in source
    assert "run_fingerprint" in source
    # Holdout predictions are only stored on a genuinely new run.
    insert_branch = source.split("reusable = _find_reusable_run")[1]
    assert "_store_predictions(run_id, test, test_proba)" in insert_branch


def test_candidate_and_production_model_lists_are_not_duplicated():
    assert set(PRODUCTION_MODELS) == set(CANDIDATE_MODELS)
    assert "logistic_regression" in PRODUCTION_MODELS


def test_production_model_choice_is_unchanged():
    """This refactor must not change which models compete or which one wins."""
    assert sorted(CANDIDATE_MODELS) == [
        "logistic_regression",
        "logistic_regression_balanced",
        "random_forest",
        "xgboost",
    ]
