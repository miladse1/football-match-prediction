import numpy as np

from football_pipeline.ensemble_experiment import (
    PRODUCTION_REPORT,
    inverse_logloss_weights,
    mix_proba,
)
from football_pipeline.metrics import evaluate_split


def test_mix_proba_is_probability_average_not_vote():
    a = np.array([[0.05, 0.40, 0.55]])
    b = np.array([[0.55, 0.40, 0.05]])
    mixed = mix_proba({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
    np.testing.assert_allclose(mixed, [[0.30, 0.40, 0.30]])
    np.testing.assert_allclose(mixed.sum(axis=1), 1.0)
    # Hard labels disagree (home vs away); the blend's mode is draw.
    assert int(a[0].argmax()) == 2
    assert int(b[0].argmax()) == 0
    assert int(mixed[0].argmax()) == 1


def test_mix_proba_renormalizes_positive_weights():
    a = np.array([[0.1, 0.2, 0.7]])
    b = np.array([[0.7, 0.2, 0.1]])
    mixed = mix_proba({"a": a, "b": b}, {"a": 2.0, "b": 1.0})
    expected = (2 * a + b) / 3
    np.testing.assert_allclose(mixed, expected)
    np.testing.assert_allclose(mixed.sum(axis=1), 1.0)


def test_inverse_logloss_weights_leave_one_fold_out():
    losses = {
        "logistic_regression": [0.90, 0.90, 0.90, 1.20],
        "xgboost": [1.10, 1.10, 1.10, 0.80],
    }
    last = inverse_logloss_weights(losses, 3, ("logistic_regression", "xgboost"))
    # Last fold is excluded, so logistic (0.90) gets more weight than xgboost (1.10).
    assert last["logistic_regression"] > last["xgboost"]
    np.testing.assert_allclose(sum(last.values()), 1.0)
    first = inverse_logloss_weights(losses, 0, ("logistic_regression", "xgboost"))
    np.testing.assert_allclose(sum(first.values()), 1.0)


def test_evaluate_split_reports_class_brier():
    y = np.array([2, 1, 0])
    proba = np.array([[0.1, 0.1, 0.8], [0.2, 0.6, 0.2], [0.7, 0.2, 0.1]])
    metrics = evaluate_split(y, proba)
    assert set(metrics["brier_by_class"]) == {"away", "draw", "home"}
    assert metrics["brier_by_class"]["home"] >= 0


def test_experiment_output_path_is_not_production_report():
    from football_pipeline.ensemble_experiment import REPORT_PATH

    assert REPORT_PATH != PRODUCTION_REPORT
    assert "experiments" in str(REPORT_PATH)
