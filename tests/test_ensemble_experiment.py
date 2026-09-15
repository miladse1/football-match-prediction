import numpy as np

from football_pipeline.ensemble_experiment import (
    EXPERIMENT_DIR,
    FIXED_ENSEMBLES,
    INVLL_NAME,
    PRODUCTION_REPORT,
    REPORT_PATH,
    inverse_logloss_weights,
    is_complex_candidate,
    mix_proba,
    recommend_action,
    report_path_for,
)
from football_pipeline.metrics import evaluate_split
from football_pipeline.registry import report_path_for as production_report_path


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
    assert REPORT_PATH != PRODUCTION_REPORT
    assert "experiments" in str(REPORT_PATH)
    assert EXPERIMENT_DIR in REPORT_PATH.parents


def test_per_league_reports_stay_under_experiments_and_apart():
    e0 = report_path_for("E0")
    sp1 = report_path_for("SP1")
    assert e0 != sp1
    assert e0.parent != sp1.parent
    for path, code in ((e0, "E0"), (sp1, "SP1"), (report_path_for("D1"), "D1")):
        assert "experiments" in path.parts
        assert path.name == "ensemble_walkforward.json"
        assert path.parts[-2] == code
        assert path.resolve() != production_report_path(code).resolve()


def test_fixed_ensembles_cover_requested_blends():
    members = {name: frozenset(weights) for name, weights in FIXED_ENSEMBLES.items()}
    assert frozenset({"logistic_regression", "xgboost"}) in members.values()
    assert frozenset({"logistic_regression", "poisson_dixon_coles"}) in members.values()
    assert frozenset({"xgboost", "poisson_dixon_coles"}) in members.values()
    assert frozenset({"logistic_regression", "xgboost", "poisson_dixon_coles"}) in members.values()
    assert INVLL_NAME.startswith("ensemble_")
    assert is_complex_candidate(INVLL_NAME)
    assert is_complex_candidate("poisson_dixon_coles")
    assert not is_complex_candidate("logistic_regression")
    assert not is_complex_candidate("xgboost")


def test_mix_proba_three_models_equal_weights():
    a = np.array([[0.6, 0.2, 0.2]])
    b = np.array([[0.2, 0.6, 0.2]])
    c = np.array([[0.2, 0.2, 0.6]])
    mixed = mix_proba({"a": a, "b": b, "c": c}, {"a": 1.0, "b": 1.0, "c": 1.0})
    np.testing.assert_allclose(mixed, [[1 / 3, 1 / 3, 1 / 3]])


def test_recommend_keep_when_walkforward_agrees_with_production():
    decision = recommend_action(
        production="logistic_regression",
        winner="logistic_regression",
        wf_production=0.98,
        wf_winner=0.98,
        holdout_production=1.01,
        holdout_winner=1.01,
    )
    assert decision["action"] == "KEEP CURRENT PRODUCTION"


def test_recommend_insufficient_for_la_liga_scale_xgboost_gap():
    """SP1 production XGBoost vs LR was ~0.0002. That is fold noise, not a switch."""
    decision = recommend_action(
        production="logistic_regression",
        winner="xgboost",
        wf_production=0.9954,
        wf_winner=0.9952,
        holdout_production=1.0100,
        holdout_winner=1.0098,
    )
    assert decision["action"] == "INSUFFICIENT IMPROVEMENT"
    assert decision["wf_log_loss_gain"] < 0.005


def test_recommend_insufficient_when_ensemble_gain_is_small():
    decision = recommend_action(
        production="logistic_regression",
        winner="ensemble_lr_xgb_50_50",
        wf_production=0.980,
        wf_winner=0.972,
        holdout_production=1.000,
        holdout_winner=0.993,
    )
    assert decision["action"] == "INSUFFICIENT IMPROVEMENT"


def test_recommend_switch_only_for_clear_simple_model_gain():
    decision = recommend_action(
        production="logistic_regression",
        winner="xgboost",
        wf_production=1.000,
        wf_winner=0.990,
        holdout_production=1.020,
        holdout_winner=1.010,
    )
    assert decision["action"] == "SWITCH TO NEW MODEL"


def test_recommend_insufficient_when_holdout_does_not_confirm():
    decision = recommend_action(
        production="logistic_regression",
        winner="xgboost",
        wf_production=1.000,
        wf_winner=0.990,
        holdout_production=1.000,
        holdout_winner=1.004,
    )
    assert decision["action"] == "INSUFFICIENT IMPROVEMENT"

