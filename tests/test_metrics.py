import numpy as np

from football_pipeline.metrics import evaluate_split, pad_proba, uniform_proba


def test_pad_proba_inserts_missing_draw_column():
    proba = np.array([[0.7, 0.3], [0.2, 0.8]])
    padded = pad_proba(proba, np.array([0, 2]))
    assert padded.shape == (2, 3)
    assert padded[0, 1] == 0.0
    np.testing.assert_allclose(padded.sum(axis=1), 1.0)


def test_uniform_log_loss_is_log_three():
    y = np.array([0, 1, 2])
    metrics = evaluate_split(y, uniform_proba(3))
    np.testing.assert_allclose(metrics["log_loss"], np.log(3.0), rtol=1e-6)
    assert metrics["accuracy"] == 1.0 / 3.0


def test_perfect_home_predictions():
    y = np.array([2, 2, 2])
    proba = np.array([[0.05, 0.05, 0.90]] * 3)
    metrics = evaluate_split(y, proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix"][2][2] == 3
    assert metrics["draw_proba"]["pct_argmax_draw"] == 0.0
    assert metrics["brier"] < 0.05
