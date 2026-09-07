import numpy as np

from football_pipeline.poisson import DixonColesPoisson, matrix_to_1x2, score_matrix


def test_score_matrix_1x2_sums_to_one():
    proba = matrix_to_1x2(score_matrix(1.4, 1.1, -0.05))
    np.testing.assert_allclose(proba.sum(), 1.0, atol=1e-9)
    assert (proba >= 0).all()


def test_poisson_fit_uses_train_only_and_outputs_3way():
    model = DixonColesPoisson()
    home = [1, 1, 2, 2, 3]
    away = [2, 3, 1, 3, 1]
    hg = [2, 1, 0, 1, 3]
    ag = [0, 1, 1, 1, 1]
    model.fit(home, away, hg, ag)
    proba = model.predict_proba([1, 2], [2, 3])
    assert proba.shape == (2, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert model.rho_ in set(np.round(np.linspace(-0.15, 0.15, 13), 4))
