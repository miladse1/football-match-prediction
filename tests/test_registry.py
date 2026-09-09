"""One production-model lookup, resolved from model_runs rather than guessed on disk."""

import json
from pathlib import Path

import joblib
import pytest

from football_pipeline import registry
from football_pipeline.registry import (
    ProductionModel,
    ProductionModelUnavailable,
    artifact_path_for_run,
    known_algorithms,
    reported_algorithm,
)


def _write_report(tmp_path, payload):
    path = tmp_path / "model_metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_known_algorithms_are_the_buildable_sklearn_pipelines():
    assert known_algorithms() == frozenset(
        {
            "logistic_regression",
            "logistic_regression_balanced",
            "random_forest",
            "xgboost",
        }
    )


def test_poisson_is_not_a_selectable_production_algorithm():
    """The unreachable branch removed from predict.py stays unreachable."""
    assert "poisson_dixon_coles" not in known_algorithms()


def test_reported_algorithm_prefers_the_walkforward_key(tmp_path):
    path = _write_report(
        tmp_path,
        {
            "selected_by_walkforward_log_loss": "logistic_regression",
            "selected_by_valid_log_loss": "xgboost",
        },
    )
    assert reported_algorithm(path) == "logistic_regression"


def test_reported_algorithm_falls_through_legacy_keys(tmp_path):
    path = _write_report(tmp_path, {"selected_sklearn_by_valid_log_loss": "random_forest"})
    assert reported_algorithm(path) == "random_forest"


def test_reported_algorithm_ignores_an_unbuildable_name(tmp_path):
    path = _write_report(tmp_path, {"selected_by_walkforward_log_loss": "poisson_dixon_coles"})
    assert reported_algorithm(path) is None


def test_reported_algorithm_handles_a_missing_or_corrupt_report(tmp_path):
    assert reported_algorithm(tmp_path / "absent.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert reported_algorithm(broken) is None


def test_artifact_path_is_unique_per_run():
    first = artifact_path_for_run("logistic_regression", 7)
    second = artifact_path_for_run("logistic_regression", 8)
    assert first != second
    assert first.name == "logistic_regression-run00007.joblib"
    assert second.name == "logistic_regression-run00008.joblib"


def test_artifact_path_is_stable_for_the_same_run():
    assert artifact_path_for_run("xgboost", 12) == artifact_path_for_run("xgboost", 12)


def test_load_estimator_reads_the_path_recorded_on_the_run(tmp_path):
    """An older run must load its own artifact, not whatever was written last."""
    old_artifact = tmp_path / "logistic_regression-run00001.joblib"
    new_artifact = tmp_path / "logistic_regression-run00002.joblib"
    joblib.dump({"which": "old"}, old_artifact)
    joblib.dump({"which": "new"}, new_artifact)

    old_run = ProductionModel(
        algorithm="logistic_regression",
        model_run_id=1,
        artifact_path=str(old_artifact),
        feature_version="v2-draw-aware",
    )
    assert registry.load_estimator(old_run) == {"which": "old"}


def test_load_estimator_falls_back_to_the_legacy_unversioned_artifact(tmp_path, monkeypatch):
    """Runs trained before artifacts were versioned keep working."""
    monkeypatch.setattr(registry, "MODEL_DIR", tmp_path)
    legacy = tmp_path / "logistic_regression.joblib"
    joblib.dump({"which": "legacy"}, legacy)

    model = ProductionModel(
        algorithm="logistic_regression",
        model_run_id=3,
        artifact_path=str(tmp_path / "missing-run00003.joblib"),
        feature_version="v2-draw-aware",
    )
    assert registry.load_estimator(model) == {"which": "legacy"}


def test_load_estimator_raises_when_nothing_is_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODEL_DIR", tmp_path)
    model = ProductionModel(
        algorithm="logistic_regression",
        model_run_id=4,
        artifact_path=str(tmp_path / "nope.joblib"),
        feature_version="v2-draw-aware",
    )
    with pytest.raises(ProductionModelUnavailable, match="No artifact on disk"):
        registry.load_estimator(model)


def test_production_model_exposes_everything_callers_need():
    model = ProductionModel(
        algorithm="logistic_regression",
        model_run_id=9,
        artifact_path="/tmp/x.joblib",
        feature_version="v2-draw-aware",
    )
    assert model.algorithm == "logistic_regression"
    assert model.model_run_id == 9
    assert model.feature_version == "v2-draw-aware"
    assert model.artifact.name == "x.joblib"


def test_callers_share_one_lookup():
    """predict, the simulator and the dashboard must not re-implement selection."""
    import football_dashboard.queries as queries
    import football_pipeline.predict as predict
    import football_pipeline.season_sim as season_sim

    assert predict.production_model is registry.production_model
    assert season_sim.production_model is registry.production_model
    assert queries._registry_production_model is registry.production_model


def test_registry_does_not_require_sklearn():
    """The dashboard container ships without sklearn; resolving the model must still work."""
    source = Path(registry.__file__).read_text(encoding="utf-8")
    assert "from football_pipeline.models import" not in source
    assert "import sklearn" not in source


def test_algorithm_names_never_drift_from_the_built_pipelines():
    from football_pipeline.constants import PRODUCTION_ALGORITHMS
    from football_pipeline.models import SKLEARN_MODELS

    assert set(SKLEARN_MODELS) == set(PRODUCTION_ALGORITHMS)
    assert known_algorithms() == PRODUCTION_ALGORITHMS
