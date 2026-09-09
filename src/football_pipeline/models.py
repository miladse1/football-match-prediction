"""Sklearn / XGBoost pipelines. Imputers and scalers fit on train only."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from football_pipeline.constants import PRODUCTION_ALGORITHMS
from football_pipeline.dataset import FEATURE_COLUMNS
from football_pipeline.metrics import pad_proba

H2H_COLUMNS = ("h2h_home_win_rate_n", "h2h_draw_rate_n")
H2H_FILL = 0.5


def fill_h2h(frame):
    """Missing H2H means no prior meetings, not a 0% rate."""
    out = frame.copy()
    for column in H2H_COLUMNS:
        if column in out.columns:
            out[column] = out[column].fillna(H2H_FILL)
    return out


def feature_matrix(frame) -> np.ndarray:
    return fill_h2h(frame)[list(FEATURE_COLUMNS)].to_numpy(dtype=float)


def target_vector(frame) -> np.ndarray:
    return frame["result_code"].to_numpy(dtype=int)


def logreg_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def logreg_balanced_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def random_forest_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=3,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


class XGBThreeClass(BaseEstimator, ClassifierMixin):
    """XGBoost remaps labels to 0..K-1. Keep original 0/1/2 codes for pad_proba."""

    def __init__(self):
        self._encoder = LabelEncoder()
        self._model = XGBClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.1,
            eval_metric="logloss",
            n_jobs=1,
            random_state=42,
            verbosity=0,
        )

    def fit(self, X, y):
        y_enc = self._encoder.fit_transform(y)
        self._model.fit(X, y_enc)
        self.classes_ = self._encoder.classes_
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)


def xgboost_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", XGBThreeClass()),
        ]
    )


SKLEARN_MODELS = {
    "logistic_regression": logreg_pipeline,
    "logistic_regression_balanced": logreg_balanced_pipeline,
    "random_forest": random_forest_pipeline,
    "xgboost": xgboost_pipeline,
}

# The dependency-free name list in constants must stay in step with the
# pipelines built here. test_registry asserts this too.
assert set(SKLEARN_MODELS) == set(PRODUCTION_ALGORITHMS), (
    "SKLEARN_MODELS and constants.PRODUCTION_ALGORITHMS have drifted apart"
)


def predict_proba_3way(estimator, X: np.ndarray) -> np.ndarray:
    raw = estimator.predict_proba(X)
    classes = getattr(estimator, "classes_", None)
    if classes is None and hasattr(estimator, "named_steps"):
        classes = estimator.named_steps["model"].classes_
    if classes is None:
        classes = np.arange(raw.shape[1])
    return pad_proba(raw, np.asarray(classes))
