"""Dependency-free constants shared across the pipeline and the dashboard.

This module must not import anything beyond the standard library. The dashboard
container deliberately ships without scikit-learn, so anything it needs to
resolve the production model has to live here rather than in models.py.
"""

FEATURE_VERSION = "v2-draw-aware"

# Algorithm names eligible to be the production model. models.SKLEARN_MODELS
# builds exactly these; a test asserts the two never drift apart.
PRODUCTION_ALGORITHMS = frozenset(
    {
        "logistic_regression",
        "logistic_regression_balanced",
        "random_forest",
        "xgboost",
    }
)
