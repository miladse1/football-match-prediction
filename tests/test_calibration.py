import numpy as np

from football_pipeline.calibration import class_brier, reliability_table


def test_class_brier_is_zero_when_probability_matches_outcome():
    y = np.array([1, 1, 0, 0])
    p = np.array([1.0, 1.0, 0.0, 0.0])
    assert class_brier(y, p, cls=1) == 0.0


def test_reliability_table_merges_tiny_bins():
    rng = np.random.default_rng(0)
    p = np.concatenate(
        [
            rng.uniform(0.10, 0.15, 8),
            rng.uniform(0.15, 0.20, 40),
            rng.uniform(0.20, 0.25, 40),
        ]
    )
    y = (rng.random(len(p)) < p).astype(float)
    bins = reliability_table(p, y, width=0.05, min_n=30)
    assert all(bucket["n"] >= 30 for bucket in bins)
    assert abs(sum(bucket["n"] for bucket in bins) - len(p)) < 1
    for bucket in bins:
        assert "difference" in bucket
        assert bucket["hi"] > bucket["lo"]
