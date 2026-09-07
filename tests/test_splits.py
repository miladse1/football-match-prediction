from datetime import date

import pytest

from football_pipeline.splits import ChronoSplit, SplitError, assert_splits_are_chronological


def test_dates_fall_into_train_valid_test_in_order():
    split = ChronoSplit(train_end=date(2023, 8, 21), valid_end=date(2023, 8, 26))
    assert split.assign(date(2023, 8, 21)) == "train"
    assert split.assign(date(2023, 8, 22)) == "valid"
    assert split.assign(date(2023, 8, 26)) == "valid"
    assert split.assign(date(2023, 8, 27)) == "test"


def test_train_end_must_precede_valid_end():
    with pytest.raises(SplitError):
        ChronoSplit(train_end=date(2023, 8, 26), valid_end=date(2023, 8, 21))


def test_chrono_guard_rejects_shuffled_labels():
    with pytest.raises(SplitError, match="Train overlaps valid"):
        assert_splits_are_chronological(
            [
                (date(2023, 8, 27), "train"),
                (date(2023, 8, 12), "valid"),
            ]
        )


def test_dates_after_test_end_are_live_only():
    split = ChronoSplit(
        train_end=date(2024, 7, 31),
        valid_end=date(2025, 7, 31),
        test_end=date(2026, 7, 31),
    )
    assert split.assign(date(2024, 5, 19)) == "train"
    assert split.assign(date(2025, 5, 25)) == "valid"
    assert split.assign(date(2026, 5, 24)) == "test"
    assert split.assign(date(2026, 8, 15)) is None


def test_valid_end_must_precede_test_end():
    with pytest.raises(SplitError):
        ChronoSplit(
            train_end=date(2024, 7, 31),
            valid_end=date(2026, 7, 31),
            test_end=date(2025, 7, 31),
        )


def test_chrono_guard_accepts_contiguous_blocks():
    assert_splits_are_chronological(
        [
            (date(2023, 8, 12), "train"),
            (date(2023, 8, 21), "train"),
            (date(2023, 8, 25), "valid"),
            (date(2023, 8, 27), "test"),
        ]
    )
