"""Chronological train / valid / test assignment.

Football results are ordered in time. A random split would put May 2024 in train
and August 2023 in test, which answers the wrong question and can leak
season-level information.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


class SplitError(Exception):
    """Invalid cutoff dates or a row that cannot be labelled."""


@dataclass(frozen=True)
class ChronoSplit:
    train_end: date
    valid_end: date
    test_end: date | None = None

    def __post_init__(self) -> None:
        if self.train_end >= self.valid_end:
            raise SplitError("train_end must be strictly before valid_end")
        if self.test_end is not None and self.valid_end >= self.test_end:
            raise SplitError("valid_end must be strictly before test_end")

    def assign(self, match_date: date) -> str | None:
        """Return train/valid/test, or None for live matches after test_end."""
        if match_date <= self.train_end:
            return "train"
        if match_date <= self.valid_end:
            return "valid"
        if self.test_end is None or match_date <= self.test_end:
            return "test"
        return None


def assert_splits_are_chronological(rows: list[tuple[date, str]]) -> None:
    """Every train date is on or before every valid date, which is on or before every test date."""
    by_split: dict[str, list[date]] = {"train": [], "valid": [], "test": []}
    for match_date, split in rows:
        if split not in by_split:
            raise SplitError(f"Unknown split {split!r}")
        by_split[split].append(match_date)

    train_max = max(by_split["train"], default=None)
    valid_min = min(by_split["valid"], default=None)
    valid_max = max(by_split["valid"], default=None)
    test_min = min(by_split["test"], default=None)

    if train_max is not None and valid_min is not None and train_max > valid_min:
        raise SplitError(f"Train overlaps valid: last train {train_max} after first valid {valid_min}")
    if valid_max is not None and test_min is not None and valid_max > test_min:
        raise SplitError(f"Valid overlaps test: last valid {valid_max} after first test {test_min}")
    if train_max is not None and test_min is not None and train_max > test_min:
        raise SplitError(f"Train overlaps test: last train {train_max} after first test {test_min}")
