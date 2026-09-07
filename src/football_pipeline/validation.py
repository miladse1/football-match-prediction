"""Light checks on raw CSV structure. Row-level cleaning belongs in Milestone 2."""

from __future__ import annotations

from football_pipeline.football_data import REQUIRED_COLUMNS


class ValidationError(Exception):
    """CSV header or row is not usable as raw football-data input."""


def assert_header(fieldnames: list[str] | None) -> None:
    if not fieldnames:
        raise ValidationError("CSV has no header row")
    missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
    if missing:
        raise ValidationError(f"CSV missing required columns: {missing}")


def row_is_empty(row: dict[str, str]) -> bool:
    return not any((value or "").strip() for value in row.values())


def assert_result_if_present(row: dict[str, str], row_number: int) -> None:
    result = (row.get("FTR") or "").strip()
    if result and result not in {"H", "D", "A"}:
        raise ValidationError(f"Row {row_number}: FTR must be H, D, or A (got {result!r})")
