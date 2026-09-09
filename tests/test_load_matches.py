"""An incomplete upstream feed must never delete scored or predicted history."""

from datetime import date
from pathlib import Path

from football_pipeline.load_matches import partition_stale_matches


def _row(match_id, *, is_played=False, prediction_count=0, day=1):
    return {
        "id": match_id,
        "match_date": date(2026, 9, day),
        "is_played": is_played,
        "prediction_count": prediction_count,
    }


def test_unplayed_never_predicted_rows_are_deletable():
    deletable, protected = partition_stale_matches([_row(1), _row(2)])
    assert deletable == [1, 2]
    assert protected == []


def test_played_match_survives_a_missing_payload():
    deletable, protected = partition_stale_matches([_row(1, is_played=True)])
    assert deletable == []
    assert [row["id"] for row in protected] == [1]
    assert "already played" in protected[0]["reasons"]


def test_predicted_match_survives_a_missing_payload():
    """The frozen-prediction guarantee is the whole point of this guard."""
    deletable, protected = partition_stale_matches([_row(2, prediction_count=1)])
    assert deletable == []
    assert [row["id"] for row in protected] == [2]
    assert "1 stored prediction(s)" in protected[0]["reasons"]


def test_played_and_predicted_match_reports_both_reasons():
    deletable, protected = partition_stale_matches(
        [_row(3, is_played=True, prediction_count=2)]
    )
    assert deletable == []
    assert protected[0]["reasons"] == ["already played", "2 stored prediction(s)"]


def test_mixed_batch_keeps_history_and_drops_only_placeholders():
    candidates = [
        _row(1, is_played=True),
        _row(2, prediction_count=3),
        _row(3),
        _row(4, is_played=True, prediction_count=1),
        _row(5),
    ]
    deletable, protected = partition_stale_matches(candidates)
    assert deletable == [3, 5]
    assert sorted(row["id"] for row in protected) == [1, 2, 4]


def test_empty_candidate_list_is_a_no_op():
    assert partition_stale_matches([]) == ([], [])


def test_null_prediction_count_is_treated_as_zero():
    deletable, protected = partition_stale_matches(
        [{"id": 9, "match_date": date(2026, 9, 1), "is_played": False, "prediction_count": None}]
    )
    assert deletable == [9]
    assert protected == []


def test_loader_no_longer_bulk_deletes_predictions():
    """The old code deleted prediction rows before deleting matches."""
    source = Path("src/football_pipeline/load_matches.py").read_text(encoding="utf-8")
    assert "DELETE FROM predictions" not in source
    assert "partition_stale_matches" in source


def test_loader_reports_preserved_rows_to_the_caller():
    source = Path("src/football_pipeline/load_matches.py").read_text(encoding="utf-8")
    assert '"stale_protected"' in source
