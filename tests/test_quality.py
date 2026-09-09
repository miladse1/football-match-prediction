"""Bad data must stop the pipeline instead of reaching model training."""

import pytest

from football_pipeline.quality import (
    DataQualityError,
    _fail,
    summary_problems,
)

LIVE_YEAR = 2026


def _season(start_year, matches=380, played=None, unplayed=0):
    played = matches if played is None else played
    return {
        "start_year": start_year,
        "season": f"{start_year}/{start_year + 1}",
        "matches": matches,
        "played": played,
        "unplayed": unplayed,
    }


def _summary(seasons, played, draws=None):
    draws = played // 4 if draws is None else draws
    remaining = played - draws
    return {
        "seasons": seasons,
        "played": played,
        "unplayed": 0,
        "matches": played,
        "result_code_played": {
            "away": remaining // 2,
            "draw": draws,
            "home": remaining - remaining // 2,
        },
    }


def test_healthy_history_passes():
    summary = _summary(
        [_season(2024), _season(2025), _season(2026, matches=30, played=30)],
        played=790,
    )
    assert summary_problems(summary, LIVE_YEAR) == []


def test_short_completed_season_is_rejected():
    summary = _summary([_season(2024, matches=379, played=379)], played=379)
    problems = summary_problems(summary, LIVE_YEAR)
    assert any("379 matches, expected 380" in p for p in problems)


def test_completed_season_with_unplayed_matches_is_rejected():
    summary = _summary([_season(2024, matches=380, played=370, unplayed=10)], played=370)
    problems = summary_problems(summary, LIVE_YEAR)
    assert any("still has 10 unplayed" in p for p in problems)


def test_partial_live_season_is_allowed():
    summary = _summary([_season(2026, matches=30, played=30)], played=30)
    assert summary_problems(summary, LIVE_YEAR) == []


def test_oversized_live_season_is_rejected():
    summary = _summary([_season(2026, matches=400, played=400)], played=400)
    problems = summary_problems(summary, LIVE_YEAR)
    assert any("more than a full season" in p for p in problems)


def test_empty_ingest_is_rejected():
    problems = summary_problems({"seasons": [], "played": 0}, LIVE_YEAR)
    assert any("No seasons found" in p for p in problems)
    assert any("No played matches" in p for p in problems)


def test_played_matches_without_result_codes_are_rejected():
    summary = _summary([_season(2024)], played=380)
    summary["result_code_played"] = {"away": 100, "draw": 90, "home": 100}
    problems = summary_problems(summary, LIVE_YEAR)
    assert any("only 290 carry a result code" in p for p in problems)


def test_implausible_draw_rate_is_rejected():
    summary = _summary([_season(2024)], played=380, draws=5)
    problems = summary_problems(summary, LIVE_YEAR)
    assert any("draw rate" in p for p in problems)

    summary = _summary([_season(2024)], played=380, draws=300)
    problems = summary_problems(summary, LIVE_YEAR)
    assert any("draw rate" in p for p in problems)


def test_fail_raises_with_every_problem_listed():
    with pytest.raises(DataQualityError) as exc:
        _fail(["first thing", "second thing"], "load_core_tables")
    message = str(exc.value)
    assert "load_core_tables failed 2 data quality check(s)" in message
    assert "first thing" in message
    assert "second thing" in message


def test_fail_is_silent_when_there_are_no_problems():
    assert _fail([], "load_core_tables") is None


def test_gates_are_wired_into_the_pipeline_tasks():
    """Every gate must actually run inside an Airflow task, not just exist."""
    from pathlib import Path

    tasks = Path("src/football_pipeline/tasks.py").read_text(encoding="utf-8")
    for gate in ("check_matches", "check_features", "check_training_rows", "check_predictions"):
        assert gate in tasks, f"{gate} is not wired into tasks.py"
