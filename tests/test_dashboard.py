from football_dashboard.formatters import (
    algorithm_label,
    as_percent,
    group_by_date,
    match_note,
    paginate,
    predicted_outcome,
)
from football_dashboard.queries import live_scorecard


def test_as_percent_rounds_to_readable_whole_numbers():
    assert as_percent(0.55) == 55
    assert as_percent(0.214) == 21
    assert as_percent(0.216) == 22


def test_predicted_outcome_uses_team_names():
    assert predicted_outcome(2, "Arsenal", "Chelsea") == "Arsenal"
    assert predicted_outcome(0, "Arsenal", "Chelsea") == "Chelsea"
    assert predicted_outcome(1, "Arsenal", "Chelsea") == "Draw"


def test_algorithm_label_for_production_model():
    assert algorithm_label("logistic_regression") == "Logistic regression"


def test_draw_watch_does_not_change_predicted_class():
    assert match_note(0.4, 0.32, 0.28, predicted_class=2) == "Draw watch"
    assert predicted_outcome(2, "Arsenal", "Chelsea") == "Arsenal"
    assert match_note(0.55, 0.12, 0.33, predicted_class=2) is None


def test_close_match_when_top_two_outcomes_are_tight():
    assert match_note(0.38, 0.22, 0.4, predicted_class=0) == "Close match"


def test_paginate_keeps_pages_in_the_requested_band():
    items = list(range(360))
    page = paginate(items, page=2, page_size=16)
    assert page["page"] == 2
    assert page["page_size"] == 16
    assert page["total"] == 360
    assert page["pages"] == 23
    assert page["items"] == list(range(16, 32))
    assert paginate(items, page=1, page_size=50)["page_size"] == 20
    assert paginate(items, page=1, page_size=3)["page_size"] == 10


def test_group_by_date_keeps_fixtures_on_the_same_day_together():
    groups = group_by_date(
        [
            {"kickoff_date": "2026-09-12", "home_team": "Arsenal"},
            {"kickoff_date": "2026-09-12", "home_team": "Chelsea"},
            {"kickoff_date": "2026-09-13", "home_team": "Liverpool"},
        ]
    )
    assert [group["date"] for group in groups] == ["2026-09-12", "2026-09-13"]
    assert len(groups[0]["fixtures"]) == 2


def test_live_scorecard_is_empty_until_settled_rows_exist():
    empty = live_scorecard([])
    assert empty["n"] == 0
    assert empty["accuracy"] is None
    assert empty["by_actual"] == {}


def test_live_scorecard_accuracy_log_loss_and_class_slices():
    rows = [
        {
            "result_code": 2,
            "predicted_class": 2,
            "p_away": 0.2,
            "p_draw": 0.2,
            "p_home": 0.6,
        },
        {
            "result_code": 1,
            "predicted_class": 2,
            "p_away": 0.2,
            "p_draw": 0.3,
            "p_home": 0.5,
        },
    ]
    score = live_scorecard(rows)
    assert score["n"] == 2
    assert score["correct"] == 1
    assert score["accuracy"] == 0.5
    assert score["log_loss"] > 0
    assert score["by_actual"]["home"]["n"] == 1
    assert score["by_actual"]["home"]["correct"] == 1
    assert score["by_actual"]["draw"]["correct"] == 0
    assert score["by_predicted"]["home"]["n"] == 2
    assert score["by_predicted"]["draw"]["n"] == 0
