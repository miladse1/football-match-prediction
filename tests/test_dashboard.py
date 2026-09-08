from football_dashboard.crests import team_badge, team_initials
from football_dashboard.formatters import (
    algorithm_label,
    as_percent,
    extract_match_stats,
    filter_prior_h2h,
    group_by_date,
    match_note,
    ordinal,
    paginate,
    predicted_outcome,
)
from football_dashboard.queries import _serialize_fixture, live_scorecard


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


def test_ordinal_rank_labels():
    assert ordinal(1) == "1st"
    assert ordinal(2) == "2nd"
    assert ordinal(3) == "3rd"
    assert ordinal(11) == "11th"
    assert ordinal(12) == "12th"
    assert ordinal(21) == "21st"


def test_team_badge_uses_public_crest_and_initials_fallback():
    badge = team_badge("Manchester City")
    assert badge["crest_url"].endswith("/65.png")
    assert team_initials("Manchester City") == "MC"
    assert team_initials("Brighton & Hove Albion") == "BHA"
    unknown = team_badge("Not A Club")
    assert unknown["crest_url"] is None
    assert unknown["initials"] == "NAC"


def test_extract_match_stats_keeps_only_real_csv_values():
    rows = extract_match_stats(
        {
            "HS": "14",
            "AS": "16",
            "HST": "6",
            "AST": "6",
            "HF": "12",
            "AF": "7",
            "HY": "2",
            "AY": "3",
            "HR": "0",
            "AR": "0",
            "HC": "6",
            "AC": "4",
        }
    )
    keys = [row["key"] for row in rows]
    assert keys == [
        "shots",
        "shots_on_target",
        "fouls",
        "yellow_cards",
        "red_cards",
        "corners",
    ]
    assert "possession" not in keys
    assert "passes" not in keys
    assert "pass_accuracy" not in keys
    assert "offsides" not in keys
    shots = next(row for row in rows if row["key"] == "shots")
    assert shots["home"] == 14
    assert shots["away"] == 16
    assert shots["leader"] == "away"


def test_extract_match_stats_empty_when_payload_has_no_stats():
    assert extract_match_stats(None) == []
    assert extract_match_stats({}) == []
    assert extract_match_stats({"HS": "", "AS": "", "HomeTeam": "Arsenal"}) == []
    assert extract_match_stats({"HS": "10"}) == []


def test_prior_h2h_keeps_only_played_meetings_before_fixture():
    meetings = [
        {
            "match_id": 1,
            "kickoff_date": "2024-04-01",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "is_played": True,
        },
        {
            "match_id": 2,
            "kickoff_date": "2025-12-01",
            "home_team": "Chelsea",
            "away_team": "Arsenal",
            "is_played": True,
        },
        {
            "match_id": 3,
            "kickoff_date": "2026-09-13",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "is_played": False,
        },
        {
            "match_id": 4,
            "kickoff_date": "2023-01-01",
            "home_team": "Arsenal",
            "away_team": "Liverpool",
            "is_played": True,
        },
        {
            "match_id": 5,
            "kickoff_date": "2026-10-01",
            "home_team": "Chelsea",
            "away_team": "Arsenal",
            "is_played": True,
        },
    ]
    selected = filter_prior_h2h(
        meetings,
        fixture_date="2026-09-13",
        fixture_match_id=3,
        home_team="Arsenal",
        away_team="Chelsea",
        limit=5,
    )
    assert [row["match_id"] for row in selected] == [2, 1]
    assert all(row["kickoff_date"] < "2026-09-13" for row in selected)
    assert all({row["home_team"], row["away_team"]} == {"Arsenal", "Chelsea"} for row in selected)


def test_historical_result_serialization_hides_prediction_fields():
    row = {
        "match_id": 3126,
        "match_date": "2026-03-03",
        "kickoff_time": None,
        "home_team": "Bournemouth",
        "away_team": "Brentford",
        "home_goals": 0,
        "away_goals": 0,
        "result_code": 1,
        "p_home": 0.49,
        "p_draw": 0.29,
        "p_away": 0.22,
        "predicted_class": 2,
        "algorithm": "logistic_regression",
        "feature_version": "v2-draw-aware",
        "predicted_at": None,
    }
    hidden = _serialize_fixture(row, include_result=True, include_prediction=False)
    assert hidden["has_prediction"] is False
    assert "p_home" not in hidden
    assert "predicted_outcome" not in hidden
    assert hidden["correct"] is None
    assert hidden["actual_outcome"] == "Draw"
    shown = _serialize_fixture(row, include_result=True, include_prediction=True)
    assert shown["has_prediction"] is True
    assert shown["predicted_outcome"] == "Bournemouth"
    assert shown["correct"] is False

