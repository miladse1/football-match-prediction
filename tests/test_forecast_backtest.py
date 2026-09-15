from datetime import date
from pathlib import Path

from football_pipeline.forecast_backtest import (
    PRODUCTION_FORECAST,
    PRODUCTION_METRICS,
    REPORT_PATH,
    actual_champion,
    cutoff_for_checkpoint,
    remaining_feature_frame,
    score_against_champion,
    target_league_matches,
)
from football_pipeline.season_sim import ARTIFACT_PATH


def _match(match_id, day, home_id, away_id, hg, ag, result, month=8, year=2023):
    home = {1: "Arsenal", 2: "Chelsea", 3: "Liverpool", 4: "Everton"}[home_id]
    away = {1: "Arsenal", 2: "Chelsea", 3: "Liverpool", 4: "Everton"}[away_id]
    return {
        "match_id": match_id,
        "match_date": date(year, month, day),
        "kickoff_time": "15:00:00",
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_team": home,
        "away_team": away,
        "home_goals": hg,
        "away_goals": ag,
        "result": result,
        "is_played": True,
    }


def test_target_match_count_is_half_the_games_played():
    assert target_league_matches(5) == 50
    assert target_league_matches(30) == 300
    assert target_league_matches(5, n_clubs=4) == 10


def test_cutoff_includes_the_whole_matchday_and_allows_games_in_hand():
    played = [
        _match(1, 12, 1, 2, 1, 0, "H"),
        _match(2, 12, 3, 4, 2, 2, "D"),
        _match(3, 19, 1, 3, 3, 0, "H"),
        _match(4, 19, 2, 4, 0, 1, "A"),
        _match(5, 19, 4, 1, 1, 1, "D"),
        _match(6, 26, 2, 3, 0, 0, "D"),
    ]
    meta = cutoff_for_checkpoint(played, n_per_team=2, n_clubs=4)
    assert meta["target_matches"] == 4
    assert meta["cutoff_date"] == date(2023, 8, 19)
    assert meta["n_played"] == 5
    assert meta["played_min"] == 2
    assert meta["played_max"] == 3


def test_actual_champion_uses_points_then_goal_difference():
    played = [
        _match(1, 12, 1, 2, 2, 0, "H"),
        _match(2, 12, 3, 4, 1, 0, "H"),
        _match(3, 19, 2, 3, 0, 0, "D"),
        _match(4, 19, 4, 1, 0, 1, "A"),
    ]
    champ = actual_champion(played, ["Arsenal", "Chelsea", "Liverpool", "Everton"])
    assert champ == "Arsenal"


def test_remaining_features_ignore_later_results():
    played = [_match(1, 12, 1, 2, 7, 0, "H")]
    remaining = [
        {
            **_match(2, 19, 1, 3, 1, 1, "D"),
            "is_played": False,
            "home_goals": None,
            "away_goals": None,
            "result": None,
        },
        {
            **_match(4, 30, 1, 3, 1, 1, "D"),
            "is_played": False,
            "home_goals": None,
            "away_goals": None,
            "result": None,
        },
    ]
    clean = remaining_feature_frame(played, remaining)
    leaked_played = played + [_match(3, 26, 1, 4, 0, 5, "A")]
    leaked = remaining_feature_frame(leaked_played, remaining)
    assert clean.iloc[0]["home_goals_scored_avg_l5"] == 7.0
    assert leaked.iloc[0]["home_goals_scored_avg_l5"] == 3.5
    # A remaining fixture after the leaked result would also see a different Elo.
    later = clean.set_index("match_id").loc[4]
    later_leaked = leaked.set_index("match_id").loc[4]
    assert later["home_elo"] != later_leaked["home_elo"]


def test_unplayed_remaining_rows_do_not_update_elo():
    played = [_match(1, 12, 1, 2, 1, 0, "H")]
    remaining = [
        {
            **_match(2, 19, 1, 3, 9, 0, "H"),
            "is_played": False,
            "home_goals": None,
            "away_goals": None,
            "result": None,
        },
        {
            **_match(3, 26, 1, 4, 9, 0, "H"),
            "is_played": False,
            "home_goals": None,
            "away_goals": None,
            "result": None,
        },
    ]
    feats = remaining_feature_frame(played, remaining)
    assert feats.iloc[0]["home_elo"] == feats.iloc[1]["home_elo"]


def test_score_against_champion_flags():
    checkpoint = {
        "season": "2023/24",
        "n_per_team": 10,
        "cutoff_date": "2023-11-01",
        "n_played": 100,
        "played_min": 9,
        "played_max": 11,
        "favorite": "Arsenal",
        "favorite_title_pct": 0.4,
        "title_table": [
            {"team": "Arsenal", "title_prob": 0.4, "rank": 1, "current_points": 20, "current_played": 10},
            {"team": "Man City", "title_prob": 0.35, "rank": 2, "current_points": 19, "current_played": 10},
            {"team": "Liverpool", "title_prob": 0.2, "rank": 3, "current_points": 18, "current_played": 10},
        ],
    }
    hit = score_against_champion(checkpoint, "Arsenal")
    miss = score_against_champion(checkpoint, "Liverpool")
    assert hit["correct_1"] is True
    assert miss["correct_1"] is False
    assert miss["correct_top2"] is False
    assert miss["correct_top3"] is True
    assert miss["champion_rank"] == 3


def test_experiment_does_not_write_production_artifacts():
    source = Path("src/football_pipeline/forecast_backtest.py").read_text(encoding="utf-8")
    assert "INSERT INTO predictions" not in source
    assert "INSERT INTO model_runs" not in source
    assert "joblib.dump" not in source
    assert str(REPORT_PATH).endswith("experiments/forecast_backtest.json")
    assert REPORT_PATH != PRODUCTION_FORECAST
    assert REPORT_PATH != PRODUCTION_METRICS
    assert ARTIFACT_PATH == PRODUCTION_FORECAST
