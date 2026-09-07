"""Leakage tests for Spark windows. Run inside the spark Compose service."""

from datetime import date

import pytest

from football_pipeline.spark_features import build_feature_frame, elo_before_matches, spark_session


def _match(match_id, day, home, away, hg, ag, result):
    return {
        "match_id": match_id,
        "match_date": date(2023, 8, day),
        "kickoff_time": "15:00:00",
        "home_team_id": home,
        "away_team_id": away,
        "home_goals": hg,
        "away_goals": ag,
        "result": result,
        "is_played": True,
    }


@pytest.fixture(scope="module")
def spark():
    session = spark_session("feature-tests")
    yield session
    session.stop()


def test_first_match_has_no_prior_form(spark):
    rows = [_match(1, 12, 10, 20, 2, 1, "H")]
    out = {row.match_id: row for row in build_feature_frame(spark, rows).collect()}
    assert out[1].home_prior_n == 0
    assert out[1].home_win_rate_l5 is None
    assert out[1].h2h_home_win_rate_n is None
    assert out[1].home_elo == 1500.0
    assert out[1].away_elo == 1500.0


def test_current_match_result_is_excluded_from_its_own_features(spark):
    # Team 1 scores 7 at home, then plays again. The 7-0 must not appear as this row's form.
    rows = [
        _match(1, 12, 1, 2, 7, 0, "H"),
        _match(2, 19, 1, 3, 1, 1, "D"),
    ]
    out = {row.match_id: row for row in build_feature_frame(spark, rows).collect()}
    assert out[1].home_goals_scored_avg_l5 is None
    assert out[2].home_goals_scored_avg_l5 == pytest.approx(7.0)
    assert out[2].home_win_rate_l5 == pytest.approx(1.0)
    assert out[2].home_draw_rate_l5 == pytest.approx(0.0)
    assert out[2].home_prior_n == 1


def test_draw_in_last_five_is_not_the_current_row(spark):
    rows = [
        _match(1, 12, 1, 2, 1, 1, "D"),
        _match(2, 19, 1, 3, 2, 2, "D"),
        _match(3, 26, 1, 4, 3, 0, "H"),
    ]
    out = {row.match_id: row for row in build_feature_frame(spark, rows).collect()}
    assert out[1].home_draw_rate_l5 is None
    assert out[2].home_draw_rate_l5 == pytest.approx(1.0)
    assert out[3].home_draw_rate_l5 == pytest.approx(1.0)
    assert out[3].h2h_draw_rate_n is None


def test_future_draw_does_not_change_earlier_draw_features(spark):
    early = [
        _match(1, 12, 1, 2, 1, 1, "D"),
        _match(2, 19, 1, 3, 2, 0, "H"),
    ]
    with_future = early + [_match(3, 26, 3, 1, 0, 0, "D")]
    by_id_early = {row.match_id: row for row in build_feature_frame(spark, early).collect()}
    by_id_full = {row.match_id: row for row in build_feature_frame(spark, with_future).collect()}
    for match_id in (1, 2):
        assert by_id_early[match_id].home_draw_rate_l5 == by_id_full[match_id].home_draw_rate_l5
        assert by_id_early[match_id].h2h_draw_rate_n == by_id_full[match_id].h2h_draw_rate_n
    assert by_id_full[2].away_draw_rate_l5 is None or by_id_full[2].away_draw_rate_l5 == pytest.approx(0.0)
    assert by_id_full[3].away_draw_rate_l5 == pytest.approx(0.5)


def test_future_result_does_not_change_earlier_features(spark):
    early = [
        _match(1, 12, 1, 2, 2, 0, "H"),
        _match(2, 19, 1, 3, 1, 0, "H"),
    ]
    with_future = early + [_match(3, 26, 1, 4, 10, 0, "H")]
    by_id_early = {row.match_id: row for row in build_feature_frame(spark, early).collect()}
    by_id_full = {row.match_id: row for row in build_feature_frame(spark, with_future).collect()}
    for match_id in (1, 2):
        assert by_id_early[match_id].home_win_rate_l5 == by_id_full[match_id].home_win_rate_l5
        assert by_id_early[match_id].home_goals_scored_avg_l5 == by_id_full[match_id].home_goals_scored_avg_l5
        assert by_id_early[match_id].home_elo == by_id_full[match_id].home_elo
    assert by_id_full[3].home_prior_n == 2
    assert by_id_full[3].home_win_rate_l5 == pytest.approx(1.0)


def test_upcoming_form_uses_played_history_only(spark):
    played = [
        _match(1, 12, 1, 2, 7, 0, "H"),
        _match(2, 19, 1, 3, 1, 1, "D"),
    ]
    upcoming = {
        "match_id": 3,
        "match_date": date(2023, 9, 2),
        "kickoff_time": "15:00:00",
        "home_team_id": 1,
        "away_team_id": 4,
        "home_goals": None,
        "away_goals": None,
        "result": None,
        "is_played": False,
    }
    out = {row.match_id: row for row in build_feature_frame(spark, played + [upcoming]).collect()}
    assert out[3].home_prior_n == 2
    assert out[3].home_goals_scored_avg_l5 == pytest.approx(4.0)
    assert out[3].home_win_rate_l5 == pytest.approx(0.5)
    assert out[1].home_goals_scored_avg_l5 is None
    assert out[2].home_goals_scored_avg_l5 == pytest.approx(7.0)

    poisoned = {**upcoming, "home_goals": 99, "away_goals": 0, "result": "H", "is_played": False}
    with_poison = {row.match_id: row for row in build_feature_frame(spark, played + [poisoned]).collect()}
    assert with_poison[1].home_goals_scored_avg_l5 is None
    assert with_poison[2].home_goals_scored_avg_l5 == pytest.approx(7.0)
    assert with_poison[3].home_goals_scored_avg_l5 == pytest.approx(4.0)


def test_elo_updates_only_after_kickoff():
    rows = [
        _match(1, 12, 1, 2, 1, 0, "H"),
        _match(2, 19, 1, 3, 0, 0, "D"),
    ]
    elo = elo_before_matches(rows)
    assert elo[1] == (1500.0, 1500.0)
    assert elo[2][0] != 1500.0
    assert elo[2][0] == elo_before_matches(rows + [_match(3, 26, 1, 4, 9, 0, "H")])[2][0]
