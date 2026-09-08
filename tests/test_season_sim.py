import copy
from pathlib import Path

import numpy as np

from football_pipeline.season_sim import (
    PlayedMatch,
    RemainingFixture,
    SeasonSnapshot,
    TeamRecord,
    apply_played_match,
    build_table,
    forecast_matches_snapshot,
    match_points,
    rank_indices,
    sample_1x2,
    sample_outcomes,
    simulate_season,
)


def test_match_points_awards_three_one_zero():
    assert match_points(2, 0) == (3, 0)
    assert match_points(0, 1) == (0, 3)
    assert match_points(1, 1) == (1, 1)


def test_played_table_accumulates_points_and_goal_difference():
    table = build_table(
        [
            PlayedMatch("Arsenal", "Chelsea", 2, 1),
            PlayedMatch("Chelsea", "Arsenal", 0, 0),
        ],
        ["Arsenal", "Chelsea", "Liverpool"],
    )
    assert table["Arsenal"].points == 4
    assert table["Chelsea"].points == 1
    assert table["Liverpool"].points == 0
    assert table["Arsenal"].gd == 1
    assert table["Chelsea"].gd == -1
    assert table["Arsenal"].played == 2


def test_apply_played_match_does_not_require_prior_row():
    table: dict[str, TeamRecord] = {}
    apply_played_match(table, PlayedMatch("Everton", "Fulham", 3, 3))
    assert table["Everton"].points == 1
    assert table["Fulham"].gf == 3


def test_sample_1x2_follows_degenerate_probabilities():
    rng = np.random.default_rng(0)
    assert {sample_1x2(1, 0, 0, rng) for _ in range(20)} == {0}
    assert {sample_1x2(0, 1, 0, rng) for _ in range(20)} == {1}
    assert {sample_1x2(0, 0, 1, rng) for _ in range(20)} == {2}


def test_sample_outcomes_frequency_is_close_to_stored_probs():
    rng = np.random.default_rng(11)
    probs = np.array([[0.2, 0.3, 0.5]])
    draws = sample_outcomes(probs, rng, 20_000).ravel()
    shares = np.bincount(draws, minlength=3) / 20_000
    assert np.allclose(shares, [0.2, 0.3, 0.5], atol=0.02)


def test_simulation_is_reproducible_with_a_fixed_seed():
    teams = ["Arsenal", "Chelsea", "Liverpool"]
    table = build_table(
        [PlayedMatch("Arsenal", "Chelsea", 1, 0)],
        teams,
    )
    remaining = [
        RemainingFixture(1, "Liverpool", "Arsenal", 0.25, 0.25, 0.50),
        RemainingFixture(2, "Chelsea", "Liverpool", 0.40, 0.30, 0.30),
    ]
    first = simulate_season(teams=teams, table=table, remaining=remaining, n_sims=2_000, seed=99)
    second = simulate_season(teams=teams, table=table, remaining=remaining, n_sims=2_000, seed=99)
    assert [row["title_prob"] for row in first["teams"]] == [row["title_prob"] for row in second["teams"]]
    other = simulate_season(teams=teams, table=table, remaining=remaining, n_sims=2_000, seed=100)
    assert [row["title_prob"] for row in first["teams"]] != [row["title_prob"] for row in other["teams"]]


def test_title_top4_and_relegation_shares_sum_to_place_counts():
    teams = [f"Team {i}" for i in range(20)]
    table = build_table([], teams)
    remaining = [
        RemainingFixture(i, teams[i % 20], teams[(i + 1) % 20], 0.33, 0.34, 0.33) for i in range(10)
    ]
    report = simulate_season(teams=teams, table=table, remaining=remaining, n_sims=500, seed=7)
    assert abs(report["title_prob_sum"] - 1.0) < 1e-12
    assert abs(report["top4_prob_sum"] - 4.0) < 1e-12
    assert abs(report["relegation_prob_sum"] - 3.0) < 1e-12


def test_points_tie_uses_current_goal_difference_not_simulated_scores():
    teams = ["Alpha", "Beta"]
    table = {
        "Alpha": TeamRecord(name="Alpha", points=6, gf=5, ga=1),
        "Beta": TeamRecord(name="Beta", points=6, gf=2, ga=1),
    }
    report = simulate_season(teams=teams, table=table, remaining=[], n_sims=50, seed=1)
    by_name = {row["team"]: row for row in report["teams"]}
    assert by_name["Alpha"]["title_prob"] == 1.0
    assert by_name["Beta"]["title_prob"] == 0.0
    order = rank_indices(
        np.array([6, 6]),
        np.array([4, 1]),
        np.array([5, 2]),
        teams,
    )
    assert teams[int(order[0])] == "Alpha"


def test_simulation_does_not_mutate_stored_probabilities():
    remaining = [
        RemainingFixture(10, "Arsenal", "Chelsea", 0.21, 0.27, 0.52),
    ]
    snapshot = copy.deepcopy(remaining)
    table = build_table([PlayedMatch("Arsenal", "Chelsea", 1, 0)], ["Arsenal", "Chelsea"])
    simulate_season(
        teams=["Arsenal", "Chelsea"],
        table=table,
        remaining=remaining,
        n_sims=200,
        seed=3,
    )
    assert remaining == snapshot
    assert remaining[0].p_draw == 0.27


def test_season_sim_module_never_writes_predictions():
    source = Path("src/football_pipeline/season_sim.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "insert into predictions" not in lowered
    assert "update predictions" not in lowered
    assert "delete from predictions" not in lowered
    assert "joblib" not in lowered
    assert "train_and_evaluate" not in source


def test_forecast_artifact_is_rejected_when_database_state_changes():
    remaining = [RemainingFixture(1, "Arsenal", "Chelsea", 0.2, 0.3, 0.5)]
    snapshot = SeasonSnapshot(
        season="2026/27",
        start_year=2026,
        teams=["Arsenal", "Chelsea"],
        played=[PlayedMatch("Arsenal", "Chelsea", 1, 0)],
        remaining=remaining,
        missing_remaining=0,
        model_run_id=82,
        algorithm="logistic_regression",
        feature_version="v2-draw-aware",
        predictions_updated_at="2026-09-07T18:17:00+00:00",
    )
    report = {
        "model_run_id": 82,
        "n_completed": 1,
        "n_remaining": 1,
        "predictions_updated_at": "2026-09-07T18:17:00+00:00",
    }
    assert forecast_matches_snapshot(report, snapshot)
    assert not forecast_matches_snapshot({**report, "n_completed": 20}, snapshot)
    assert not forecast_matches_snapshot({**report, "model_run_id": 99}, snapshot)


def test_dag_runs_season_forecast_after_predict_upcoming():
    dag = Path("airflow/dags/football_match_dag.py").read_text(encoding="utf-8")
    tasks = Path("src/football_pipeline/tasks.py").read_text(encoding="utf-8")
    assert "def simulate_live_season" in tasks
    assert "run_live_forecast" in tasks
    assert dag.index(">> predict_upcoming()") < dag.index(">> season_forecast()")
    assert "insert into predictions" not in tasks.lower()


def test_dag_is_scheduled_twice_weekly_without_catchup_or_overlap():
    dag = Path("airflow/dags/football_match_dag.py").read_text(encoding="utf-8")
    assert 'schedule="0 6 * * 1,4"' in dag
    assert 'tz="America/New_York"' in dag or "America/New_York" in dag
    assert "catchup=False" in dag
    assert "max_active_runs=1" in dag
    assert "migrate_db()" in dag
    assert ">> ingest_raw()" in dag
    assert ">> season_forecast()" in dag
