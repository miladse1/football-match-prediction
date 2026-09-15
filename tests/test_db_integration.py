"""Integration tests against a real PostgreSQL instance.

Skipped unless RUN_DB_TESTS=1 (see conftest.py). CI runs them against a
throwaway service container. Every test builds its own randomly named database
and drops it afterwards, so running these locally never touches the project
database.
"""

from __future__ import annotations

import os
import uuid
from datetime import date, timedelta

import psycopg
import pytest

from football_pipeline import config, migrate
from football_pipeline.load_matches import _delete_stale_matches, _upsert_match
from football_pipeline.normalize import CuratedMatch

ADMIN_DB = os.getenv("POSTGRES_DB", "postgres")


def _admin_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        user=os.getenv("POSTGRES_USER", "football"),
        password=os.getenv("POSTGRES_PASSWORD", "football"),
        dbname=ADMIN_DB,
        autocommit=True,
    )


@pytest.fixture()
def fresh_database(monkeypatch):
    """A throwaway database with all migrations applied."""
    name = f"fmp_test_{uuid.uuid4().hex[:12]}"
    with _admin_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{name}"')
    monkeypatch.setattr(config, "POSTGRES_DB", name)
    try:
        migrate.apply_migrations()
        yield name
    finally:
        monkeypatch.undo()
        with _admin_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s",
                    (name,),
                )
                cur.execute(f'DROP DATABASE IF EXISTS "{name}"')


def _connect(name):
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        user=os.getenv("POSTGRES_USER", "football"),
        password=os.getenv("POSTGRES_PASSWORD", "football"),
        dbname=name,
    )


EXPECTED_TABLES = {
    "competitions",
    "seasons",
    "teams",
    "matches",
    "raw_match_payloads",
    "match_features",
    "training_rows",
    "model_runs",
    "predictions",
    "schema_migrations",
}


def test_migrations_create_every_expected_table(fresh_database):
    with _connect(fresh_database) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
            tables = {row[0] for row in cur.fetchall()}
    assert EXPECTED_TABLES <= tables


def test_migrations_are_idempotent(fresh_database):
    """Re-running must be a no-op, not an error or a duplicate apply."""
    with _connect(fresh_database) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM schema_migrations")
            first = int(cur.fetchone()[0])
    migrate.apply_migrations()
    with _connect(fresh_database) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM schema_migrations")
            second = int(cur.fetchone()[0])
    assert first == second > 0


def _seed_match(cur, *, home, away, played, match_date=date(2026, 9, 12)):
    cur.execute(
        """
        INSERT INTO competitions (code, name, country) VALUES ('E0', 'EPL', 'England')
        ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name RETURNING id
        """
    )
    competition_id = cur.fetchone()[0]
    cur.execute(
        """
        INSERT INTO seasons (competition_id, start_year, name) VALUES (%s, 2026, '2026/2027')
        ON CONFLICT (competition_id, start_year) DO UPDATE SET name = EXCLUDED.name RETURNING id
        """,
        (competition_id,),
    )
    season_id = cur.fetchone()[0]
    ids = []
    for club in (home, away):
        cur.execute(
            """
            INSERT INTO teams (competition_id, source_name, canonical_name) VALUES (%s, %s, %s)
            ON CONFLICT (competition_id, canonical_name) DO UPDATE SET source_name = teams.source_name
            RETURNING id
            """,
            (competition_id, club, club),
        )
        ids.append(cur.fetchone()[0])
    cur.execute(
        """
        INSERT INTO matches (
            competition_id, season_id, match_date, home_team_id, away_team_id,
            home_goals, away_goals, result, result_code, is_played
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            competition_id,
            season_id,
            match_date,
            ids[0],
            ids[1],
            2 if played else None,
            1 if played else None,
            "H" if played else None,
            2 if played else None,
            played,
        ),
    )
    return cur.fetchone()[0], season_id


def _seed_prediction(cur, match_id):
    cur.execute("SELECT competition_id FROM matches WHERE id = %s", (match_id,))
    competition_id = cur.fetchone()[0]
    cur.execute(
        """
        INSERT INTO model_runs (competition_id, feature_version, algorithm, artifact_path)
        VALUES (%s, 'v2-draw-aware', 'logistic_regression', '/tmp/x.joblib')
        RETURNING id
        """,
        (competition_id,),
    )
    run_id = cur.fetchone()[0]
    cur.execute(
        """
        INSERT INTO predictions (match_id, model_run_id, p_away, p_draw, p_home, predicted_class)
        VALUES (%s, %s, 0.25, 0.25, 0.50, 2)
        """,
        (match_id, run_id),
    )
    return run_id


def test_played_match_and_its_prediction_survive_a_missing_payload(fresh_database):
    """The frozen-prediction guarantee, proved against a real database."""
    with _connect(fresh_database) as conn:
        with conn.cursor() as cur:
            played_id, season_id = _seed_match(cur, home="Arsenal", away="Chelsea", played=True)
            _seed_prediction(cur, played_id)
            unplayed_id, _ = _seed_match(
                cur, home="Everton", away="Fulham", played=False, match_date=date(2026, 10, 3)
            )
            conn.commit()

            # The upstream feed now lists neither match.
            deleted, protected = _delete_stale_matches(
                cur, season_ids={season_id}, keep_ids=[-1]
            )
            conn.commit()

            cur.execute("SELECT count(*) FROM matches WHERE id = %s", (played_id,))
            assert int(cur.fetchone()[0]) == 1, "played match was deleted"
            cur.execute("SELECT count(*) FROM predictions WHERE match_id = %s", (played_id,))
            assert int(cur.fetchone()[0]) == 1, "stored prediction was deleted"
            cur.execute("SELECT count(*) FROM matches WHERE id = %s", (unplayed_id,))
            assert int(cur.fetchone()[0]) == 0, "stale placeholder should have been dropped"

    assert deleted == 1
    assert protected == 1


def test_unplayed_but_predicted_match_survives(fresh_database):
    with _connect(fresh_database) as conn:
        with conn.cursor() as cur:
            match_id, season_id = _seed_match(
                cur, home="Liverpool", away="Brentford", played=False
            )
            _seed_prediction(cur, match_id)
            conn.commit()

            deleted, protected = _delete_stale_matches(
                cur, season_ids={season_id}, keep_ids=[-1]
            )
            conn.commit()

            cur.execute("SELECT count(*) FROM matches WHERE id = %s", (match_id,))
            assert int(cur.fetchone()[0]) == 1
    assert deleted == 0
    assert protected == 1


def test_probability_sum_constraint_is_enforced(fresh_database):
    with _connect(fresh_database) as conn:
        with conn.cursor() as cur:
            match_id, _ = _seed_match(cur, home="Burnley", away="Leeds United", played=False)
            cur.execute(
                """
                INSERT INTO model_runs (competition_id, feature_version, algorithm, artifact_path)
                VALUES (
                    (SELECT competition_id FROM matches WHERE id = %s),
                    'v2-draw-aware', 'logistic_regression', '/tmp/x.joblib'
                ) RETURNING id
                """,
                (match_id,),
            )
            run_id = cur.fetchone()[0]
            conn.commit()
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    """
                    INSERT INTO predictions (
                        match_id, model_run_id, p_away, p_draw, p_home, predicted_class
                    )
                    VALUES (%s, %s, 0.9, 0.9, 0.9, 2)
                    """,
                    (match_id, run_id),
                )


def test_argmax_check_matches_numpy_tie_breaking(fresh_database):
    """Uniform baseline rows are (1/3, 1/3, 1/3) with class 0. argmax picks the first max."""
    import numpy as np

    from football_pipeline.quality import ARGMAX_CLASS_SQL, DataQualityError, check_predictions

    third = 1.0 / 3.0
    cases = [
        (third, third, third, 0),          # three-way tie -> away
        (0.40, 0.40, 0.20, 0),             # away/draw tie -> away
        (0.20, 0.40, 0.40, 1),             # draw/home tie -> draw
        (0.40, 0.20, 0.40, 0),             # away/home tie -> away
        (0.20, 0.30, 0.50, 2),             # clear home
        (0.50, 0.30, 0.20, 0),             # clear away
        (0.25, 0.50, 0.25, 1),             # clear draw
    ]
    with _connect(fresh_database) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO competitions (code, name, country)
                VALUES ('E0', 'Premier League', 'England')
                ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name
                """
            )
            cur.execute(
                """
                INSERT INTO model_runs (competition_id, feature_version, algorithm, artifact_path)
                VALUES (
                    (SELECT id FROM competitions WHERE code = 'E0'),
                    'v2-draw-aware', 'uniform', '/tmp/x.joblib'
                ) RETURNING id
                """
            )
            run_id = cur.fetchone()[0]
            for i, (away, draw, home, expected) in enumerate(cases):
                assert int(np.argmax([away, draw, home])) == expected
                match_id, _ = _seed_match(
                    cur,
                    home=f"Home {i}",
                    away=f"Away {i}",
                    played=False,
                    match_date=date(2026, 9, 1) + timedelta(days=i),
                )
                cur.execute(
                    """
                    INSERT INTO predictions (
                        match_id, model_run_id, p_away, p_draw, p_home, predicted_class
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (match_id, run_id, away, draw, home, expected),
                )
            conn.commit()

            cur.execute(f"SELECT p_away, p_draw, p_home, {ARGMAX_CLASS_SQL} FROM predictions")
            for away, draw, home, sql_class in cur.fetchall():
                assert int(np.argmax([float(away), float(draw), float(home)])) == int(sql_class)

    # The gate accepts these correct rows.
    assert check_predictions()["checks_passed"] is True

    # And rejects a genuinely wrong predicted_class.
    with _connect(fresh_database) as conn:
        with conn.cursor() as cur:
            match_id, _ = _seed_match(
                cur, home="Wrong Home", away="Wrong Away", played=False,
                match_date=date(2026, 12, 25),
            )
            cur.execute(
                """
                INSERT INTO predictions (
                    match_id, model_run_id, p_away, p_draw, p_home, predicted_class
                )
                SELECT %s, id, 0.10, 0.10, 0.80, 0 FROM model_runs LIMIT 1
                """,
                (match_id,),
            )
            conn.commit()
    with pytest.raises(DataQualityError, match="argmax"):
        check_predictions()


def test_postponement_keeps_frozen_predictions_and_drops_the_date_duplicate(fresh_database):
    """A one-day postponement must not create a second match_id."""
    with _connect(fresh_database) as conn:
        with conn.cursor() as cur:
            old_id, _season_id = _seed_match(
                cur,
                home="Coventry City",
                away="Chelsea",
                played=False,
                match_date=date(2026, 12, 26),
            )
            _seed_prediction(cur, old_id)
            _seed_match(
                cur,
                home="Coventry City",
                away="Chelsea",
                played=False,
                match_date=date(2026, 12, 27),
            )
            conn.commit()
            cur.execute(
                """
                SELECT competition_id, season_id, home_team_id, away_team_id,
                       p_away, p_draw, p_home
                FROM matches AS m
                JOIN predictions AS p ON p.match_id = m.id
                WHERE m.id = %s
                """,
                (old_id,),
            )
            competition_id, season_id, home_id, away_id, p_away, p_draw, p_home = cur.fetchone()
            incoming = CuratedMatch(
                match_date=date(2026, 12, 27),
                kickoff_time=None,
                home_source_name="Coventry",
                away_source_name="Chelsea",
                home_canonical_name="Coventry City",
                away_canonical_name="Chelsea",
                home_goals=None,
                away_goals=None,
                result=None,
                result_code=None,
                is_played=False,
                source_row_hash="fixture-reschedule",
            )
            kept_id, inserted = _upsert_match(
                cur,
                competition_id=competition_id,
                season_id=season_id,
                home_team_id=home_id,
                away_team_id=away_id,
                match=incoming,
                source_file="epl-2026.json",
            )
            conn.commit()
            assert kept_id == old_id
            assert inserted is False
            cur.execute(
                """
                SELECT count(*), min(id), max(id), min(match_date), max(match_date)
                FROM matches
                WHERE home_team_id = %s AND away_team_id = %s
                """,
                (home_id, away_id),
            )
            n, min_id, max_id, min_date, max_date = cur.fetchone()
            assert n == 1
            assert min_id == max_id == old_id
            assert min_date == max_date == date(2026, 12, 27)
            cur.execute(
                "SELECT p_away, p_draw, p_home FROM predictions WHERE match_id = %s",
                (old_id,),
            )
            assert cur.fetchone() == (p_away, p_draw, p_home)
            assert (float(p_away), float(p_draw), float(p_home)) == (0.25, 0.25, 0.50)


def test_teams_unique_constraint_is_competition_scoped(fresh_database):
    with _connect(fresh_database) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT constraint_name
                FROM information_schema.table_constraints
                WHERE table_schema = 'public'
                  AND table_name = 'teams'
                  AND constraint_type = 'UNIQUE'
                """
            )
            names = {row[0] for row in cur.fetchall()}
            cur.execute(
                """
                SELECT column_name, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'teams' AND column_name = 'competition_id'
                """
            )
            column, nullable = cur.fetchone()
    assert "teams_competition_canonical_unique" in names
    assert "teams_canonical_name_unique" not in names
    assert column == "competition_id"
    assert nullable == "NO"


def test_same_club_name_can_exist_in_two_competitions(fresh_database):
    with _connect(fresh_database) as conn:
        with conn.cursor() as cur:
            _seed_match(cur, home="Arsenal", away="Chelsea", played=True)
            cur.execute(
                """
                INSERT INTO competitions (code, name, country)
                VALUES ('SP1', 'La Liga', 'Spain')
                RETURNING id
                """
            )
            sp1_id = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO teams (competition_id, source_name, canonical_name)
                VALUES (%s, 'Arsenal', 'Arsenal')
                RETURNING id
                """,
                (sp1_id,),
            )
            liga_id = cur.fetchone()[0]
            cur.execute(
                "SELECT id, competition_id FROM teams WHERE canonical_name = 'Arsenal' ORDER BY id"
            )
            rows = cur.fetchall()
            conn.commit()
    assert len(rows) == 2
    assert liga_id in {row[0] for row in rows}
    assert {row[1] for row in rows} == {rows[0][1], sp1_id}
