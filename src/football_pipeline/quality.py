"""Data quality gates between pipeline stages.

Before this module the pipeline counted rows but never asserted anything, so a
season that downloaded as zero rows would flow silently into model training.
These checks fail the relevant Airflow task loudly instead.

Each gate returns a report dict and raises DataQualityError on a hard failure.
Counting logic is reused from football_pipeline.summarize rather than
duplicated, so there is still one implementation of "how many matches do we
have".
"""

from __future__ import annotations

import logging

from football_pipeline import seasons
from football_pipeline.db import connect
from football_pipeline.summarize import collect_match_summary

logger = logging.getLogger(__name__)

# A Premier League season is 20 clubs playing each other home and away.
EXPECTED_TEAMS_PER_SEASON = 20
EXPECTED_MATCHES_PER_SEASON = 380

# Elo starts at 1500 with K=20. Anything outside this band means the sequential
# scan has gone wrong, not that a club is unusually good or bad.
ELO_MIN = 1000.0
ELO_MAX = 2200.0

# Sanity band for the league-wide draw rate across all played history.
DRAW_RATE_MIN = 0.15
DRAW_RATE_MAX = 0.35

PROBABILITY_TOLERANCE = 0.01

# Mirrors numpy.argmax over the [away, draw, home] column order used when
# predicted_class is written. argmax returns the first maximum, so a tie
# resolves to away, then draw, then home.
ARGMAX_CLASS_SQL = """CASE
    WHEN p_away >= p_draw AND p_away >= p_home THEN 0
    WHEN p_draw >= p_home THEN 1
    ELSE 2
END"""


class DataQualityError(Exception):
    """Input data is incomplete or corrupted. The pipeline must not continue."""


def _fail(problems: list[str], stage: str) -> None:
    if problems:
        detail = "\n  - ".join(problems)
        raise DataQualityError(f"{stage} failed {len(problems)} data quality check(s):\n  - {detail}")


def summary_problems(summary: dict, live_year: int) -> list[str]:
    """Pure checks over the counts collected by summarize.collect_match_summary.

    Split out from check_matches so the rules can be tested without a database.
    """
    problems: list[str] = []

    if not summary.get("seasons"):
        problems.append("No seasons found in matches. Ingest produced nothing.")

    for row in summary.get("seasons", []):
        year = int(row["start_year"])
        total = int(row["matches"])
        label = row.get("season") or year
        if year < live_year:
            if total != EXPECTED_MATCHES_PER_SEASON:
                problems.append(
                    f"Completed season {label} has {total} matches, expected "
                    f"{EXPECTED_MATCHES_PER_SEASON}. Upstream feed is likely incomplete."
                )
            if int(row["unplayed"]) > 0:
                problems.append(
                    f"Completed season {label} still has {row['unplayed']} unplayed matches."
                )
        elif total > EXPECTED_MATCHES_PER_SEASON:
            problems.append(
                f"Live season {label} has {total} matches, more than a full season "
                f"({EXPECTED_MATCHES_PER_SEASON}). Duplicate or rescheduled rows suspected."
            )

    results = summary.get("result_code_played") or {}
    played = int(summary.get("played") or 0)
    if played == 0:
        problems.append("No played matches. Training cannot proceed.")
    else:
        counted = sum(int(v) for v in results.values())
        if counted != played:
            problems.append(
                f"{played} played matches but only {counted} carry a result code."
            )
        draws = int(results.get("draw") or 0)
        draw_rate = draws / played if played else 0.0
        if not DRAW_RATE_MIN <= draw_rate <= DRAW_RATE_MAX:
            problems.append(
                f"League-wide draw rate {draw_rate:.1%} is outside the plausible band "
                f"{DRAW_RATE_MIN:.0%}-{DRAW_RATE_MAX:.0%}. Results may be mis-parsed."
            )

    return problems


def check_matches(summary: dict | None = None, *, today=None) -> dict:
    """Gate after load_core_tables.

    Completed seasons must be whole. The live season may be partial, but it can
    never exceed a full season. Played matches must carry a result.
    """
    summary = summary or collect_match_summary()
    live_year = seasons.live_season_start_year(today)
    problems = summary_problems(summary, live_year)
    problems.extend(_check_teams_and_results(live_year))
    _fail(problems, "load_core_tables")
    logger.info(
        "Data quality OK after load: %s seasons, %s played, %s unplayed",
        len(summary.get("seasons", [])),
        summary.get("played"),
        summary.get("unplayed"),
    )
    return {"stage": "load", "checks_passed": True, "seasons": len(summary.get("seasons", []))}


def _check_teams_and_results(live_year: int) -> list[str]:
    problems: list[str] = []
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.start_year, s.name, count(DISTINCT t.id)
                FROM matches AS m
                JOIN seasons AS s ON s.id = m.season_id
                JOIN teams AS t
                  ON t.id = m.home_team_id OR t.id = m.away_team_id
                GROUP BY s.start_year, s.name
                ORDER BY s.start_year
                """
            )
            for start_year, name, n_teams in cur.fetchall():
                if int(n_teams) != EXPECTED_TEAMS_PER_SEASON:
                    problems.append(
                        f"Season {name} references {n_teams} clubs, expected "
                        f"{EXPECTED_TEAMS_PER_SEASON}. Alias mapping may have split a club."
                    )

            cur.execute(
                """
                SELECT count(*) FROM matches
                WHERE is_played
                  AND (result IS NULL OR result_code IS NULL
                       OR home_goals IS NULL OR away_goals IS NULL)
                """
            )
            broken = int(cur.fetchone()[0])
            if broken:
                problems.append(f"{broken} played match(es) are missing a result or score.")

            cur.execute(
                """
                SELECT count(*) FROM matches
                WHERE NOT is_played
                  AND (home_goals IS NOT NULL OR away_goals IS NOT NULL)
                """
            )
            ghost = int(cur.fetchone()[0])
            if ghost:
                problems.append(f"{ghost} unplayed match(es) carry a score.")
    return problems


def check_features(expected_rows: int | None = None) -> dict:
    """Gate after spark_features. Every match needs a feature row in a sane range."""
    problems: list[str] = []
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM matches")
            n_matches = int(cur.fetchone()[0])
            cur.execute("SELECT count(*) FROM match_features")
            n_features = int(cur.fetchone()[0])
            if n_features != n_matches:
                problems.append(
                    f"{n_matches} matches but {n_features} feature rows. "
                    "Feature build did not cover every match."
                )
            if expected_rows is not None and int(expected_rows) != n_features:
                problems.append(
                    f"Feature build reported {expected_rows} rows written but "
                    f"{n_features} are stored."
                )

            cur.execute(
                """
                SELECT count(*) FROM match_features
                WHERE home_elo IS NULL OR away_elo IS NULL
                   OR home_elo NOT BETWEEN %s AND %s
                   OR away_elo NOT BETWEEN %s AND %s
                """,
                (ELO_MIN, ELO_MAX, ELO_MIN, ELO_MAX),
            )
            bad_elo = int(cur.fetchone()[0])
            if bad_elo:
                problems.append(
                    f"{bad_elo} feature row(s) have Elo outside {ELO_MIN}-{ELO_MAX}."
                )

            cur.execute(
                "SELECT count(*) FROM match_features WHERE feature_version IS NULL"
            )
            if int(cur.fetchone()[0]):
                problems.append("Some feature rows have no feature_version stamp.")

            cur.execute(
                """
                SELECT count(*) FROM match_features
                WHERE home_prior_n < 0 OR away_prior_n < 0
                   OR home_prior_n > 5 OR away_prior_n > 5
                """
            )
            bad_prior = int(cur.fetchone()[0])
            if bad_prior:
                problems.append(
                    f"{bad_prior} feature row(s) have a prior-match count outside 0-5."
                )
    _fail(problems, "spark_features")
    logger.info("Data quality OK after features: %s rows", n_features)
    return {"stage": "features", "checks_passed": True, "match_features": n_features}


def check_training_rows(summary: dict) -> dict:
    """Gate after assemble_training_table. Every split must be populated."""
    problems: list[str] = []
    if int(summary.get("rows") or 0) == 0:
        problems.append("training_rows is empty.")
    for split in ("train", "valid", "test"):
        if int(summary.get(split) or 0) == 0:
            problems.append(
                f"Split '{split}' has no rows. The chronological cutoffs and the "
                "ingested history disagree."
            )
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM training_rows WHERE result_code NOT IN (0, 1, 2)"
            )
            if int(cur.fetchone()[0]):
                problems.append("training_rows contains an invalid result_code.")
            cur.execute(
                "SELECT count(*) FROM training_rows WHERE home_elo IS NULL OR away_elo IS NULL"
            )
            if int(cur.fetchone()[0]):
                problems.append("training_rows contains a NULL Elo.")
    _fail(problems, "assemble_training_table")
    logger.info(
        "Data quality OK after dataset: train=%s valid=%s test=%s",
        summary.get("train"),
        summary.get("valid"),
        summary.get("test"),
    )
    return {"stage": "training_rows", "checks_passed": True, "rows": summary.get("rows")}


def check_predictions() -> dict:
    """Gate after predict_upcoming. Stored probabilities must be usable."""
    problems: list[str] = []
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) FROM predictions
                WHERE p_away < 0 OR p_draw < 0 OR p_home < 0
                   OR abs((p_away + p_draw + p_home) - 1.0) > %s
                """,
                (PROBABILITY_TOLERANCE,),
            )
            bad = int(cur.fetchone()[0])
            if bad:
                problems.append(f"{bad} prediction row(s) have invalid probabilities.")

            # predicted_class is numpy argmax over [away, draw, home], and argmax
            # returns the FIRST maximum. Ties therefore resolve away > draw > home.
            # Historical uniform-baseline rows are exactly (1/3, 1/3, 1/3) with
            # predicted_class 0, and they are correct.
            cur.execute(
                f"""
                SELECT count(*) FROM predictions
                WHERE predicted_class <> {ARGMAX_CLASS_SQL}
                """
            )
            mismatched = int(cur.fetchone()[0])
            if mismatched:
                problems.append(
                    f"{mismatched} prediction row(s) have a predicted_class that is not "
                    "the argmax of the stored probabilities."
                )
            cur.execute("SELECT count(*) FROM predictions")
            total = int(cur.fetchone()[0])
    _fail(problems, "predict_upcoming")
    logger.info("Data quality OK after predict: %s stored prediction rows", total)
    return {"stage": "predictions", "checks_passed": True, "predictions": total}
