"""Row counts and class mix at each pipeline stage."""

from __future__ import annotations

import json
import logging

from football_pipeline.config import ROOT
from football_pipeline.db import connect

logger = logging.getLogger(__name__)

SUMMARY_PATH = ROOT / "data" / "processed" / "pipeline_summary.json"
RESULT_LABELS = {0: "away", 1: "draw", 2: "home"}


def _result_counts(rows: list[tuple]) -> dict[str, int]:
    counts = {"away": 0, "draw": 0, "home": 0}
    for code, n in rows:
        if code is None:
            continue
        counts[RESULT_LABELS.get(int(code), str(code))] = int(n)
    return counts


def collect_match_summary(*, competition: str | None = None) -> dict:
    league_filter = ""
    params: tuple = ()
    if competition:
        league_filter = "JOIN competitions AS c ON c.id = m.competition_id WHERE c.code = %s"
        params = (competition,)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT s.start_year, s.name,
                       count(*) AS n,
                       count(*) FILTER (WHERE m.is_played) AS played,
                       count(*) FILTER (WHERE NOT m.is_played) AS unplayed
                FROM matches AS m
                JOIN seasons AS s ON s.id = m.season_id
                {league_filter}
                GROUP BY s.start_year, s.name
                ORDER BY s.start_year
                """,
                params,
            )
            by_season = [
                {
                    "start_year": int(start_year),
                    "season": name,
                    "matches": int(n),
                    "played": int(played),
                    "unplayed": int(unplayed),
                }
                for start_year, name, n, played, unplayed in cur.fetchall()
            ]
            cur.execute(
                f"""
                SELECT result_code, count(*)
                FROM matches AS m
                {"JOIN competitions AS c ON c.id = m.competition_id" if competition else ""}
                WHERE m.is_played
                  {"AND c.code = %s" if competition else ""}
                GROUP BY result_code
                """,
                params,
            )
            results = _result_counts(cur.fetchall())
            cur.execute(
                f"""
                SELECT count(*) FROM match_features AS f
                JOIN matches AS m ON m.id = f.match_id
                {"JOIN competitions AS c ON c.id = m.competition_id WHERE c.code = %s" if competition else ""}
                """,
                params,
            )
            n_features = int(cur.fetchone()[0])
    played = sum(item["played"] for item in by_season)
    unplayed = sum(item["unplayed"] for item in by_season)
    summary = {
        "competition": competition,
        "seasons": by_season,
        "played": played,
        "unplayed": unplayed,
        "matches": played + unplayed,
        "result_code_played": results,
        "match_features": n_features,
    }
    logger.info(
        "Matches: %s played, %s unplayed, features=%s, results=%s",
        played,
        unplayed,
        n_features,
        results,
    )
    return summary


def write_pipeline_summary(payload: dict, *, competition: str | None = None) -> dict:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if SUMMARY_PATH.is_file():
        try:
            existing = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    if competition:
        leagues = existing.setdefault("competitions", {})
        league = leagues.setdefault(competition, {})
        league.update(payload)
        if competition == "E0":
            existing.update(payload)
    else:
        existing.update(payload)
    SUMMARY_PATH.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", SUMMARY_PATH)
    return existing
