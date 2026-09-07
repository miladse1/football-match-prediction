"""Load curated teams and matches from raw_match_payloads.

Reads JSON that ingest already stored. Does not download CSVs.
"""

from __future__ import annotations

import argparse
import logging

from football_pipeline.db import connect
from football_pipeline.normalize import CuratedMatch, TransformError, transform_payload

logger = logging.getLogger(__name__)


def load_matches(*, competition: str | None = None, start_year: int | None = None) -> dict[str, int]:
    with connect() as conn:
        with conn.cursor() as cur:
            payloads = _fetch_payloads(cur, competition=competition, start_year=start_year)
            if not payloads:
                raise TransformError("No raw_match_payloads rows matched the filter. Run ingest first.")

            team_ids: dict[str, int] = {}
            inserted = 0
            updated = 0
            kept_ids: list[int] = []
            season_ids: set[int] = set()
            for raw_id, competition_id, season_id, source_file, row_number, payload in payloads:
                try:
                    match = transform_payload(payload)
                except TransformError as exc:
                    raise TransformError(
                        f"raw_match_payloads.id={raw_id} ({source_file} row {row_number}): {exc}"
                    ) from exc

                home_id = _upsert_team(cur, team_ids, match.home_source_name, match.home_canonical_name)
                away_id = _upsert_team(cur, team_ids, match.away_source_name, match.away_canonical_name)
                match_id, was_insert = _upsert_match(
                    cur,
                    competition_id=competition_id,
                    season_id=season_id,
                    home_team_id=home_id,
                    away_team_id=away_id,
                    match=match,
                    source_file=source_file,
                )
                kept_ids.append(match_id)
                season_ids.add(season_id)
                if was_insert:
                    inserted += 1
                else:
                    updated += 1
            stale = _delete_stale_matches(cur, season_ids=season_ids, keep_ids=kept_ids)
        conn.commit()

    logger.info(
        "Load finished: %s raw rows, %s matches inserted, %s matches updated, %s stale dropped, %s teams",
        len(payloads),
        inserted,
        updated,
        stale,
        len(team_ids),
    )
    return {
        "raw_rows": len(payloads),
        "matches_inserted": inserted,
        "matches_updated": updated,
        "stale_dropped": stale,
        "teams": len(team_ids),
    }


def _fetch_payloads(cur, *, competition: str | None, start_year: int | None):
    query = """
        SELECT r.id, s.competition_id, r.season_id, r.source_file, r.row_number, r.payload
        FROM raw_match_payloads AS r
        JOIN seasons AS s ON s.id = r.season_id
        JOIN competitions AS c ON c.id = s.competition_id
        WHERE (%s::text IS NULL OR c.code = %s)
          AND (%s::int IS NULL OR s.start_year = %s)
        ORDER BY r.id
    """
    cur.execute(query, (competition, competition, start_year, start_year))
    return cur.fetchall()


def _upsert_team(cur, cache: dict[str, int], source_name: str, canonical_name: str) -> int:
    if canonical_name in cache:
        return cache[canonical_name]
    cur.execute("SELECT id FROM teams WHERE canonical_name = %s", (canonical_name,))
    row = cur.fetchone()
    if row:
        cache[canonical_name] = row[0]
        return row[0]
    cur.execute("SELECT id, canonical_name FROM teams WHERE source_name = %s", (source_name,))
    row = cur.fetchone()
    if row:
        team_id, existing_canonical = row
        if existing_canonical != canonical_name:
            cur.execute("SELECT id FROM teams WHERE canonical_name = %s", (canonical_name,))
            taken = cur.fetchone()
            if taken:
                cache[canonical_name] = taken[0]
                return taken[0]
            cur.execute(
                "UPDATE teams SET canonical_name = %s WHERE id = %s",
                (canonical_name, team_id),
            )
        cache[canonical_name] = team_id
        return team_id
    cur.execute(
        """
        INSERT INTO teams (source_name, canonical_name)
        VALUES (%s, %s)
        ON CONFLICT (canonical_name) DO UPDATE
            SET source_name = teams.source_name
        RETURNING id
        """,
        (source_name, canonical_name),
    )
    team_id = cur.fetchone()[0]
    cache[canonical_name] = team_id
    return team_id


def _upsert_match(
    cur,
    *,
    competition_id: int,
    season_id: int,
    home_team_id: int,
    away_team_id: int,
    match: CuratedMatch,
    source_file: str,
) -> tuple[int, bool]:
    cur.execute(
        """
        INSERT INTO matches (
            competition_id,
            season_id,
            match_date,
            kickoff_time,
            home_team_id,
            away_team_id,
            home_goals,
            away_goals,
            result,
            result_code,
            is_played,
            source_file,
            source_row_hash
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (competition_id, match_date, home_team_id, away_team_id) DO UPDATE
            SET season_id = EXCLUDED.season_id,
                kickoff_time = COALESCE(EXCLUDED.kickoff_time, matches.kickoff_time),
                home_goals = CASE WHEN EXCLUDED.is_played THEN EXCLUDED.home_goals ELSE matches.home_goals END,
                away_goals = CASE WHEN EXCLUDED.is_played THEN EXCLUDED.away_goals ELSE matches.away_goals END,
                result = CASE WHEN EXCLUDED.is_played THEN EXCLUDED.result ELSE matches.result END,
                result_code = CASE WHEN EXCLUDED.is_played THEN EXCLUDED.result_code ELSE matches.result_code END,
                is_played = matches.is_played OR EXCLUDED.is_played,
                source_file = CASE
                    WHEN EXCLUDED.is_played OR NOT matches.is_played THEN EXCLUDED.source_file
                    ELSE matches.source_file
                END,
                source_row_hash = CASE
                    WHEN EXCLUDED.is_played OR NOT matches.is_played THEN EXCLUDED.source_row_hash
                    ELSE matches.source_row_hash
                END
        RETURNING id, (xmax = 0) AS inserted
        """,
        (
            competition_id,
            season_id,
            match.match_date,
            match.kickoff_time,
            home_team_id,
            away_team_id,
            match.home_goals,
            match.away_goals,
            match.result,
            match.result_code,
            match.is_played,
            source_file,
            match.source_row_hash,
        ),
    )
    match_id, inserted = cur.fetchone()
    return int(match_id), bool(inserted)


def _delete_stale_matches(cur, *, season_ids: set[int], keep_ids: list[int]) -> int:
    """Drop leftover demo-fixture rows whose date/teams do not match the official CSV."""
    if not season_ids or not keep_ids:
        return 0
    cur.execute(
        """
        DELETE FROM predictions
        WHERE match_id IN (
            SELECT id FROM matches
            WHERE season_id = ANY(%s) AND NOT (id = ANY(%s))
        )
        """,
        (list(season_ids), keep_ids),
    )
    cur.execute(
        """
        DELETE FROM matches
        WHERE season_id = ANY(%s) AND NOT (id = ANY(%s))
        """,
        (list(season_ids), keep_ids),
    )
    return int(cur.rowcount)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Transform raw payloads into teams and matches.")
    parser.add_argument("--competition", default=None, help="Limit to a competition code, e.g. E0")
    parser.add_argument("--start-year", type=int, default=None, help="Limit to a season start year")
    args = parser.parse_args()
    load_matches(competition=args.competition, start_year=args.start_year)


if __name__ == "__main__":
    main()
