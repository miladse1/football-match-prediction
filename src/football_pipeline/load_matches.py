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
            stale, protected = _delete_stale_matches(
                cur, season_ids=season_ids, keep_ids=kept_ids
            )
        conn.commit()

    logger.info(
        "Load finished: %s raw rows, %s matches inserted, %s matches updated, "
        "%s stale dropped, %s stale preserved, %s teams",
        len(payloads),
        inserted,
        updated,
        stale,
        protected,
        len(team_ids),
    )
    return {
        "raw_rows": len(payloads),
        "matches_inserted": inserted,
        "matches_updated": updated,
        "stale_dropped": stale,
        "stale_protected": protected,
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


def partition_stale_matches(candidates: list[dict]) -> tuple[list[int], list[dict]]:
    """Split stale candidates into rows safe to delete and rows that must survive.

    A match is protected when it has already been played, or when a prediction
    is stored against it. Those two rules exist because the project's core
    promise is that a prediction, once made, is never rewritten or lost. An
    incomplete upstream feed must never be able to destroy scored history.

    Only genuinely unplayed, never-predicted fixtures -- rescheduled or dropped
    placeholder rows -- can be removed.
    """
    deletable: list[int] = []
    protected: list[dict] = []
    for row in candidates:
        reasons = []
        if row.get("is_played"):
            reasons.append("already played")
        if int(row.get("prediction_count") or 0) > 0:
            reasons.append(f"{int(row['prediction_count'])} stored prediction(s)")
        if reasons:
            protected.append({**row, "reasons": reasons})
        else:
            deletable.append(int(row["id"]))
    return deletable, protected


def _fetch_stale_candidates(cur, *, season_ids: set[int], keep_ids: list[int]) -> list[dict]:
    cur.execute(
        """
        SELECT m.id,
               m.match_date,
               m.is_played,
               (SELECT count(*) FROM predictions AS p WHERE p.match_id = m.id) AS prediction_count
        FROM matches AS m
        WHERE m.season_id = ANY(%s) AND NOT (m.id = ANY(%s))
        ORDER BY m.id
        """,
        (list(season_ids), keep_ids),
    )
    return [
        {"id": row[0], "match_date": row[1], "is_played": row[2], "prediction_count": row[3]}
        for row in cur.fetchall()
    ]


def _delete_stale_matches(cur, *, season_ids: set[int], keep_ids: list[int]) -> tuple[int, int]:
    """Drop leftover fixture rows that the official feed no longer lists.

    Returns (deleted, protected). Played matches and matches carrying a stored
    prediction are never deleted; they are logged loudly instead.
    """
    if not season_ids or not keep_ids:
        return 0, 0

    candidates = _fetch_stale_candidates(cur, season_ids=season_ids, keep_ids=keep_ids)
    if not candidates:
        return 0, 0

    deletable, protected = partition_stale_matches(candidates)

    for row in protected:
        logger.warning(
            "Preserving match id=%s (%s) missing from the current payloads: %s. "
            "Upstream feed may be incomplete; not deleting scored or predicted history.",
            row["id"],
            row.get("match_date"),
            "; ".join(row["reasons"]),
        )
    if protected:
        logger.warning(
            "%s stale match(es) preserved because they are played or already predicted.",
            len(protected),
        )

    if not deletable:
        return 0, len(protected)

    cur.execute("DELETE FROM matches WHERE id = ANY(%s)", (deletable,))
    deleted = int(cur.rowcount)
    logger.info("Dropped %s unplayed, never-predicted stale fixture row(s)", deleted)
    return deleted, len(protected)


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
