"""Extract: download (or read) a season CSV and land rows in raw_match_payloads.

Does not write teams or matches. Run `python -m football_pipeline.load_matches` after ingest.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path

from psycopg.types.json import Jsonb

from football_pipeline.config import RAW_DATA_DIR
from football_pipeline.db import connect
from football_pipeline.football_data import (
    IngestError,
    current_season_start_year,
    download_season_csv,
    season_name,
)
from football_pipeline.validation import (
    assert_header,
    assert_result_if_present,
    row_is_empty,
)

logger = logging.getLogger(__name__)

COMPETITIONS = {
    "E0": {"name": "English Premier League", "country": "England"},
}


def ingest_season(
    competition_code: str,
    start_year: int,
    *,
    from_file: Path | None = None,
    force_download: bool = False,
) -> dict[str, int]:
    if competition_code not in COMPETITIONS:
        known = ", ".join(sorted(COMPETITIONS))
        raise IngestError(f"Unknown competition {competition_code!r}. Known: {known}")

    source_file = f"{competition_code}_{start_year}-{start_year + 1}.csv"
    dest = RAW_DATA_DIR / source_file
    source_url = dest.as_uri()

    if from_file is not None:
        src = from_file.expanduser().resolve()
        if not src.is_file():
            raise IngestError(f"--from-file not found: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        source_url = src.as_uri()
        logger.info("Using local file %s (copied to %s)", src, dest)
    elif dest.exists() and not force_download:
        logger.info("Reuse existing %s (pass --force-download to refresh)", dest)
    else:
        source_url = download_season_csv(competition_code, start_year, dest)

    rows = _read_csv_rows(dest)
    logger.info("Parsed %s data rows from %s", len(rows), dest)

    with connect() as conn:
        with conn.cursor() as cur:
            season_id = _upsert_season(cur, competition_code, start_year)
            inserted = _insert_payloads(cur, season_id, source_url, source_file, rows)
        conn.commit()

    skipped = len(rows) - inserted
    logger.info(
        "Ingest %s %s: %s CSV rows, %s inserted, %s updated",
        competition_code,
        season_name(start_year),
        len(rows),
        inserted,
        skipped,
    )
    return {
        "competition": competition_code,
        "start_year": start_year,
        "season": season_name(start_year),
        "rows": len(rows),
        "inserted": inserted,
        "updated": skipped,
    }


def ingest_seasons(
    competition_code: str,
    start_year: int,
    end_year: int | None = None,
    *,
    force_download: bool = True,
) -> dict:
    last_year = current_season_start_year() if end_year is None else end_year
    if last_year < start_year:
        raise IngestError(f"end_year {last_year} is before start_year {start_year}")
    seasons = [
        ingest_season(competition_code, year, force_download=force_download)
        for year in range(start_year, last_year + 1)
    ]
    summary = {
        "competition": competition_code,
        "start_year": start_year,
        "end_year": last_year,
        "season_count": len(seasons),
        "rows": sum(item["rows"] for item in seasons),
        "inserted": sum(item["inserted"] for item in seasons),
        "updated": sum(item["updated"] for item in seasons),
        "seasons": seasons,
    }
    logger.info(
        "Ingested %s seasons %s–%s (%s CSV rows)",
        summary["season_count"],
        season_name(start_year),
        season_name(last_year),
        summary["rows"],
    )
    return summary


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        assert_header(reader.fieldnames)
        parsed: list[dict[str, str]] = []
        for index, row in enumerate(reader, start=2):
            if row_is_empty(row):
                continue
            assert_result_if_present(row, index)
            parsed.append({key: (value if value is not None else "") for key, value in row.items()})
        return parsed


def _upsert_season(cur, competition_code: str, start_year: int) -> int:
    meta = COMPETITIONS[competition_code]
    cur.execute(
        """
        INSERT INTO competitions (code, name, country)
        VALUES (%s, %s, %s)
        ON CONFLICT (code) DO UPDATE
            SET name = EXCLUDED.name,
                country = EXCLUDED.country
        RETURNING id
        """,
        (competition_code, meta["name"], meta["country"]),
    )
    competition_id = cur.fetchone()[0]
    cur.execute(
        """
        INSERT INTO seasons (competition_id, start_year, name)
        VALUES (%s, %s, %s)
        ON CONFLICT (competition_id, start_year) DO UPDATE
            SET name = EXCLUDED.name
        RETURNING id
        """,
        (competition_id, start_year, season_name(start_year)),
    )
    return cur.fetchone()[0]


def _insert_payloads(
    cur,
    season_id: int,
    source_url: str,
    source_file: str,
    rows: list[dict[str, str]],
) -> int:
    inserted = 0
    for row_number, payload in enumerate(rows, start=1):
        cur.execute(
            """
            INSERT INTO raw_match_payloads
                (season_id, source_url, source_file, row_number, payload)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (source_file, row_number) DO UPDATE SET
                payload = EXCLUDED.payload,
                source_url = EXCLUDED.source_url,
                season_id = EXCLUDED.season_id
            RETURNING (xmax = 0) AS inserted
            """,
            (season_id, source_url, source_file, row_number, Jsonb(payload)),
        )
        if cur.fetchone()[0]:
            inserted += 1
    return inserted


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Ingest one competition-season CSV into raw storage.")
    parser.add_argument("--competition", default="E0", help="football-data.co.uk division code")
    parser.add_argument("--start-year", type=int, default=2018, help="First season start year, e.g. 2018 for 2018/19")
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last season start year (inclusive). Default: current Premier League season.",
    )
    parser.add_argument(
        "--from-file",
        type=Path,
        default=None,
        help="Skip HTTP download and load this CSV (single season only)",
    )
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    if args.from_file is not None:
        ingest_season(
            args.competition,
            args.start_year,
            from_file=args.from_file,
            force_download=args.force_download,
        )
        return
    ingest_seasons(
        args.competition,
        args.start_year,
        end_year=args.end_year,
        force_download=True,
    )


if __name__ == "__main__":
    main()
