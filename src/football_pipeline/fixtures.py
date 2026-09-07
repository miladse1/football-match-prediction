"""Upcoming Premier League fixtures. Separate from football-data.co.uk results ingest.

Source: fixturedownload.com JSON for the current EPL season. Fixture scores in
that JSON are ignored. Official results stay owned by football-data.co.uk.
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from datetime import datetime, timezone

from psycopg.types.json import Jsonb

from football_pipeline.config import FIXTURES_URL, RAW_DATA_DIR
from football_pipeline.db import connect
from football_pipeline.football_data import USER_AGENT, current_season_start_year, season_name
from football_pipeline.ingest import COMPETITIONS, _insert_payloads, _upsert_season

logger = logging.getLogger(__name__)


class FixtureError(Exception):
    """Upcoming fixture download or parse failure."""


def _download_json(url: str) -> list[dict]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
    except Exception as exc:  # noqa: BLE001 — surface as FixtureError
        raise FixtureError(f"Could not download fixtures from {url}: {exc}") from exc
    stripped = body.lstrip()
    if stripped.startswith(b"<"):
        raise FixtureError(f"{url} returned HTML instead of JSON")
    data = json.loads(body)
    if not isinstance(data, list):
        raise FixtureError("Fixture payload is not a JSON list")
    return data


def _parse_kickoff(value: str) -> tuple[str, str]:
    raw = value.strip().replace("T", " ")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+0000"
    elif raw.endswith("+00:00"):
        raw = raw[:-6] + "+0000"
    stamp = None
    for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            stamp = datetime.strptime(raw, fmt)
            break
        except ValueError:
            continue
    if stamp is None:
        raise FixtureError(f"Unrecognised fixture DateUtc: {value!r}")
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    utc = stamp.astimezone(timezone.utc)
    return utc.strftime("%d/%m/%Y"), utc.strftime("%H:%M")


def fixtures_to_payloads(rows: list[dict]) -> list[dict[str, str]]:
    """Every listed fixture is stored unplayed. Official scores come from results ingest."""
    payloads: list[dict[str, str]] = []
    for row in rows:
        match_date, kickoff = _parse_kickoff(str(row["DateUtc"]))
        payloads.append(
            {
                "Div": "E0",
                "Date": match_date,
                "Time": kickoff,
                "HomeTeam": str(row["HomeTeam"]).strip(),
                "AwayTeam": str(row["AwayTeam"]).strip(),
                "FTHG": "",
                "FTAG": "",
                "FTR": "",
                "MatchNumber": str(row.get("MatchNumber") or ""),
                "RoundNumber": str(row.get("RoundNumber") or ""),
            }
        )
    return payloads


def ingest_upcoming_fixtures(
    *,
    competition: str = "E0",
    start_year: int | None = None,
    url: str | None = None,
) -> dict:
    if competition not in COMPETITIONS:
        raise FixtureError(f"Unknown competition {competition}")
    year = current_season_start_year() if start_year is None else int(start_year)
    source_url = url or FIXTURES_URL
    raw_rows = _download_json(source_url)
    payloads = fixtures_to_payloads(raw_rows)
    source_file = f"{competition}_{year}-{year + 1}_fixtures.json"
    dest = RAW_DATA_DIR / source_file
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payloads, indent=2), encoding="utf-8")

    with connect() as conn:
        with conn.cursor() as cur:
            season_id = _upsert_season(cur, competition, year)
            inserted = _insert_payloads(cur, season_id, source_url, source_file, payloads)
        conn.commit()

    summary = {
        "source": source_url,
        "season": season_name(year),
        "fixture_rows": len(raw_rows),
        "unplayed_ingested": len(payloads),
        "payloads_inserted": inserted,
        "payloads_updated": len(payloads) - inserted,
    }
    logger.info(
        "Fixtures %s: %s listed, %s unplayed stored in raw payloads (load_matches runs separately)",
        summary["season"],
        summary["fixture_rows"],
        summary["unplayed_ingested"],
    )
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Ingest unplayed Premier League fixtures.")
    parser.add_argument("--competition", default="E0")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--url", default=None)
    args = parser.parse_args()
    ingest_upcoming_fixtures(competition=args.competition, start_year=args.start_year, url=args.url)


if __name__ == "__main__":
    main()
