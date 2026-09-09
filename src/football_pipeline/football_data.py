"""football-data.co.uk season paths and CSV download."""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

from football_pipeline import seasons

logger = logging.getLogger(__name__)

# www currently 503s; the apex host serves the same CSVs.
BASE_URLS = (
    "https://football-data.co.uk/mmz4281",
    "https://www.football-data.co.uk/mmz4281",
)
USER_AGENT = "football-match-prediction/0.1 (portfolio pipeline)"

REQUIRED_COLUMNS = (
    "Div",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
)


class IngestError(Exception):
    """Download or CSV format failure."""


def season_code(start_year: int) -> str:
    """Map 2023 -> '2324' (the folder name football-data.co.uk uses)."""
    end_year = start_year + 1
    return f"{start_year % 100:02d}{end_year % 100:02d}"


def season_name(start_year: int) -> str:
    return seasons.long_season_name(start_year)


def current_season_start_year(today: date | None = None) -> int:
    """Start year of the Premier League season that contains `today`.

    Thin alias kept for callers and tests. The rule itself lives in
    football_pipeline.seasons, which is the single source of truth for every
    season boundary in the project.
    """
    return seasons.live_season_start_year(today)


_AUTO_YEAR_TOKENS = frozenset({"", "auto", "current", "null", "none"})


def parse_year_override(value: object) -> int | None:
    """Return a season start year, or None when the caller asked for the default."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise IngestError(f"Invalid year: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if text.lower() in _AUTO_YEAR_TOKENS:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise IngestError(f"Invalid year: {value!r}") from exc


def resolve_ingest_start_year(
    value: object, *, default: int = seasons.DEFAULT_INGEST_START_YEAR
) -> int:
    parsed = parse_year_override(value)
    return default if parsed is None else parsed


def resolve_ingest_end_year(
    value: object,
    *,
    today: date | None = None,
    env_default: int | None = None,
) -> int:
    """End season start year to ingest. Empty/`auto` means the current PL season."""
    parsed = parse_year_override(value)
    if parsed is not None:
        return parsed
    if env_default is not None:
        return env_default
    return current_season_start_year(today)


def resolve_from_file(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_competition(value: object, *, default: str = "E0") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def season_csv_url(competition_code: str, start_year: int, *, base: str | None = None) -> str:
    root = base or BASE_URLS[0]
    return f"{root}/{season_code(start_year)}/{competition_code}.csv"


def season_csv_urls(competition_code: str, start_year: int) -> tuple[str, ...]:
    return tuple(season_csv_url(competition_code, start_year, base=base) for base in BASE_URLS)


def download_csv(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Download %s -> %s", url, dest)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise IngestError(f"Could not download {url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise IngestError(f"Could not download {url}: {exc}") from exc

    stripped = body.lstrip()
    if stripped.startswith(b"<") or stripped.startswith(b"<!DOCTYPE"):
        raise IngestError(
            f"{url} returned HTML instead of CSV (site may be down). "
            "Re-run later, or pass --from-file with a local CSV."
        )
    dest.write_bytes(body)


def download_season_csv(competition_code: str, start_year: int, dest: Path) -> str:
    """Try known hosts until one returns a CSV. Returns the URL that worked."""
    errors: list[str] = []
    for url in season_csv_urls(competition_code, start_year):
        try:
            download_csv(url, dest)
            return url
        except IngestError as exc:
            logger.warning("%s", exc)
            errors.append(str(exc))
    raise IngestError("All football-data hosts failed: " + " | ".join(errors))
