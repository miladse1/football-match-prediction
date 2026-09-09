"""Single source of truth for season boundaries and every time window.

A Premier League season runs August to July, so it crosses the calendar year:
7 Sep 2026 and 15 May 2027 are both the 2026/27 season (start year 2026).

Everything downstream is derived from one fact -- which season contains "today":

    live season          the season currently being played (scored, never trained on)
    holdout season       the most recently completed season (scored once, never selected on)
    production train end the last day of the season before the holdout
    walk-forward folds   the N completed seasons before the holdout

Nothing in this module imports from the rest of the package, so it can be used
from config, walk-forward, the simulator and the dashboard without a cycle.
Every function takes an optional ``today`` so the behaviour is testable at any
point in the calendar, especially across the 31 July / 1 August boundary.
"""

from __future__ import annotations

from datetime import date

# A season starts on 1 August and ends on 31 July of the following year.
SEASON_START_MONTH = 8
SEASON_END_MONTH = 7
SEASON_END_DAY = 31

DEFAULT_INGEST_START_YEAR = 2018
DEFAULT_WALKFORWARD_FOLDS = 4

# A walk-forward fold needs earlier seasons to train on. Never validate on a
# season that leaves fewer than this many complete seasons behind it.
MIN_TRAIN_SEASONS = 2

FIXTURES_URL_TEMPLATE = "https://fixturedownload.com/feed/json/epl-{start_year}"


class SeasonError(Exception):
    """The requested season window cannot be built from the available history."""


def season_start(start_year: int) -> date:
    """First day of the season that starts in ``start_year`` (1 August)."""
    return date(int(start_year), SEASON_START_MONTH, 1)


def season_end(start_year: int) -> date:
    """Inclusive last day of the season that starts in ``start_year`` (31 July)."""
    return date(int(start_year) + 1, SEASON_END_MONTH, SEASON_END_DAY)


def short_season_name(start_year: int) -> str:
    """2026 -> '2026/27'. Used for fold names and display."""
    return f"{int(start_year)}/{str(int(start_year) + 1)[2:]}"


def long_season_name(start_year: int) -> str:
    """2026 -> '2026/2027'. Matches the value stored in seasons.name."""
    return f"{int(start_year)}/{int(start_year) + 1}"


def live_season_start_year(today: date | None = None) -> int:
    """Start year of the season that contains ``today``.

    31 Jul 2026 is still 2025/26; 1 Aug 2026 is the first day of 2026/27.
    """
    today = today or date.today()
    return today.year if today.month >= SEASON_START_MONTH else today.year - 1


def completed_season_start_year(today: date | None = None) -> int:
    """Start year of the most recently completed season."""
    return live_season_start_year(today) - 1


def holdout_season_start_year(today: date | None = None) -> int:
    """The untouched test season: the most recently completed one.

    It is scored once after model selection and never used to choose a model,
    tune hyperparameters, pick class weights, or set thresholds.
    """
    return completed_season_start_year(today)


def holdout_season_name(today: date | None = None) -> str:
    return short_season_name(holdout_season_start_year(today))


def holdout_season_end(today: date | None = None) -> date:
    return season_end(holdout_season_start_year(today))


def production_train_end(today: date | None = None) -> date:
    """Last day of permitted training history: the end of the season before the holdout."""
    return season_end(holdout_season_start_year(today) - 1)


def train_end(today: date | None = None) -> date:
    """Inclusive end of the 'train' split label in training_rows."""
    return season_end(holdout_season_start_year(today) - 2)


def valid_end(today: date | None = None) -> date:
    """Inclusive end of the 'valid' split label. Equals the production train cutoff."""
    return production_train_end(today)


def test_end(today: date | None = None) -> date:
    """Inclusive end of the 'test' split label. Matches later than this are live-only."""
    return holdout_season_end(today)


def walkforward_valid_years(
    today: date | None = None,
    *,
    n_folds: int = DEFAULT_WALKFORWARD_FOLDS,
    ingest_start_year: int = DEFAULT_INGEST_START_YEAR,
) -> tuple[int, ...]:
    """Start years of the seasons used as walk-forward validation windows.

    Expanding window: each fold trains on every earlier season and validates on
    the next one. The newest fold validates on the season immediately before the
    holdout, so the holdout can never appear in a validation window.
    """
    if n_folds < 1:
        raise SeasonError("n_folds must be at least 1")
    holdout = holdout_season_start_year(today)
    earliest = int(ingest_start_year) + MIN_TRAIN_SEASONS
    first = max(holdout - n_folds, earliest)
    if first >= holdout:
        raise SeasonError(
            f"No walk-forward folds available: holdout season starts {holdout} but the "
            f"earliest season that leaves {MIN_TRAIN_SEASONS} training seasons behind it "
            f"is {earliest}. Ingest more history or lower n_folds."
        )
    return tuple(range(first, holdout))


def fixtures_url(today: date | None = None, *, template: str = FIXTURES_URL_TEMPLATE) -> str:
    """Fixture feed for the live season."""
    return template.format(start_year=live_season_start_year(today))


def season_windows(today: date | None = None) -> dict:
    """Every derived boundary at once. Useful for logging and pipeline reports."""
    live = live_season_start_year(today)
    holdout = holdout_season_start_year(today)
    valid_years = walkforward_valid_years(today)
    return {
        "as_of": (today or date.today()).isoformat(),
        "live_season_start_year": live,
        "live_season": short_season_name(live),
        "holdout_season_start_year": holdout,
        "holdout_season": short_season_name(holdout),
        "holdout_season_end": holdout_season_end(today).isoformat(),
        "production_train_end": production_train_end(today).isoformat(),
        "train_end": train_end(today).isoformat(),
        "valid_end": valid_end(today).isoformat(),
        "test_end": test_end(today).isoformat(),
        "walkforward_valid_seasons": [short_season_name(year) for year in valid_years],
        "fixtures_url": fixtures_url(today),
    }
