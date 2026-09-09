"""Season windows must roll forward on their own, and never leak the holdout."""

from datetime import date

import pytest

from football_pipeline import seasons

# The season boundary is the interesting place: 31 July and 1 August.
BEFORE_ROLLOVER = date(2026, 7, 31)
AFTER_ROLLOVER = date(2026, 8, 1)
MID_SEASON = date(2027, 1, 15)
NEXT_ROLLOVER = date(2027, 8, 1)


def test_live_season_flips_on_the_first_of_august():
    assert seasons.live_season_start_year(BEFORE_ROLLOVER) == 2025
    assert seasons.live_season_start_year(AFTER_ROLLOVER) == 2026
    assert seasons.live_season_start_year(MID_SEASON) == 2026
    assert seasons.live_season_start_year(date(2027, 7, 31)) == 2026
    assert seasons.live_season_start_year(NEXT_ROLLOVER) == 2027


def test_season_end_is_the_thirty_first_of_july():
    assert seasons.season_end(2026) == date(2027, 7, 31)
    assert seasons.season_start(2026) == date(2026, 8, 1)


def test_season_names():
    assert seasons.short_season_name(2026) == "2026/27"
    assert seasons.long_season_name(2026) == "2026/2027"
    assert seasons.short_season_name(1999) == "1999/00"


def test_holdout_is_the_most_recently_completed_season():
    assert seasons.holdout_season_start_year(AFTER_ROLLOVER) == 2025
    assert seasons.holdout_season_name(AFTER_ROLLOVER) == "2025/26"
    assert seasons.holdout_season_start_year(NEXT_ROLLOVER) == 2026
    assert seasons.holdout_season_name(NEXT_ROLLOVER) == "2026/27"


def test_windows_today_match_the_previously_hardcoded_values():
    """The derived windows must reproduce the constants they replaced."""
    today = date(2026, 9, 9)
    assert seasons.production_train_end(today) == date(2025, 7, 31)
    assert seasons.test_end(today) == date(2026, 7, 31)
    assert seasons.valid_end(today) == date(2025, 7, 31)
    assert seasons.train_end(today) == date(2024, 7, 31)
    assert seasons.holdout_season_name(today) == "2025/26"
    assert seasons.walkforward_valid_years(today) == (2021, 2022, 2023, 2024)
    assert seasons.fixtures_url(today).endswith("epl-2026")


def test_windows_move_forward_by_one_season_after_rollover():
    before = seasons.season_windows(date(2027, 7, 31))
    after = seasons.season_windows(date(2027, 8, 1))
    assert before["holdout_season"] == "2025/26"
    assert after["holdout_season"] == "2026/27"
    assert before["production_train_end"] == "2025-07-31"
    assert after["production_train_end"] == "2026-07-31"
    assert before["walkforward_valid_seasons"] == ["2021/22", "2022/23", "2023/24", "2024/25"]
    assert after["walkforward_valid_seasons"] == ["2022/23", "2023/24", "2024/25", "2025/26"]
    assert after["fixtures_url"].endswith("epl-2027")


@pytest.mark.parametrize("today", [BEFORE_ROLLOVER, AFTER_ROLLOVER, MID_SEASON, NEXT_ROLLOVER])
def test_holdout_is_never_a_validation_fold(today):
    """The core leakage invariant, checked at every point in the calendar."""
    holdout = seasons.holdout_season_start_year(today)
    valid_years = seasons.walkforward_valid_years(today)
    assert holdout not in valid_years
    assert max(valid_years) < holdout
    # Every validation window ends on or before the production training cutoff.
    assert seasons.season_end(max(valid_years)) <= seasons.production_train_end(today)
    # And the holdout starts strictly after that cutoff.
    assert seasons.season_end(holdout) > seasons.production_train_end(today)


@pytest.mark.parametrize("today", [BEFORE_ROLLOVER, AFTER_ROLLOVER, MID_SEASON, NEXT_ROLLOVER])
def test_live_season_is_never_in_the_training_or_test_window(today):
    live = seasons.live_season_start_year(today)
    assert seasons.season_start(live) > seasons.test_end(today)


@pytest.mark.parametrize("today", [BEFORE_ROLLOVER, AFTER_ROLLOVER, MID_SEASON, NEXT_ROLLOVER])
def test_split_cutoffs_are_strictly_ordered(today):
    assert seasons.train_end(today) < seasons.valid_end(today) < seasons.test_end(today)


def test_folds_are_contiguous_and_expanding():
    years = seasons.walkforward_valid_years(date(2026, 9, 9))
    assert list(years) == sorted(years)
    assert all(b - a == 1 for a, b in zip(years, years[1:]))


def test_folds_are_clamped_to_available_history():
    """Never validate on a season that has too little history behind it."""
    years = seasons.walkforward_valid_years(
        date(2022, 9, 1), n_folds=10, ingest_start_year=2018
    )
    # Holdout is 2021/22, so folds must stop at 2020/21 and start no earlier
    # than 2020 (2018 and 2019 are needed as training seasons).
    assert years == (2020,)


def test_no_folds_available_raises_a_clear_error():
    with pytest.raises(seasons.SeasonError, match="No walk-forward folds"):
        seasons.walkforward_valid_years(date(2019, 9, 1), ingest_start_year=2018)


def test_n_folds_must_be_positive():
    with pytest.raises(seasons.SeasonError, match="at least 1"):
        seasons.walkforward_valid_years(date(2026, 9, 9), n_folds=0)


def test_fixtures_url_tracks_the_live_season():
    assert seasons.fixtures_url(AFTER_ROLLOVER).endswith("epl-2026")
    assert seasons.fixtures_url(BEFORE_ROLLOVER).endswith("epl-2025")
    assert seasons.fixtures_url(NEXT_ROLLOVER).endswith("epl-2027")
