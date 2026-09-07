from datetime import date

import pytest

from football_pipeline.football_data import (
    IngestError,
    current_season_start_year,
    resolve_competition,
    resolve_from_file,
    resolve_ingest_end_year,
    resolve_ingest_start_year,
    season_code,
    season_csv_url,
)
from football_pipeline.ingest import ingest_seasons
from football_pipeline.validation import ValidationError, assert_header, assert_result_if_present, row_is_empty


def test_season_code_maps_start_year_to_folder_name():
    assert season_code(2023) == "2324"
    assert season_code(1999) == "9900"
    assert season_code(2018) == "1819"
    assert season_code(2026) == "2627"


def test_season_csv_url_uses_apex_host_first():
    assert season_csv_url("E0", 2023) == "https://football-data.co.uk/mmz4281/2324/E0.csv"


def test_current_season_start_year_crosses_calendar_year():
    assert current_season_start_year(date(2026, 8, 1)) == 2026
    assert current_season_start_year(date(2026, 9, 7)) == 2026
    assert current_season_start_year(date(2027, 1, 15)) == 2026
    assert current_season_start_year(date(2027, 5, 15)) == 2026
    assert current_season_start_year(date(2027, 7, 31)) == 2026
    assert current_season_start_year(date(2026, 7, 31)) == 2025
    assert current_season_start_year(date(2027, 8, 1)) == 2027


def test_default_trigger_params_cover_2018_through_current_season():
    today = date(2027, 5, 15)
    assert resolve_competition("") == "E0"
    assert resolve_ingest_start_year(None) == 2018
    assert resolve_ingest_start_year("auto") == 2018
    assert resolve_ingest_end_year("auto", today=today) == 2026
    assert resolve_ingest_end_year("", today=today) == 2026
    assert resolve_ingest_end_year(None, today=today) == 2026
    assert resolve_from_file("") is None
    assert resolve_from_file("  ") is None
    assert resolve_from_file(None) is None


def test_trigger_year_overrides_are_kept_for_backfills():
    today = date(2027, 5, 15)
    assert resolve_ingest_start_year(2020) == 2020
    assert resolve_ingest_end_year("2024", today=today) == 2024
    assert resolve_ingest_end_year(2024, today=today) == 2024
    assert resolve_from_file("/opt/project/tests/fixtures/E0_2324_opening.csv").endswith(
        "E0_2324_opening.csv"
    )


def test_invalid_year_override_is_rejected():
    with pytest.raises(IngestError, match="Invalid year"):
        resolve_ingest_end_year("yesterday")


def test_ingest_seasons_rejects_reversed_range():
    with pytest.raises(IngestError, match="before start_year"):
        ingest_seasons("E0", 2024, end_year=2018)


def test_header_requires_result_columns():
    with pytest.raises(ValidationError, match="missing required columns"):
        assert_header(["Div", "Date", "HomeTeam"])


def test_empty_row_detected():
    assert row_is_empty({"Div": "", "Date": "  "})
    assert not row_is_empty({"Div": "E0"})


def test_invalid_full_time_result_rejected():
    with pytest.raises(ValidationError, match="FTR"):
        assert_result_if_present({"FTR": "W"}, row_number=3)
