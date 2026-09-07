from datetime import date, time

import pytest

from football_pipeline.normalize import TransformError, canonical_team_name, transform_payload


def test_aliases_collapse_common_spellings():
    assert canonical_team_name("Man City") == "Manchester City"
    assert canonical_team_name("Nott'm Forest") == "Nottingham Forest"
    assert canonical_team_name("  Arsenal  ") == "Arsenal"


def test_dd_mm_yyyy_and_two_digit_year_parse():
    long_year = transform_payload(
        {
            "Date": "12/08/2023",
            "Time": "15:00",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Nott'm Forest",
            "FTHG": "2",
            "FTAG": "1",
            "FTR": "H",
        }
    )
    short_year = transform_payload(
        {
            "Date": "12/08/23",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "FTHG": "0",
            "FTAG": "0",
            "FTR": "D",
        }
    )
    assert long_year.match_date == date(2023, 8, 12)
    assert long_year.kickoff_time == time(15, 0)
    assert short_year.match_date == date(2023, 8, 12)


def test_result_code_encoding():
    home = transform_payload(
        {"Date": "12/08/2023", "HomeTeam": "A", "AwayTeam": "B", "FTHG": "1", "FTAG": "0", "FTR": "H"}
    )
    draw = transform_payload(
        {"Date": "12/08/2023", "HomeTeam": "A", "AwayTeam": "B", "FTHG": "1", "FTAG": "1", "FTR": "D"}
    )
    away = transform_payload(
        {"Date": "12/08/2023", "HomeTeam": "A", "AwayTeam": "B", "FTHG": "0", "FTAG": "2", "FTR": "A"}
    )
    assert (home.result_code, draw.result_code, away.result_code) == (2, 1, 0)


def test_ftr_must_agree_with_goals():
    with pytest.raises(TransformError, match="does not match goals"):
        transform_payload(
            {
                "Date": "12/08/2023",
                "HomeTeam": "A",
                "AwayTeam": "B",
                "FTHG": "2",
                "FTAG": "1",
                "FTR": "A",
            }
        )


def test_unplayed_match_has_no_result():
    row = transform_payload(
        {
            "Date": "12/08/2023",
            "HomeTeam": "A",
            "AwayTeam": "B",
            "FTHG": "",
            "FTAG": "",
            "FTR": "",
        }
    )
    assert row.is_played is False
    assert row.result is None
    assert row.result_code is None


def test_same_team_both_sides_rejected():
    with pytest.raises(TransformError, match="same"):
        transform_payload(
            {
                "Date": "12/08/2023",
                "HomeTeam": "Man City",
                "AwayTeam": "Manchester City",
                "FTHG": "1",
                "FTAG": "0",
                "FTR": "H",
            }
        )
