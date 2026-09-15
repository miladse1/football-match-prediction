from datetime import date

import pytest

from football_pipeline.competitions import (
    COMPETITIONS,
    CompetitionError,
    get,
    parse_competition,
    parse_competition_list,
    processed_dir,
)


def test_catalog_covers_the_five_requested_leagues():
    assert tuple(COMPETITIONS) == ("E0", "SP1", "D1", "I1", "F1")
    assert COMPETITIONS["E0"].name == "Premier League"
    assert COMPETITIONS["SP1"].n_matches == 380
    assert COMPETITIONS["D1"].n_teams == 18
    assert COMPETITIONS["D1"].n_matches == 306
    assert COMPETITIONS["I1"].n_teams == 20
    assert COMPETITIONS["F1"].n_teams == 18


def test_ligue_1_size_changes_in_2023():
    ligue = COMPETITIONS["F1"]
    assert ligue.expected_teams(2018) == 20
    assert ligue.expected_matches(2018) == 380
    assert ligue.expected_teams(2019) == 20
    assert ligue.expected_matches(2019) == 279
    assert ligue.expected_teams(2022) == 20
    assert ligue.expected_matches(2022) == 380
    assert ligue.expected_teams(2023) == 18
    assert ligue.expected_matches(2023) == 306
    assert ligue.expected_teams(2026) == 18


def test_parse_competition_accepts_slugs_and_codes():
    assert parse_competition("e0") == "E0"
    assert parse_competition("la-liga") == "SP1"
    assert parse_competition("bundesliga") == "D1"
    assert parse_competition(None) == "E0"


def test_unknown_competition_is_rejected():
    with pytest.raises(CompetitionError):
        get("XX")


def test_parse_competition_list_all_token():
    assert parse_competition_list("all") == ("E0", "SP1", "D1", "I1", "F1")
    assert parse_competition_list("SP1") == ("SP1",)
    assert parse_competition_list("E0,D1") == ("E0", "D1")


def test_premier_league_keeps_legacy_processed_paths(tmp_path):
    assert processed_dir("E0", root=tmp_path) == tmp_path / "data" / "processed"
    assert processed_dir("SP1", root=tmp_path) == tmp_path / "data" / "processed" / "competitions" / "SP1"


def test_fixture_urls_are_per_competition():
    from football_pipeline.competitions import fixtures_url_template
    from football_pipeline.seasons import fixtures_url

    assert fixtures_url_template("E0").endswith("epl-{start_year}")
    assert fixtures_url_template("SP1").endswith("la-liga-{start_year}")
    assert fixtures_url_template("D1").endswith("bundesliga-{start_year}")
    assert fixtures_url_template("I1").endswith("serie-a-{start_year}")
    assert fixtures_url_template("F1").endswith("ligue-1-{start_year}")
    assert fixtures_url(date(2026, 9, 1), competition="D1").endswith("bundesliga-2026")
