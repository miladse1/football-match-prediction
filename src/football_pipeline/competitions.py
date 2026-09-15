"""Supported competitions. One catalog, reused by ingest, quality, training, forecast, and the dashboard.

Leagues are identified by football-data.co.uk division codes. Season length is
competition-aware (and, for Ligue 1, year-aware) so quality gates never assume
a Premier League 20-club / 380-match season globally.
"""

from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

DEFAULT_COMPETITION = "E0"


class CompetitionError(Exception):
    """Unknown or unsupported competition code."""


@dataclass(frozen=True)
class EuropeanBand:
    """Inclusive finishing positions that count as this European competition.

    These are documented presentation conventions, not a full UEFA-access-list
    model. Cup winners, coefficient extra slots, and relegation play-offs are
    not simulated.
    """

    key: str
    label: str
    min_position: int
    max_position: int


@dataclass(frozen=True)
class Competition:
    code: str
    slug: str
    name: str
    short_name: str
    country: str
    fixture_slug: str
    mark: str
    # Default (modern) size. Overridden per season via size_overrides.
    n_teams: int
    n_matches: int
    relegation_places: int
    ucl: EuropeanBand
    europa: EuropeanBand
    conference: EuropeanBand
    size_overrides: tuple[tuple[int, int, int, int], ...] = ()
    # (first_start_year, last_start_year inclusive, n_teams, n_matches)
    qualification_note: str = ""

    def size_for_season(self, start_year: int) -> tuple[int, int]:
        year = int(start_year)
        for first, last, n_teams, n_matches in self.size_overrides:
            if first <= year <= last:
                return n_teams, n_matches
        return self.n_teams, self.n_matches

    def expected_teams(self, start_year: int) -> int:
        return self.size_for_season(start_year)[0]

    def expected_matches(self, start_year: int) -> int:
        return self.size_for_season(start_year)[1]

    def european_places(self) -> int:
        return self.conference.max_position

    def as_public_dict(self) -> dict:
        return {
            "code": self.code,
            "slug": self.slug,
            "name": self.name,
            "short_name": self.short_name,
            "country": self.country,
            "mark": self.mark,
            "n_teams": self.n_teams,
            "n_matches": self.n_matches,
            "relegation_places": self.relegation_places,
            "ucl_label": self.ucl.label,
            "ucl_positions": [self.ucl.min_position, self.ucl.max_position],
            "europa_label": self.europa.label,
            "conference_label": self.conference.label,
            "qualification_note": self.qualification_note,
        }


_UCL_1_4 = EuropeanBand("ucl", "Champions League", 1, 4)
_UCL_1_3 = EuropeanBand("ucl", "Champions League", 1, 3)
_EL_5 = EuropeanBand("europa", "Europa League", 5, 5)
_EL_4 = EuropeanBand("europa", "Europa League", 4, 4)
_ECL_6 = EuropeanBand("conference", "Conference League", 6, 6)
_ECL_5 = EuropeanBand("conference", "Conference League", 5, 5)

_TWENTY_NOTE = (
    "European places are a simple league-table convention: Champions League 1–4, "
    "Europa League 5, Conference League 6. Cup winners, extra UEFA coefficient "
    "slots, and play-offs are not modelled."
)
_BUNDESLIGA_NOTE = (
    "Bundesliga convention: Champions League 1–4, Europa League 5, Conference League 6. "
    "Relegation is the bottom two automatic places. The 16th-place relegation play-off "
    "is not modelled separately."
)
_LIGUE1_NOTE = (
    "Ligue 1 convention (18-club era): Champions League 1–3, Europa League 4, "
    "Conference League 5. Relegation is the bottom two automatic places. The "
    "relegation play-off is not modelled separately."
)


COMPETITIONS: dict[str, Competition] = {
    "E0": Competition(
        code="E0",
        slug="premier-league",
        name="Premier League",
        short_name="PL",
        country="England",
        fixture_slug="epl",
        mark="PL",
        n_teams=20,
        n_matches=380,
        relegation_places=3,
        ucl=_UCL_1_4,
        europa=_EL_5,
        conference=_ECL_6,
        qualification_note=_TWENTY_NOTE,
    ),
    "SP1": Competition(
        code="SP1",
        slug="la-liga",
        name="La Liga",
        short_name="LL",
        country="Spain",
        fixture_slug="la-liga",
        mark="LL",
        n_teams=20,
        n_matches=380,
        relegation_places=3,
        ucl=_UCL_1_4,
        europa=_EL_5,
        conference=_ECL_6,
        qualification_note=_TWENTY_NOTE,
    ),
    "D1": Competition(
        code="D1",
        slug="bundesliga",
        name="Bundesliga",
        short_name="BL",
        country="Germany",
        fixture_slug="bundesliga",
        mark="BL",
        n_teams=18,
        n_matches=306,
        relegation_places=2,
        ucl=_UCL_1_4,
        europa=_EL_5,
        conference=_ECL_6,
        qualification_note=_BUNDESLIGA_NOTE,
    ),
    "I1": Competition(
        code="I1",
        slug="serie-a",
        name="Serie A",
        short_name="SA",
        country="Italy",
        fixture_slug="serie-a",
        mark="SA",
        n_teams=20,
        n_matches=380,
        relegation_places=3,
        ucl=_UCL_1_4,
        europa=_EL_5,
        conference=_ECL_6,
        qualification_note=_TWENTY_NOTE,
    ),
    "F1": Competition(
        code="F1",
        slug="ligue-1",
        name="Ligue 1",
        short_name="L1",
        country="France",
        fixture_slug="ligue-1",
        mark="L1",
        n_teams=18,
        n_matches=306,
        relegation_places=2,
        ucl=_UCL_1_3,
        europa=_EL_4,
        conference=_ECL_5,
        # 2018/19 and 2020/21–2022/23 were 20-club / 380-match seasons.
        # 2019/20 was abandoned during COVID; football-data.co.uk publishes 279
        # completed matches and no remaining fixtures for that season.
        # 2023/24 onward is 18 clubs.
        size_overrides=(
            (2018, 2018, 20, 380),
            (2019, 2019, 20, 279),
            (2020, 2022, 20, 380),
        ),
        qualification_note=_LIGUE1_NOTE,
    ),
}

SUPPORTED_CODES: tuple[str, ...] = tuple(COMPETITIONS)
SLUG_TO_CODE = {spec.slug: spec.code for spec in COMPETITIONS.values()}
ALL_TOKEN = "all"


def get(code: str) -> Competition:
    spec = COMPETITIONS.get(str(code).strip().upper())
    if spec is None:
        known = ", ".join(SUPPORTED_CODES)
        raise CompetitionError(f"Unknown competition {code!r}. Known: {known}")
    return spec


def ingest_meta() -> dict[str, dict[str, str]]:
    """Shape expected by the historical ingest helpers."""
    return {
        code: {"name": spec.name, "country": spec.country, "slug": spec.slug}
        for code, spec in COMPETITIONS.items()
    }


def parse_competition(value: object, *, default: str = DEFAULT_COMPETITION) -> str:
    """Return one supported division code."""
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    folded = text.casefold()
    if folded in SLUG_TO_CODE:
        return SLUG_TO_CODE[folded]
    return get(text).code


def parse_competition_list(value: object, *, default: tuple[str, ...] = SUPPORTED_CODES) -> tuple[str, ...]:
    """Return one or more codes. ``all`` (the scheduled default) means every supported league."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        codes = tuple(parse_competition(item) for item in value if str(item).strip())
        return codes or default
    text = str(value).strip()
    if not text or text.casefold() in {ALL_TOKEN, "top5", "top-5"}:
        return tuple(SUPPORTED_CODES)
    if "," in text:
        return tuple(parse_competition(part) for part in text.split(",") if part.strip())
    return (parse_competition(text),)


def public_catalog() -> list[dict]:
    return [spec.as_public_dict() for spec in COMPETITIONS.values()]


def fixtures_url_template(code: str) -> str:
    return f"https://fixturedownload.com/feed/json/{get(code).fixture_slug}-{{start_year}}"


def processed_dir(code: str, *, root: Path) -> Path:
    """On-disk reports for one league.

    Premier League keeps the historical ``data/processed`` paths so existing
    artifacts and dashboard readers stay valid. Other leagues nest under
    ``data/processed/competitions/<code>``.
    """
    spec = get(code)
    if spec.code == DEFAULT_COMPETITION:
        return root / "data" / "processed"
    return root / "data" / "processed" / "competitions" / spec.code
