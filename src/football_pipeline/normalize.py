"""Pure transforms for curated match rows. No database I/O."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

RESULT_CODE = {"A": 0, "D": 1, "H": 2}

# football-data.co.uk spellings -> one club identity.
# Keys are casefolded. Unlisted names are kept as stripped source text.
TEAM_ALIASES = {
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "man united": "Manchester United",
    "manchester united": "Manchester United",
    "manchester utd": "Manchester United",
    "nott'm forest": "Nottingham Forest",
    "nottingham forest": "Nottingham Forest",
    "wolves": "Wolverhampton Wanderers",
    "tottenham": "Tottenham Hotspur",
    "spurs": "Tottenham Hotspur",
    "newcastle": "Newcastle United",
    "west ham": "West Ham United",
    "brighton": "Brighton & Hove Albion",
    "man utd": "Manchester United",
    "hull": "Hull City",
    "ipswich": "Ipswich Town",
    "leeds": "Leeds United",
    "leeds united": "Leeds United",
    "sunderland": "Sunderland",
    "coventry": "Coventry City",
    # La Liga
    "ath madrid": "Atletico Madrid",
    "atletico madrid": "Atletico Madrid",
    "atlético de madrid": "Atletico Madrid",
    "atletico de madrid": "Atletico Madrid",
    "ath bilbao": "Athletic Club",
    "athletic club": "Athletic Club",
    "espanol": "Espanyol",
    "rcd espanyol de barcelona": "Espanyol",
    "sociedad": "Real Sociedad",
    "vallecano": "Rayo Vallecano",
    "betis": "Real Betis",
    "celta": "Celta Vigo",
    "sp gijon": "Sporting Gijon",
    "la coruna": "Deportivo La Coruna",
    "rc deportivo": "Deportivo La Coruna",
    "fc barcelona": "Barcelona",
    "ca osasuna": "Osasuna",
    "deportivo alavés": "Alaves",
    "deportivo alaves": "Alaves",
    "elche cf": "Elche",
    "getafe cf": "Getafe",
    "levante ud": "Levante",
    "málaga cf": "Malaga",
    "malaga cf": "Malaga",
    "málaga": "Malaga",
    "malaga": "Malaga",
    "r. racing club": "Racing Santander",
    "racing santander": "Racing Santander",
    "santander": "Racing Santander",
    "sevilla fc": "Sevilla",
    "valencia cf": "Valencia",
    "villarreal cf": "Villarreal",
    # Bundesliga
    "bayern munich": "Bayern Munich",
    "fc bayern münchen": "Bayern Munich",
    "fc bayern munchen": "Bayern Munich",
    "dortmund": "Borussia Dortmund",
    "m'gladbach": "Borussia Mönchengladbach",
    "ein frankfurt": "Eintracht Frankfurt",
    "leverkusen": "Bayer Leverkusen",
    "bayer 04 leverkusen": "Bayer Leverkusen",
    "fc koln": "FC Köln",
    "1. fc köln": "FC Köln",
    "1. fc koln": "FC Köln",
    "mainz": "Mainz 05",
    "1. fsv mainz 05": "Mainz 05",
    "hertha": "Hertha BSC",
    "1. fc union berlin": "Union Berlin",
    "union berlin": "Union Berlin",
    "fc augsburg": "Augsburg",
    "augsburg": "Augsburg",
    "fc schalke 04": "Schalke 04",
    "schalke 04": "Schalke 04",
    "hamburger sv": "Hamburg",
    "hamburg": "Hamburg",
    "sc paderborn 07": "Paderborn",
    "paderborn": "Paderborn",
    "sv elversberg": "Elversberg",
    "elversberg": "Elversberg",
    "sv werder bremen": "Werder Bremen",
    "werder bremen": "Werder Bremen",
    "sport-club freiburg": "Freiburg",
    "freiburg": "Freiburg",
    "tsg hoffenheim": "Hoffenheim",
    "hoffenheim": "Hoffenheim",
    "vfb stuttgart": "Stuttgart",
    "stuttgart": "Stuttgart",
    "rb leipzig": "RB Leipzig",
    "leipzig": "RB Leipzig",
    "gladbach": "Borussia Mönchengladbach",
    "mgladbach": "Borussia Mönchengladbach",
    "koln": "FC Köln",
    "cologne": "FC Köln",
    # Serie A
    "milan": "AC Milan",
    "ac milan": "AC Milan",
    "inter": "Internazionale",
    "roma": "AS Roma",
    "as roma": "AS Roma",
    "verona": "Hellas Verona",
    # Ligue 1
    "paris sg": "Paris Saint-Germain",
    "psg": "Paris Saint-Germain",
    "marseille": "Olympique Marseille",
    "olympique de marseille": "Olympique Marseille",
    "lyon": "Olympique Lyon",
    "olympique lyonnais": "Olympique Lyon",
    "st etienne": "Saint-Étienne",
    "aj auxerre": "Auxerre",
    "auxerre": "Auxerre",
    "as monaco": "Monaco",
    "monaco": "Monaco",
    "angers sco": "Angers",
    "angers": "Angers",
    "estac troyes": "Troyes",
    "troyes": "Troyes",
    "fc lorient": "Lorient",
    "lorient": "Lorient",
    "havre athletic club": "Le Havre",
    "le havre": "Le Havre",
    "losc lille": "Lille",
    "lille": "Lille",
    "le mans fc": "Le Mans",
    "le mans": "Le Mans",
    "ogc nice": "Nice",
    "nice": "Nice",
    "rc lens": "Lens",
    "lens": "Lens",
    "rc strasbourg alsace": "Strasbourg",
    "strasbourg": "Strasbourg",
    "stade brestois 29": "Brest",
    "brest": "Brest",
    "stade rennais fc": "Rennes",
    "rennes": "Rennes",
    "toulouse fc": "Toulouse",
    "toulouse": "Toulouse",
    "paris fc": "Paris FC",
}


class TransformError(Exception):
    """A raw payload cannot become a matches row."""


@dataclass(frozen=True)
class CuratedMatch:
    match_date: date
    kickoff_time: time | None
    home_source_name: str
    away_source_name: str
    home_canonical_name: str
    away_canonical_name: str
    home_goals: int | None
    away_goals: int | None
    result: str | None
    result_code: int | None
    is_played: bool
    source_row_hash: str


def canonical_team_name(source_name: str) -> str:
    cleaned = " ".join(source_name.split())
    if not cleaned:
        raise TransformError("Team name is empty")
    return TEAM_ALIASES.get(cleaned.casefold(), cleaned)


def parse_match_date(value: str) -> date:
    text = value.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise TransformError(f"Unrecognised date {value!r}; expected DD/MM/YYYY or DD/MM/YY")


def parse_kickoff_time(value: str | None) -> time | None:
    text = (value or "").strip()
    if not text:
        return None
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    raise TransformError(f"Unrecognised time {value!r}; expected HH:MM")


def parse_goal(value: str | None) -> int | None:
    text = (value or "").strip()
    if text == "":
        return None
    try:
        goals = int(text)
    except ValueError as exc:
        raise TransformError(f"Goals must be an integer (got {value!r})") from exc
    if goals < 0:
        raise TransformError(f"Goals cannot be negative (got {goals})")
    return goals


def result_from_goals(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _field(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None:
        return ""
    return str(value)


def transform_payload(payload: dict[str, Any]) -> CuratedMatch:
    home_source = " ".join(_field(payload, "HomeTeam").split())
    away_source = " ".join(_field(payload, "AwayTeam").split())
    home_canonical = canonical_team_name(home_source)
    away_canonical = canonical_team_name(away_source)
    if home_canonical == away_canonical:
        raise TransformError(f"Home and away team are the same ({home_canonical})")

    match_date = parse_match_date(_field(payload, "Date"))
    kickoff_time = parse_kickoff_time(_field(payload, "Time"))
    home_goals = parse_goal(_field(payload, "FTHG"))
    away_goals = parse_goal(_field(payload, "FTAG"))
    raw_result = _field(payload, "FTR").strip().upper()

    if (home_goals is None) != (away_goals is None):
        raise TransformError("FTHG and FTAG must both be present or both empty")

    if home_goals is None:
        is_played = False
        result = None
        result_code = None
        if raw_result:
            raise TransformError("FTR is set but full-time goals are missing")
    else:
        is_played = True
        result = result_from_goals(home_goals, away_goals)
        if raw_result and raw_result != result:
            raise TransformError(
                f"FTR {raw_result!r} does not match goals {home_goals}-{away_goals} ({result})"
            )
        result_code = RESULT_CODE[result]

    digest = hashlib.sha256(
        json.dumps(
            {
                "date": match_date.isoformat(),
                "home": home_canonical,
                "away": away_canonical,
                "fthg": home_goals,
                "ftag": away_goals,
                "ftr": result,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    return CuratedMatch(
        match_date=match_date,
        kickoff_time=kickoff_time,
        home_source_name=home_source,
        away_source_name=away_source,
        home_canonical_name=home_canonical,
        away_canonical_name=away_canonical,
        home_goals=home_goals,
        away_goals=away_goals,
        result=result,
        result_code=result_code,
        is_played=is_played,
        source_row_hash=digest,
    )
