"""Read-only helpers for the prediction dashboard. No model fitting."""

from __future__ import annotations

CLASS_LABELS = {0: "away", 1: "draw", 2: "home"}

ALGORITHM_LABELS = {
    "logistic_regression": "Logistic regression",
    "logistic_regression_balanced": "Logistic regression (class-weighted)",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "poisson_dixon_coles": "Poisson / Dixon–Coles",
}


def as_percent(value: float) -> int:
    """Nearest whole-number percentage for display."""
    return int(round(float(value) * 100))


def predicted_outcome(predicted_class: int, home: str, away: str) -> str:
    if int(predicted_class) == 2:
        return home
    if int(predicted_class) == 0:
        return away
    return "Draw"


def actual_outcome(result_code: int | None, home: str, away: str) -> str | None:
    if result_code is None:
        return None
    return predicted_outcome(int(result_code), home, away)


def algorithm_label(algorithm: str) -> str:
    return ALGORITHM_LABELS.get(algorithm, algorithm.replace("_", " ").title())


def match_note(p_home: float, p_draw: float, p_away: float, predicted_class: int) -> str | None:
    """Display-only closeness flag. Never changes the predicted class."""
    home = float(p_home)
    draw = float(p_draw)
    away = float(p_away)
    ranked = sorted((away, draw, home), reverse=True)
    top, second = ranked[0], ranked[1]
    if int(predicted_class) != 1 and draw >= 0.25:
        return "Draw watch"
    if (top - second) < 0.08:
        return "Close match"
    return None


def paginate(items: list, page: int = 1, page_size: int = 16) -> dict:
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 1
    page = max(1, page)
    try:
        page_size = int(page_size)
    except (TypeError, ValueError):
        page_size = 16
    page_size = min(20, max(10, page_size))
    total = len(items)
    pages = max(1, (total + page_size - 1) // page_size) if total else 1
    page = min(page, pages)
    start = (page - 1) * page_size
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": pages,
        "items": items[start : start + page_size],
    }


def ordinal(value: int | None) -> str | None:
    if value is None:
        return None
    number = int(value)
    if 10 <= (number % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def competition_label(name: str | None) -> str:
    text = (name or "").strip() or "Premier League"
    if text.casefold() in {"english premier league", "e0"}:
        return "Premier League"
    return text


def result_side_label(result_code: int | None) -> str | None:
    if result_code is None:
        return None
    return {0: "Away win", 1: "Draw", 2: "Home win"}.get(int(result_code))


def _parse_stat_int(payload: dict, key: str) -> int | None:
    raw = payload.get(key)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


# football-data.co.uk match-stat columns actually present in historical CSVs.
# Possession, passes, pass accuracy, and offsides are listed for completeness
# but are omitted from the response unless those keys exist with real values.
MATCH_STAT_FIELDS = (
    ("shots", "Shots", "HS", "AS", "higher"),
    ("shots_on_target", "Shots on target", "HST", "AST", "higher"),
    ("possession", "Possession", "HPoss", "APoss", "higher"),
    ("passes", "Passes", "HPass", "APass", "higher"),
    ("pass_accuracy", "Pass accuracy", "HPassAcc", "APassAcc", "higher"),
    ("fouls", "Fouls", "HF", "AF", "lower"),
    ("yellow_cards", "Yellow cards", "HY", "AY", "lower"),
    ("red_cards", "Red cards", "HR", "AR", "lower"),
    ("offsides", "Offsides", "HOff", "AOff", "lower"),
    ("corners", "Corners", "HC", "AC", "higher"),
)


def extract_match_stats(payload: dict | None) -> list[dict]:
    """Return only stats that exist on the raw CSV row. Never invents values."""
    if not payload:
        return []
    rows: list[dict] = []
    for key, label, home_key, away_key, better in MATCH_STAT_FIELDS:
        home = _parse_stat_int(payload, home_key)
        away = _parse_stat_int(payload, away_key)
        if home is None or away is None:
            continue
        leader = None
        if home != away:
            if better == "higher":
                leader = "home" if home > away else "away"
            else:
                leader = "home" if home < away else "away"
        rows.append(
            {
                "key": key,
                "label": label,
                "home": home,
                "away": away,
                "leader": leader,
            }
        )
    return rows


def _meeting_date(row: dict) -> str:
    value = row.get("kickoff_date") or row.get("match_date")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def filter_prior_h2h(
    meetings: list[dict],
    *,
    fixture_date: str,
    fixture_match_id: int,
    home_team: str,
    away_team: str,
    limit: int = 5,
) -> list[dict]:
    """Keep played meetings of the same pair strictly before the fixture date."""
    pair = {home_team, away_team}
    prior = []
    for row in meetings:
        if not row.get("is_played"):
            continue
        if int(row["match_id"]) == int(fixture_match_id):
            continue
        if _meeting_date(row) >= str(fixture_date):
            continue
        if {row["home_team"], row["away_team"]} != pair:
            continue
        prior.append(row)
    prior.sort(key=lambda item: (_meeting_date(item), str(item.get("kickoff_time") or "")), reverse=True)
    return prior[:limit]


def group_by_date(fixtures: list[dict]) -> list[dict]:
    groups: list[dict] = []
    for item in fixtures:
        date_key = item["kickoff_date"]
        if not groups or groups[-1]["date"] != date_key:
            groups.append({"date": date_key, "fixtures": []})
        groups[-1]["fixtures"].append(item)
    return groups
