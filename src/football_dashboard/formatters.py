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


def group_by_date(fixtures: list[dict]) -> list[dict]:
    groups: list[dict] = []
    for item in fixtures:
        date_key = item["kickoff_date"]
        if not groups or groups[-1]["date"] != date_key:
            groups.append({"date": date_key, "fixtures": []})
        groups[-1]["fixtures"].append(item)
    return groups
