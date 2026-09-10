"""Postgres reads for the dashboard. Does not fit or score models."""

from __future__ import annotations

import json
from datetime import date, time

import numpy as np

from football_dashboard.crests import team_badge
from football_dashboard.formatters import (
    actual_outcome,
    algorithm_label,
    as_percent,
    competition_label,
    extract_match_stats,
    filter_prior_h2h,
    group_by_date,
    match_note,
    ordinal,
    paginate,
    predicted_outcome,
    result_side_label,
)
from football_pipeline import seasons
from football_pipeline.config import INGEST_START_YEAR, ROOT
from football_pipeline.db import connect
from football_pipeline.registry import production_model as _registry_production_model

REPORT_PATH = ROOT / "data" / "processed" / "model_metrics.json"
# Derived, not pinned: the live season rolls over on 1 August.
LIVE_START_YEAR = seasons.live_season_start_year()
LIVE_SEASON = seasons.short_season_name(LIVE_START_YEAR)


def _selected_algorithm() -> str:
    """Production algorithm name. Resolved by football_pipeline.registry."""
    return _registry_production_model().algorithm


def _iso_date(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _iso_time(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, time):
        return value.strftime("%H:%M")
    text = str(value)
    return text[:5] if len(text) >= 5 else text


def _iso_stamp(value) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _has_prediction(row: dict) -> bool:
    return row.get("predicted_class") is not None and row.get("p_home") is not None


def _serialize_fixture(row: dict, *, include_result: bool, include_prediction: bool = True) -> dict:
    home = row["home_team"]
    away = row["away_team"]
    payload = {
        "match_id": int(row["match_id"]),
        "kickoff_date": _iso_date(row["match_date"]),
        "kickoff_time": _iso_time(row["kickoff_time"]),
        "home_team": home,
        "away_team": away,
        "home_crest": team_badge(home),
        "away_crest": team_badge(away),
        "has_prediction": False,
    }
    if include_prediction and _has_prediction(row):
        predicted = int(row["predicted_class"])
        payload.update(
            {
                "has_prediction": True,
                "p_home": float(row["p_home"]),
                "p_draw": float(row["p_draw"]),
                "p_away": float(row["p_away"]),
                "p_home_pct": as_percent(row["p_home"]),
                "p_draw_pct": as_percent(row["p_draw"]),
                "p_away_pct": as_percent(row["p_away"]),
                "predicted_class": predicted,
                "predicted_outcome": predicted_outcome(predicted, home, away),
                "model_name": algorithm_label(row["algorithm"]),
                "model_algorithm": row["algorithm"],
                "feature_version": row["feature_version"],
                "predicted_at": _iso_stamp(row["predicted_at"]),
                "note": match_note(row["p_home"], row["p_draw"], row["p_away"], predicted),
            }
        )
    if include_result:
        code = None if row["result_code"] is None else int(row["result_code"])
        payload["home_goals"] = row["home_goals"]
        payload["away_goals"] = row["away_goals"]
        payload["result_code"] = code
        payload["actual_outcome"] = actual_outcome(code, home, away)
        payload["correct"] = (
            None
            if not payload.get("has_prediction") or code is None
            else code == int(row["predicted_class"])
        )
    return payload


def production_model() -> dict:
    """Dashboard view of the canonical production model. Same keys as before."""
    model = _registry_production_model()
    return {
        "model_run_id": model.model_run_id,
        "algorithm": model.algorithm,
        "model_name": algorithm_label(model.algorithm),
        "feature_version": model.feature_version,
        "trained_at": _iso_stamp(model.trained_at),
        "artifact_path": model.artifact_path,
    }


def list_teams() -> list[str]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT canonical_name
                FROM teams
                ORDER BY canonical_name
                """
            )
            return [row[0] for row in cur.fetchall()]


def latest_completed_match() -> dict | None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.match_date, m.kickoff_time, m.home_goals, m.away_goals,
                       home.canonical_name, away.canonical_name, m.result
                FROM matches AS m
                JOIN teams AS home ON home.id = m.home_team_id
                JOIN teams AS away ON away.id = m.away_team_id
                WHERE m.is_played
                ORDER BY m.match_date DESC, m.kickoff_time DESC NULLS LAST, m.id DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
    if not row:
        return None
    return {
        "kickoff_date": _iso_date(row[0]),
        "kickoff_time": _iso_time(row[1]),
        "home_goals": row[2],
        "away_goals": row[3],
        "home_team": row[4],
        "away_team": row[5],
        "home_crest": team_badge(row[4]),
        "away_crest": team_badge(row[5]),
        "result": row[6],
        "scoreline": f"{row[4]} {row[2]}–{row[3]} {row[5]}",
    }


def predictions_updated_at() -> str | None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(created_at) FROM predictions")
            row = cur.fetchone()
    return _iso_stamp(row[0]) if row and row[0] else None


def upcoming_fixtures(*, team: str | None = None, date_from: str | None = None, date_to: str | None = None) -> list[dict]:
    model = production_model()
    clauses = ["m.is_played = FALSE", "p.model_run_id = %s"]
    params: list = [model["model_run_id"]]
    if team:
        clauses.append("(home.canonical_name = %s OR away.canonical_name = %s)")
        params.extend([team, team])
    if date_from:
        clauses.append("m.match_date >= %s")
        params.append(date_from)
    if date_to:
        clauses.append("m.match_date <= %s")
        params.append(date_to)
    where = " AND ".join(clauses)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT m.id AS match_id, m.match_date, m.kickoff_time,
                       home.canonical_name AS home_team,
                       away.canonical_name AS away_team,
                       p.p_away, p.p_draw, p.p_home, p.predicted_class, p.created_at AS predicted_at,
                       r.algorithm, r.feature_version
                FROM matches AS m
                JOIN predictions AS p ON p.match_id = m.id
                JOIN model_runs AS r ON r.id = p.model_run_id
                JOIN teams AS home ON home.id = m.home_team_id
                JOIN teams AS away ON away.id = m.away_team_id
                WHERE {where}
                ORDER BY m.match_date, m.kickoff_time NULLS LAST, m.id
                """,
                params,
            )
            columns = [col.name for col in cur.description]
            rows = [dict(zip(columns, rec)) for rec in cur.fetchall()]
    return [_serialize_fixture(row, include_result=False) for row in rows]


def settled_live_fixtures(
    *,
    team: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """Played live-season matches that still have a frozen pre-match prediction."""
    algorithm = _selected_algorithm()
    clauses = ["m.is_played", "s.start_year = %s", "r.algorithm = %s"]
    params: list = [LIVE_START_YEAR, algorithm]
    if team:
        clauses.append("(home.canonical_name = %s OR away.canonical_name = %s)")
        params.extend([team, team])
    if date_from:
        clauses.append("m.match_date >= %s")
        params.append(date_from)
    if date_to:
        clauses.append("m.match_date <= %s")
        params.append(date_to)
    where = " AND ".join(clauses)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT ON (m.id)
                       m.id AS match_id, m.match_date, m.kickoff_time,
                       home.canonical_name AS home_team,
                       away.canonical_name AS away_team,
                       m.home_goals, m.away_goals, m.result_code,
                       p.p_away, p.p_draw, p.p_home, p.predicted_class, p.created_at AS predicted_at,
                       r.algorithm, r.feature_version
                FROM matches AS m
                JOIN seasons AS s ON s.id = m.season_id
                JOIN predictions AS p ON p.match_id = m.id
                JOIN model_runs AS r ON r.id = p.model_run_id
                JOIN teams AS home ON home.id = m.home_team_id
                JOIN teams AS away ON away.id = m.away_team_id
                WHERE {where}
                ORDER BY m.id, p.created_at DESC
                """,
                params,
            )
            columns = [col.name for col in cur.description]
            rows = [dict(zip(columns, rec)) for rec in cur.fetchall()]
    rows.sort(key=lambda item: (item["match_date"], item["kickoff_time"] or time(0, 0)), reverse=True)
    return [_serialize_fixture(row, include_result=True) for row in rows]


def live_scorecard(settled: list[dict]) -> dict:
    if not settled:
        return {"n": 0, "accuracy": None, "log_loss": None, "correct": 0, "by_actual": {}, "by_predicted": {}}
    y = np.array([row["result_code"] for row in settled], dtype=int)
    proba = np.array([[row["p_away"], row["p_draw"], row["p_home"]] for row in settled], dtype=float)
    pred = np.array([row["predicted_class"] for row in settled], dtype=int)
    clipped = np.clip(proba, 1e-15, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    log_loss = float(-np.mean(np.log(clipped[np.arange(len(y)), y])))
    correct = int((pred == y).sum())
    return {
        "n": int(len(settled)),
        "correct": correct,
        "accuracy": correct / len(settled),
        "log_loss": log_loss,
        "by_actual": _class_slice(settled, "result_code"),
        "by_predicted": _class_slice(settled, "predicted_class"),
    }


def _class_slice(settled: list[dict], key: str) -> dict:
    labels = {0: "away", 1: "draw", 2: "home"}
    out: dict[str, dict] = {}
    for code, name in labels.items():
        rows = [row for row in settled if int(row[key]) == code]
        n = len(rows)
        hits = sum(1 for row in rows if int(row["predicted_class"]) == int(row["result_code"]))
        out[name] = {
            "n": n,
            "correct": hits,
            "rate": (hits / n) if n else None,
        }
    return out


def overview_payload() -> dict:
    model = production_model()
    upcoming = upcoming_fixtures()
    settled = settled_live_fixtures()
    return {
        "model": model,
        "live_season": LIVE_SEASON,
        "predictions_updated_at": predictions_updated_at(),
        "latest_completed_match": latest_completed_match(),
        "live_scorecard": live_scorecard(settled),
        "n_upcoming": len(upcoming),
        "n_settled": len(settled),
        "next_upcoming": upcoming[:8],
        "teams": list_teams(),
    }


def upcoming_payload(
    *,
    team: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = 1,
    page_size: int = 16,
) -> dict:
    model = production_model()
    rows = upcoming_fixtures(team=team, date_from=date_from, date_to=date_to)
    page_data = paginate(rows, page=page, page_size=page_size)
    return {
        "model": model,
        "live_season": LIVE_SEASON,
        "teams": list_teams(),
        "page": page_data["page"],
        "page_size": page_data["page_size"],
        "total": page_data["total"],
        "pages": page_data["pages"],
        "groups": group_by_date(page_data["items"]),
    }


def results_payload(
    *,
    team: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = 1,
    page_size: int = 16,
) -> dict:
    model = production_model()
    rows = settled_live_fixtures(team=team, date_from=date_from, date_to=date_to)
    page_data = paginate(rows, page=page, page_size=page_size)
    return {
        "model": model,
        "live_season": LIVE_SEASON,
        "teams": list_teams(),
        "page": page_data["page"],
        "page_size": page_data["page_size"],
        "total": page_data["total"],
        "pages": page_data["pages"],
        "groups": group_by_date(page_data["items"]),
    }


def forecast_payload() -> dict:
    """Read the pipeline-generated forecast. Does not retrain or resimulate.

    Team rows are enriched with a display crest so the dashboard can render a
    club badge. That is presentation data only: no simulated value is touched,
    added, reordered, or recomputed here.
    """
    from football_pipeline.season_sim import load_dashboard_forecast

    model = production_model()
    report = load_dashboard_forecast()
    return {
        "model": model,
        "live_season": LIVE_SEASON,
        **report,
        "teams": with_team_crests(report.get("teams") or []),
    }


def with_team_crests(rows: list[dict]) -> list[dict]:
    """Attach a display badge to each forecast row, preserving order and values."""
    return [{**row, "crest": team_badge(row["team"])} for row in rows]


def performance_payload() -> dict:
    settled = settled_live_fixtures()
    score = live_scorecard(settled)
    return {
        "model": production_model(),
        "live_season": LIVE_SEASON,
        "live_scorecard": score,
        "n_settled": score["n"],
    }


def live_table_positions() -> dict[str, dict]:
    """Current live-season table from played matches. Display only; not used in training."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT home.canonical_name AS home_team,
                       away.canonical_name AS away_team,
                       m.home_goals, m.away_goals
                FROM matches AS m
                JOIN seasons AS s ON s.id = m.season_id
                JOIN teams AS home ON home.id = m.home_team_id
                JOIN teams AS away ON away.id = m.away_team_id
                WHERE m.is_played AND s.start_year = %s
                """,
                (LIVE_START_YEAR,),
            )
            played = cur.fetchall()
            cur.execute(
                """
                SELECT DISTINCT t.canonical_name
                FROM teams AS t
                JOIN matches AS m ON m.home_team_id = t.id OR m.away_team_id = t.id
                JOIN seasons AS s ON s.id = m.season_id
                WHERE s.start_year = %s
                """,
                (LIVE_START_YEAR,),
            )
            teams = [row[0] for row in cur.fetchall()]
    table = {
        name: {"played": 0, "points": 0, "gf": 0, "ga": 0, "gd": 0} for name in teams
    }
    for home, away, home_goals, away_goals in played:
        home_goals = int(home_goals)
        away_goals = int(away_goals)
        if home_goals > away_goals:
            home_pts, away_pts = 3, 0
        elif home_goals < away_goals:
            home_pts, away_pts = 0, 3
        else:
            home_pts, away_pts = 1, 1
        table[home]["played"] += 1
        table[away]["played"] += 1
        table[home]["points"] += home_pts
        table[away]["points"] += away_pts
        table[home]["gf"] += home_goals
        table[home]["ga"] += away_goals
        table[away]["gf"] += away_goals
        table[away]["ga"] += home_goals
        table[home]["gd"] = table[home]["gf"] - table[home]["ga"]
        table[away]["gd"] = table[away]["gf"] - table[away]["ga"]
    ranked = sorted(
        table.items(),
        key=lambda item: (-item[1]["points"], -item[1]["gd"], -item[1]["gf"], item[0]),
    )
    positions: dict[str, dict] = {}
    for index, (name, stats) in enumerate(ranked, start=1):
        positions[name] = {
            **stats,
            "position": index,
            "position_label": ordinal(index),
        }
    return positions


def _serialize_h2h(row: dict) -> dict:
    home = row["home_team"]
    away = row["away_team"]
    code = None if row["result_code"] is None else int(row["result_code"])
    match_id = int(row["match_id"])
    return {
        "match_id": match_id,
        "kickoff_date": _iso_date(row["match_date"]),
        "kickoff_time": _iso_time(row["kickoff_time"]),
        "home_team": home,
        "away_team": away,
        "home_crest": team_badge(home),
        "away_crest": team_badge(away),
        "home_goals": row["home_goals"],
        "away_goals": row["away_goals"],
        "result_code": code,
        "result_label": result_side_label(code),
        "detail_path": f"/results/{match_id}",
    }


def prior_head_to_head(
    *,
    home_team_id: int,
    away_team_id: int,
    home_team: str,
    away_team: str,
    fixture_date: str,
    fixture_match_id: int,
    limit: int = 5,
) -> list[dict]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.id AS match_id, m.match_date, m.kickoff_time, m.is_played,
                       home.canonical_name AS home_team,
                       away.canonical_name AS away_team,
                       m.home_goals, m.away_goals, m.result_code
                FROM matches AS m
                JOIN teams AS home ON home.id = m.home_team_id
                JOIN teams AS away ON away.id = m.away_team_id
                WHERE m.is_played
                  AND m.id <> %s
                  AND m.match_date < %s
                  AND (
                        (m.home_team_id = %s AND m.away_team_id = %s)
                     OR (m.home_team_id = %s AND m.away_team_id = %s)
                  )
                ORDER BY m.match_date DESC, m.kickoff_time DESC NULLS LAST, m.id DESC
                LIMIT %s
                """,
                (
                    fixture_match_id,
                    fixture_date,
                    home_team_id,
                    away_team_id,
                    away_team_id,
                    home_team_id,
                    limit,
                ),
            )
            columns = [col.name for col in cur.description]
            rows = [dict(zip(columns, rec)) for rec in cur.fetchall()]
    filtered = filter_prior_h2h(
        rows,
        fixture_date=fixture_date,
        fixture_match_id=fixture_match_id,
        home_team=home_team,
        away_team=away_team,
        limit=limit,
    )
    return [_serialize_h2h(row) for row in filtered]


def _apply_table_ranks(fixture: dict) -> None:
    positions = live_table_positions()
    home_table = positions.get(fixture["home_team"])
    away_table = positions.get(fixture["away_team"])
    fixture["home_position"] = None if home_table is None else home_table["position"]
    fixture["home_position_label"] = None if home_table is None else home_table["position_label"]
    fixture["away_position"] = None if away_table is None else away_table["position"]
    fixture["away_position_label"] = None if away_table is None else away_table["position_label"]


def match_detail_payload(match_id: int) -> dict | None:
    """Upcoming or completed match detail. Reads stored rows only; does not score or retrain."""
    model = production_model()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.id AS match_id, m.match_date, m.kickoff_time, m.is_played,
                       m.home_team_id, m.away_team_id,
                       m.home_goals, m.away_goals, m.result_code,
                       s.start_year,
                       home.canonical_name AS home_team,
                       away.canonical_name AS away_team,
                       c.name AS competition_name,
                       pred.p_away, pred.p_draw, pred.p_home, pred.predicted_class, pred.predicted_at,
                       pred.algorithm, pred.feature_version,
                       raw.payload
                FROM matches AS m
                JOIN teams AS home ON home.id = m.home_team_id
                JOIN teams AS away ON away.id = m.away_team_id
                JOIN competitions AS c ON c.id = m.competition_id
                JOIN seasons AS s ON s.id = m.season_id
                LEFT JOIN LATERAL (
                    SELECT p.p_away, p.p_draw, p.p_home, p.predicted_class,
                           p.created_at AS predicted_at, r.algorithm, r.feature_version
                    FROM predictions AS p
                    JOIN model_runs AS r ON r.id = p.model_run_id
                    WHERE p.match_id = m.id AND r.algorithm = %s
                    ORDER BY p.created_at DESC
                    LIMIT 1
                ) AS pred ON TRUE
                LEFT JOIN LATERAL (
                    SELECT rp.payload
                    FROM raw_match_payloads AS rp
                    WHERE rp.season_id = m.season_id
                      AND rp.payload->>'HomeTeam' = home.source_name
                      AND rp.payload->>'AwayTeam' = away.source_name
                      AND rp.payload->>'FTHG' IS NOT NULL
                      AND rp.payload->>'FTHG' <> ''
                    ORDER BY CASE WHEN rp.source_file = m.source_file THEN 0 ELSE 1 END, rp.id
                    LIMIT 1
                ) AS raw ON TRUE
                WHERE m.id = %s
                """,
                (model["algorithm"], match_id),
            )
            columns = [col.name for col in cur.description]
            rec = cur.fetchone()
    if not rec:
        return None
    row = dict(zip(columns, rec))
    played = bool(row["is_played"])
    if not played and not _has_prediction(row):
        return None
    live_settled = played and int(row["start_year"]) == LIVE_START_YEAR
    fixture = _serialize_fixture(
        row,
        include_result=played,
        include_prediction=(not played) or live_settled,
    )
    fixture["competition"] = competition_label(row.get("competition_name"))
    fixture["status"] = "Full-time" if played else "Upcoming"
    _apply_table_ranks(fixture)
    stats = extract_match_stats(row.get("payload")) if played else []
    payload = {
        "kind": "result" if played else "upcoming",
        "model": model,
        "live_season": LIVE_SEASON,
        "match": fixture,
        "timeline_available": False,
        "timeline_note": "Goal timeline not available from the current data source.",
    }
    if played:
        payload["stats"] = stats
        payload["stats_available"] = bool(stats)
        return payload
    meetings = prior_head_to_head(
        home_team_id=int(row["home_team_id"]),
        away_team_id=int(row["away_team_id"]),
        home_team=fixture["home_team"],
        away_team=fixture["away_team"],
        fixture_date=fixture["kickoff_date"],
        fixture_match_id=int(row["match_id"]),
    )
    payload["head_to_head"] = meetings
    payload["h2h_limit"] = 5
    payload["h2h_count"] = len(meetings)
    return payload


def about_payload() -> dict:
    report = {}
    if REPORT_PATH.is_file():
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    test_path = ROOT / "data" / "processed" / "final_test_metrics.json"
    test_report = json.loads(test_path.read_text(encoding="utf-8")) if test_path.is_file() else {}
    test = test_report.get("test") or report.get("test_holdout") or {}
    walk = (report.get("walkforward") or {}).get("summary") or {}
    selected = report.get("selected_by_walkforward_log_loss") or "logistic_regression"
    selected_stats = (walk.get(selected) or {}).get("mean") or {}
    model = production_model()
    holdout_year = seasons.holdout_season_start_year()
    holdout_season = seasons.short_season_name(holdout_year)
    last_train_season = seasons.short_season_name(holdout_year - 1)
    fold_years = seasons.walkforward_valid_years()
    fold_span = (
        f"{seasons.short_season_name(fold_years[0])}–"
        f"{seasons.short_season_name(fold_years[-1])}"
    )
    ingest_season = seasons.short_season_name(INGEST_START_YEAR)
    return {
        "model": model,
        "selection_reason": report.get("selection_reason"),
        "walkforward_folds": (report.get("walkforward") or {}).get("folds") or [],
        "walkforward_mean_log_loss": selected_stats.get("log_loss"),
        "walkforward_mean_accuracy": selected_stats.get("accuracy"),
        "feature_version": report.get("feature_version") or model["feature_version"],
        "features": [
            {
                "title": "Recent form",
                "detail": "Last-five win rate, points, goal difference, and goals scored/conceded for each side.",
            },
            {
                "title": "Home and away splits",
                "detail": "Home-team home form and away-team away form over the previous five matching fixtures.",
            },
            {
                "title": "Head-to-head",
                "detail": "Prior meetings between the same two clubs, including historical draw rate.",
            },
            {
                "title": "Elo ratings",
                "detail": "Pre-match Elo for each club plus the absolute rating gap.",
            },
            {
                "title": "Draw-aware rates",
                "detail": "Last-five draw rates (overall and venue-specific) and closeness proxies such as PPG and goal differentials.",
            },
        ],
        "leakage_note": (
            "Every feature is computed from matches that finished before the fixture. "
            "The current match never enters its own form, Elo, or head-to-head window."
        ),
        "history": {
            "ingest": f"Premier League results from {ingest_season} through the current season",
            "walkforward": (
                f"Expanding-window selection on {fold_span}, training through the "
                "previous season each time"
            ),
            "production_train": (
                f"Retrain the winner on permitted history through {last_train_season}"
            ),
            "test": f"Untouched {holdout_season} holdout, evaluated once after selection",
            "live": (
                f"{LIVE_SEASON} is scored live only and is never used for selection "
                "or test metrics"
            ),
        },
        "test_season": test_report.get("test_season") or holdout_season,
        "test": {
            "n": test.get("n"),
            "accuracy": test.get("accuracy"),
            "log_loss": test.get("log_loss"),
            "f1_macro": test.get("f1_macro"),
            "precision_home": test.get("precision_home"),
            "recall_home": test.get("recall_home"),
            "f1_home": test.get("f1_home"),
            "precision_draw": test.get("precision_draw"),
            "recall_draw": test.get("recall_draw"),
            "f1_draw": test.get("f1_draw"),
            "precision_away": test.get("precision_away"),
            "recall_away": test.get("recall_away"),
            "f1_away": test.get("f1_away"),
            "confusion_matrix": test.get("confusion_matrix"),
            "draw_proba_mean": ((test.get("metrics") or {}).get("draw_proba") or {}).get("mean"),
            "pct_argmax_draw": ((test.get("metrics") or {}).get("draw_proba") or {}).get("pct_argmax_draw"),
        },
        "draw_limitation": (
            "The selected unweighted logistic regression assigns meaningful draw probability "
            f"(around 22% on {holdout_season}) but almost never has Draw as the single most "
            "likely class. "
            "Argmax therefore rarely, if ever, predicts a draw. That is a known limitation of the "
            "production model, not a UI rounding choice. Predicted labels on this dashboard are "
            "exactly the stored argmax."
        ),
    }
