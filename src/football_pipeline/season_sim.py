"""Monte Carlo Premier League season forecast from frozen 1X2 probabilities.

Read-only: never trains a model and never writes to ``predictions``.
Remaining fixtures are sampled Home/Draw/Away; scorelines are not simulated.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from football_pipeline import seasons
from football_pipeline.config import ROOT
from football_pipeline.db import connect
from football_pipeline.registry import ProductionModelUnavailable, production_model

logger = logging.getLogger(__name__)


class ForecastUnavailable(RuntimeError):
    """Dashboard should not display a missing or stale forecast artifact."""

DEFAULT_N_SIMS = 20_000
DEFAULT_SEED = 202627
# Derived, not pinned: the live season rolls over on 1 August.
LIVE_START_YEAR = seasons.live_season_start_year()
ARTIFACT_PATH = ROOT / "data" / "processed" / "season_forecast.json"

OUTCOME_AWAY = 0
OUTCOME_DRAW = 1
OUTCOME_HOME = 2

TIEBREAK_NOTE = (
    "The production model predicts Home/Draw/Away, not exact scores, so remaining "
    "matches do not update goal difference. After simulated points, ties are broken "
    "by current goal difference from completed matches, then current goals scored, "
    "then club name. That is a v1 limitation, not a real Premier League tie-break."
)

EARLY_SEASON_NOTE = (
    "Early-season title probabilities can be less reliable. The current model does "
    "not directly account for transfers, new managers, squad turnover, injuries, or "
    "other major team changes. The forecast updates naturally as new results change "
    "pre-match form and Elo."
)

TOP_FOUR = 4
RELEGATION_PLACES = 3
CONTENDER_TITLE_SHARE = 0.05


@dataclass(frozen=True)
class PlayedMatch:
    home: str
    away: str
    home_goals: int
    away_goals: int


@dataclass(frozen=True)
class RemainingFixture:
    match_id: int
    home: str
    away: str
    p_away: float
    p_draw: float
    p_home: float


@dataclass
class TeamRecord:
    name: str
    points: int = 0
    played: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga


@dataclass
class SeasonSnapshot:
    season: str
    start_year: int
    teams: list[str]
    played: list[PlayedMatch]
    remaining: list[RemainingFixture]
    missing_remaining: int
    model_run_id: int
    algorithm: str
    feature_version: str
    predictions_updated_at: str | None
    table: dict[str, TeamRecord] = field(default_factory=dict)


def match_points(home_goals: int, away_goals: int) -> tuple[int, int]:
    """Return (home_points, away_points) for a completed scoreline."""
    if int(home_goals) > int(away_goals):
        return 3, 0
    if int(home_goals) < int(away_goals):
        return 0, 3
    return 1, 1


def apply_played_match(table: dict[str, TeamRecord], match: PlayedMatch) -> None:
    home = table.setdefault(match.home, TeamRecord(name=match.home))
    away = table.setdefault(match.away, TeamRecord(name=match.away))
    home_pts, away_pts = match_points(match.home_goals, match.away_goals)
    home.points += home_pts
    away.points += away_pts
    home.gf += int(match.home_goals)
    home.ga += int(match.away_goals)
    away.gf += int(match.away_goals)
    away.ga += int(match.home_goals)
    home.played += 1
    away.played += 1


def build_table(played: list[PlayedMatch], teams: list[str]) -> dict[str, TeamRecord]:
    table = {name: TeamRecord(name=name) for name in teams}
    for match in played:
        apply_played_match(table, match)
    return table


def normalize_probs(p_away: float, p_draw: float, p_home: float) -> np.ndarray:
    probs = np.array([p_away, p_draw, p_home], dtype=float)
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("Match probabilities must sum to a positive value.")
    return probs / total


def sample_1x2(p_away: float, p_draw: float, p_home: float, rng: np.random.Generator) -> int:
    """Sample 0=Away, 1=Draw, 2=Home from stored probabilities."""
    return int(rng.choice(3, p=normalize_probs(p_away, p_draw, p_home)))


def sample_outcomes(
    probs: np.ndarray,
    rng: np.random.Generator,
    n_sims: int,
) -> np.ndarray:
    """Vectorized 1X2 samples. ``probs`` is (n_fixtures, 3) as away/draw/home."""
    if probs.size == 0:
        return np.zeros((n_sims, 0), dtype=np.int8)
    totals = probs.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Match probabilities must sum to a positive value.")
    p = probs / totals
    away = p[:, 0]
    draw_end = p[:, 0] + p[:, 1]
    u = rng.random((n_sims, probs.shape[0]))
    return np.where(u < away, OUTCOME_AWAY, np.where(u < draw_end, OUTCOME_DRAW, OUTCOME_HOME)).astype(
        np.int8
    )


def rank_indices(points: np.ndarray, gd: np.ndarray, gf: np.ndarray, names: list[str]) -> np.ndarray:
    """Return team indices in finishing order (position 1 first)."""
    order = list(range(len(names)))
    order.sort(key=lambda i: (-int(points[i]), -int(gd[i]), -int(gf[i]), names[i]))
    return np.asarray(order, dtype=int)


def simulate_season(
    *,
    teams: list[str],
    table: dict[str, TeamRecord],
    remaining: list[RemainingFixture],
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run ``n_sims`` remaining-season paths. Does not mutate ``remaining`` or ``table``."""
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1")
    names = list(teams)
    n_teams = len(names)
    if n_teams == 0:
        raise ValueError("No teams to simulate")
    index = {name: i for i, name in enumerate(names)}
    current_points = np.array([table[name].points for name in names], dtype=np.int32)
    current_gd = np.array([table[name].gd for name in names], dtype=np.int32)
    current_gf = np.array([table[name].gf for name in names], dtype=np.int32)

    fixture_probs = []
    home_idx = []
    away_idx = []
    for fixture in remaining:
        home_idx.append(index[fixture.home])
        away_idx.append(index[fixture.away])
        fixture_probs.append(normalize_probs(fixture.p_away, fixture.p_draw, fixture.p_home))
    probs = np.asarray(fixture_probs, dtype=float) if fixture_probs else np.zeros((0, 3))
    home_idx_arr = np.asarray(home_idx, dtype=int)
    away_idx_arr = np.asarray(away_idx, dtype=int)

    rng = np.random.default_rng(int(seed))
    outcomes = sample_outcomes(probs, rng, n_sims)
    sim_points = np.repeat(current_points[None, :], n_sims, axis=0)
    if outcomes.shape[1]:
        home_pts = np.where(outcomes == OUTCOME_HOME, 3, np.where(outcomes == OUTCOME_DRAW, 1, 0))
        away_pts = np.where(outcomes == OUTCOME_AWAY, 3, np.where(outcomes == OUTCOME_DRAW, 1, 0))
        for fixture_i in range(outcomes.shape[1]):
            sim_points[:, home_idx_arr[fixture_i]] += home_pts[:, fixture_i]
            sim_points[:, away_idx_arr[fixture_i]] += away_pts[:, fixture_i]

    title = np.zeros(n_teams, dtype=np.int32)
    top4 = np.zeros(n_teams, dtype=np.int32)
    relegated = np.zeros(n_teams, dtype=np.int32)
    pos_sum = np.zeros(n_teams, dtype=np.float64)
    cutoff = max(n_teams - RELEGATION_PLACES, 0)
    for sim_i in range(n_sims):
        order = rank_indices(sim_points[sim_i], current_gd, current_gf, names)
        for position, team_i in enumerate(order, start=1):
            pos_sum[team_i] += position
            if position == 1:
                title[team_i] += 1
            if position <= TOP_FOUR:
                top4[team_i] += 1
            if position > cutoff:
                relegated[team_i] += 1

    n = float(n_sims)
    current_order = rank_indices(current_points, current_gd, current_gf, names)
    current_position = {int(idx): rank for rank, idx in enumerate(current_order, start=1)}
    rows = []
    for i, name in enumerate(names):
        title_prob = float(title[i] / n)
        rows.append(
            {
                "team": name,
                "current_points": int(current_points[i]),
                "current_played": int(table[name].played),
                "current_gd": int(current_gd[i]),
                "current_gf": int(current_gf[i]),
                "current_ga": int(table[name].ga),
                "current_position": current_position[i],
                "title_prob": title_prob,
                "top4_prob": float(top4[i] / n),
                "relegation_prob": float(relegated[i] / n),
                "expected_points": float(sim_points[:, i].mean()),
                "expected_position": float(pos_sum[i] / n),
                "contender": title_prob >= CONTENDER_TITLE_SHARE,
            }
        )
    rows.sort(key=lambda row: (-row["title_prob"], row["expected_position"], row["team"]))
    return {
        "n_sims": int(n_sims),
        "seed": int(seed),
        "n_teams": n_teams,
        "teams": rows,
        "title_prob_sum": float(sum(row["title_prob"] for row in rows)),
        "top4_prob_sum": float(sum(row["top4_prob"] for row in rows)),
        "relegation_prob_sum": float(sum(row["relegation_prob"] for row in rows)),
        "tiebreak": TIEBREAK_NOTE,
    }


def load_live_season(*, start_year: int = LIVE_START_YEAR) -> SeasonSnapshot:
    """SELECT-only snapshot of completed results and frozen remaining probabilities."""
    model = production_model()
    model_run_id = model.model_run_id
    algorithm = model.algorithm
    feature_version = model.feature_version
    season_name = seasons.short_season_name(start_year)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT home.canonical_name, away.canonical_name,
                       m.home_goals, m.away_goals
                FROM matches AS m
                JOIN seasons AS s ON s.id = m.season_id
                JOIN teams AS home ON home.id = m.home_team_id
                JOIN teams AS away ON away.id = m.away_team_id
                WHERE s.start_year = %s AND m.is_played
                ORDER BY m.match_date, m.id
                """,
                (start_year,),
            )
            played = [
                PlayedMatch(home=row[0], away=row[1], home_goals=int(row[2]), away_goals=int(row[3]))
                for row in cur.fetchall()
                if row[2] is not None and row[3] is not None
            ]

            cur.execute(
                """
                SELECT DISTINCT canonical_name
                FROM (
                    SELECT home.canonical_name
                    FROM matches AS m
                    JOIN seasons AS s ON s.id = m.season_id
                    JOIN teams AS home ON home.id = m.home_team_id
                    WHERE s.start_year = %s
                    UNION
                    SELECT away.canonical_name
                    FROM matches AS m
                    JOIN seasons AS s ON s.id = m.season_id
                    JOIN teams AS away ON away.id = m.away_team_id
                    WHERE s.start_year = %s
                ) AS live_teams
                ORDER BY 1
                """,
                (start_year, start_year),
            )
            teams = [row[0] for row in cur.fetchall()]

            cur.execute(
                """
                SELECT m.id, home.canonical_name, away.canonical_name,
                       p.p_away, p.p_draw, p.p_home, p.created_at
                FROM matches AS m
                JOIN seasons AS s ON s.id = m.season_id
                JOIN teams AS home ON home.id = m.home_team_id
                JOIN teams AS away ON away.id = m.away_team_id
                JOIN predictions AS p ON p.match_id = m.id AND p.model_run_id = %s
                WHERE s.start_year = %s AND m.is_played = FALSE
                ORDER BY m.match_date, m.id
                """,
                (model_run_id, start_year),
            )
            remaining_rows = cur.fetchall()
            remaining = [
                RemainingFixture(
                    match_id=int(row[0]),
                    home=row[1],
                    away=row[2],
                    p_away=float(row[3]),
                    p_draw=float(row[4]),
                    p_home=float(row[5]),
                )
                for row in remaining_rows
            ]
            pred_times = [row[6] for row in remaining_rows if row[6] is not None]
            predictions_updated_at = max(pred_times).isoformat() if pred_times else None

            cur.execute(
                """
                SELECT COUNT(*)
                FROM matches AS m
                JOIN seasons AS s ON s.id = m.season_id
                WHERE s.start_year = %s AND m.is_played = FALSE
                  AND NOT EXISTS (
                      SELECT 1 FROM predictions AS p
                      WHERE p.match_id = m.id AND p.model_run_id = %s
                  )
                """,
                (start_year, model_run_id),
            )
            missing_remaining = int(cur.fetchone()[0])

    table = build_table(played, teams)
    return SeasonSnapshot(
        season=season_name,
        start_year=start_year,
        teams=teams,
        played=played,
        remaining=remaining,
        missing_remaining=missing_remaining,
        model_run_id=model_run_id,
        algorithm=algorithm,
        feature_version=feature_version,
        predictions_updated_at=predictions_updated_at,
        table=table,
    )


def write_artifact(payload: dict, path: Path = ARTIFACT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


_CACHE: dict[tuple, dict] = {}


def run_live_forecast(
    *,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
    start_year: int = LIVE_START_YEAR,
    write: bool = True,
) -> dict:
    snapshot = load_live_season(start_year=start_year)
    cache_key = (
        snapshot.model_run_id,
        len(snapshot.played),
        len(snapshot.remaining),
        snapshot.predictions_updated_at,
        int(n_sims),
        int(seed),
    )
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached
    sim = simulate_season(
        teams=snapshot.teams,
        table=snapshot.table,
        remaining=snapshot.remaining,
        n_sims=n_sims,
        seed=seed,
    )
    payload = {
        "live_season": snapshot.season,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions_updated_at": snapshot.predictions_updated_at,
        "n_completed": len(snapshot.played),
        "n_remaining": len(snapshot.remaining),
        "n_remaining_without_prediction": snapshot.missing_remaining,
        "model_run_id": snapshot.model_run_id,
        "algorithm": snapshot.algorithm,
        "feature_version": snapshot.feature_version,
        "early_season_note": EARLY_SEASON_NOTE,
        **sim,
    }
    if write:
        write_artifact(payload)
        logger.info("Wrote %s (%s sims, seed=%s)", ARTIFACT_PATH, n_sims, seed)
    _CACHE[cache_key] = payload
    return payload


def forecast_matches_snapshot(report: dict, snapshot: SeasonSnapshot) -> bool:
    """True when the artifact was built from this live-season database state."""
    try:
        return (
            int(report.get("model_run_id")) == int(snapshot.model_run_id)
            and int(report.get("n_completed")) == len(snapshot.played)
            and int(report.get("n_remaining")) == len(snapshot.remaining)
            and str(report.get("predictions_updated_at") or "")
            == str(snapshot.predictions_updated_at or "")
        )
    except (TypeError, ValueError):
        return False


def load_dashboard_forecast() -> dict:
    """Read the pipeline artifact. Never retrains and never writes predictions."""
    if not ARTIFACT_PATH.is_file():
        raise ForecastUnavailable(
            "Season forecast has not been generated yet. Trigger football_match_pipeline in Airflow."
        )
    report = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    try:
        snapshot = load_live_season()
    except ProductionModelUnavailable as exc:
        raise ForecastUnavailable(
            "Season forecast is unavailable because no trained model is registered. "
            "Trigger football_match_pipeline in Airflow."
        ) from exc
    if not forecast_matches_snapshot(report, snapshot):
        raise ForecastUnavailable(
            "Season forecast is out of date because the last pipeline run did not "
            "finish the forecast step. Trigger football_match_pipeline again."
        )
    report.setdefault("early_season_note", EARLY_SEASON_NOTE)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Simulate the remainder of the live Premier League season from frozen probabilities."
    )
    parser.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    payload = run_live_forecast(n_sims=args.n_sims, seed=args.seed)
    top = payload["teams"][:5]
    print(
        json.dumps(
            {
                "n_sims": payload["n_sims"],
                "seed": payload["seed"],
                "n_remaining": payload["n_remaining"],
                "top_title": [
                    {"team": row["team"], "title_prob": round(row["title_prob"], 4)} for row in top
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
