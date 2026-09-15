"""Historical Season Forecast backtest. Experiment only.

Does not write predictions, model_runs, joblib artifacts, or
data/processed/season_forecast.json. Does not change the production model.

At each checkpoint in a completed season S:
- the league table uses only S matches on or before the cutoff date
- remaining-fixture features are built from matches on or before that date
- the classifier is unweighted logistic regression trained on completed
  seasons before S, never on S itself and never on the live production artifact
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import date, time

import numpy as np
import pandas as pd

from football_pipeline.config import MIN_PRIOR_N, ROOT
from football_pipeline.constants import FEATURE_VERSION
from football_pipeline.dataset import FEATURE_COLUMNS
from football_pipeline.db import connect
from football_pipeline.models import feature_matrix, logreg_pipeline, predict_proba_3way, target_vector
from football_pipeline.season_sim import (
    DEFAULT_N_SIMS,
    DEFAULT_SEED,
    PlayedMatch,
    RemainingFixture,
    build_table,
    rank_indices,
    simulate_season,
)
from football_pipeline.seasons import live_season_start_year, short_season_name

logger = logging.getLogger(__name__)

EXPERIMENT_DIR = ROOT / "data" / "processed" / "experiments"
REPORT_PATH = EXPERIMENT_DIR / "forecast_backtest.json"
PRODUCTION_FORECAST = ROOT / "data" / "processed" / "season_forecast.json"
PRODUCTION_METRICS = ROOT / "data" / "processed" / "model_metrics.json"

# 20 clubs, home and away. N matches per team => N * 10 league matches.
CLUBS_PER_SEASON = 20
CHECKPOINTS = (5, 10, 15, 20, 25, 30)
ALGORITHM = "logistic_regression"

# Must match spark_features.elo_before_matches.
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 100.0


class BacktestError(Exception):
    """The historical snapshot cannot be built without leaking or inventing data."""


def target_league_matches(n_per_team: int, n_clubs: int = CLUBS_PER_SEASON) -> int:
    """League matches after which each club has *on average* n_per_team games."""
    if n_per_team < 1 or n_clubs < 2 or n_clubs % 2:
        raise BacktestError(f"Invalid checkpoint {n_per_team} for {n_clubs} clubs")
    return n_per_team * n_clubs // 2


def _sort_key(row: dict) -> tuple:
    kickoff = row.get("kickoff_time")
    kickoff_s = kickoff.strftime("%H:%M:%S") if isinstance(kickoff, time) else str(kickoff or "")
    return (row["match_date"], kickoff_s, int(row["match_id"]))


def cutoff_for_checkpoint(
    played: list[dict], n_per_team: int, *, n_clubs: int = CLUBS_PER_SEASON
) -> dict:
    """First matchday on which the season has reached N * clubs/2 played matches.

    Games in hand mean some clubs may have N-1 or N+1. Every match on the
    cutoff date is included so a Saturday is never split. Pure: no database.
    """
    target = target_league_matches(n_per_team, n_clubs)
    ordered = sorted(played, key=_sort_key)
    if len(ordered) < target:
        raise BacktestError(
            f"Season has {len(ordered)} played matches; need {target} for {n_per_team} per team"
        )
    cutoff_date = ordered[target - 1]["match_date"]
    if isinstance(cutoff_date, str):
        cutoff_date = date.fromisoformat(cutoff_date)
    included = [row for row in ordered if row["match_date"] <= cutoff_date]
    by_team: dict[int, int] = defaultdict(int)
    for row in included:
        by_team[int(row["home_team_id"])] += 1
        by_team[int(row["away_team_id"])] += 1
    counts = list(by_team.values())
    return {
        "n_per_team": n_per_team,
        "target_matches": target,
        "cutoff_date": cutoff_date,
        "n_played": len(included),
        "played_min": int(min(counts)) if counts else 0,
        "played_median": float(np.median(counts)) if counts else 0.0,
        "played_max": int(max(counts)) if counts else 0,
    }


def elo_before_matches(rows: list[dict]) -> dict[int, tuple[float, float]]:
    """Pre-kickoff Elo. Unplayed rows do not update ratings. Same rule as Spark."""
    ratings: dict[int, float] = {}
    before: dict[int, tuple[float, float]] = {}
    for row in sorted(rows, key=_sort_key):
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        home_elo = ratings.get(home_id, ELO_START)
        away_elo = ratings.get(away_id, ELO_START)
        before[int(row["match_id"])] = (home_elo, away_elo)
        if not row.get("is_played"):
            continue
        expected_home = 1.0 / (
            1.0 + 10.0 ** ((away_elo - home_elo - ELO_HOME_ADVANTAGE) / 400.0)
        )
        hg, ag = row["home_goals"], row["away_goals"]
        if hg > ag:
            actual_home = 1.0
        elif hg < ag:
            actual_home = 0.0
        else:
            actual_home = 0.5
        ratings[home_id] = home_elo + ELO_K * (actual_home - expected_home)
        ratings[away_id] = away_elo + ELO_K * ((1.0 - actual_home) - (1.0 - expected_home))
    return before


def _team_played_log(played: list[dict], team_id: int) -> list[dict]:
    log: list[dict] = []
    for row in sorted(played, key=_sort_key):
        if int(row["home_team_id"]) == team_id:
            gf, ga, is_home = int(row["home_goals"]), int(row["away_goals"]), True
        elif int(row["away_team_id"]) == team_id:
            gf, ga, is_home = int(row["away_goals"]), int(row["home_goals"]), False
        else:
            continue
        log.append(
            {
                "is_home": is_home,
                "win": 1.0 if gf > ga else 0.0,
                "draw": 1.0 if gf == ga else 0.0,
                "gf": float(gf),
                "ga": float(ga),
                "points": 3.0 if gf > ga else (1.0 if gf == ga else 0.0),
                "gd": float(gf - ga),
            }
        )
    return log


def _last5(log: list[dict], *, venue: bool | None = None) -> dict[str, float | None]:
    rows = log if venue is None else [row for row in log if row["is_home"] is venue]
    last = rows[-5:]
    n = len(last)
    if n == 0:
        return {
            "win_rate": None,
            "draw_rate": None,
            "gf": None,
            "ga": None,
            "gt": None,
            "points": None,
            "gd": None,
            "prior_n": 0,
        }
    return {
        "win_rate": sum(row["win"] for row in last) / n,
        "draw_rate": sum(row["draw"] for row in last) / n,
        "gf": sum(row["gf"] for row in last) / n,
        "ga": sum(row["ga"] for row in last) / n,
        "gt": sum(row["gf"] + row["ga"] for row in last) / n,
        "points": sum(row["points"] for row in last),
        "gd": sum(row["gd"] for row in last),
        "prior_n": n,
    }


def _h2h(played: list[dict], home_id: int, away_id: int) -> tuple[float | None, float | None]:
    wins = []
    draws = []
    pair = {int(home_id), int(away_id)}
    for row in played:
        if {int(row["home_team_id"]), int(row["away_team_id"])} != pair:
            continue
        draws.append(1.0 if row["result"] == "D" else 0.0)
        home_won = (int(row["home_team_id"]) == int(home_id) and row["result"] == "H") or (
            int(row["away_team_id"]) == int(home_id) and row["result"] == "A"
        )
        wins.append(1.0 if home_won else 0.0)
    if not wins:
        return None, None
    return float(np.mean(wins)), float(np.mean(draws))


def remaining_feature_frame(played: list[dict], remaining: list[dict]) -> pd.DataFrame:
    """Features for remaining fixtures using only `played` history.

    Every remaining row is treated as unplayed, so Elo and form are frozen at
    the checkpoint. Later results in the same season cannot leak in.
    """
    if not remaining:
        return pd.DataFrame()
    snapshot_rows = [{**row, "is_played": True} for row in played] + [
        {**row, "is_played": False, "home_goals": None, "away_goals": None, "result": None}
        for row in remaining
    ]
    elo = elo_before_matches(snapshot_rows)
    form_cache: dict[int, list[dict]] = {}

    def form(team_id: int) -> list[dict]:
        if team_id not in form_cache:
            form_cache[team_id] = _team_played_log(played, team_id)
        return form_cache[team_id]

    records = []
    for row in remaining:
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        home = _last5(form(home_id))
        away = _last5(form(away_id))
        home_venue = _last5(form(home_id), venue=True)
        away_venue = _last5(form(away_id), venue=False)
        h2h_win, h2h_draw = _h2h(played, home_id, away_id)
        home_elo, away_elo = elo[int(row["match_id"])]
        home_prior = max(int(home["prior_n"]), 1)
        away_prior = max(int(away["prior_n"]), 1)
        records.append(
            {
                "match_id": int(row["match_id"]),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_win_rate_l5": home["win_rate"],
                "away_win_rate_l5": away["win_rate"],
                "home_goals_scored_avg_l5": home["gf"],
                "away_goals_scored_avg_l5": away["gf"],
                "home_goals_conceded_avg_l5": home["ga"],
                "away_goals_conceded_avg_l5": away["ga"],
                "home_points_l5": home["points"],
                "away_points_l5": away["points"],
                "home_gd_l5": home["gd"],
                "away_gd_l5": away["gd"],
                "home_home_win_rate_l5": home_venue["win_rate"],
                "away_away_win_rate_l5": away_venue["win_rate"],
                "h2h_home_win_rate_n": h2h_win,
                "home_elo": home_elo,
                "away_elo": away_elo,
                "home_draw_rate_l5": home["draw_rate"],
                "away_draw_rate_l5": away["draw_rate"],
                "home_home_draw_rate_l5": home_venue["draw_rate"],
                "away_away_draw_rate_l5": away_venue["draw_rate"],
                "h2h_draw_rate_n": h2h_draw,
                "elo_abs_diff": abs(home_elo - away_elo),
                "ppg_diff_l5": (home["points"] or 0.0) / home_prior - (away["points"] or 0.0) / away_prior,
                "gf_diff_l5": (home["gf"] or 0.0) - (away["gf"] or 0.0),
                "ga_diff_l5": (home["ga"] or 0.0) - (away["ga"] or 0.0),
                "recent_total_goals_avg_l5": (
                    None
                    if home["gt"] is None or away["gt"] is None
                    else (home["gt"] + away["gt"]) / 2.0
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def actual_champion(played: list[dict], teams: list[str]) -> str:
    matches = [
        PlayedMatch(row["home_team"], row["away_team"], int(row["home_goals"]), int(row["away_goals"]))
        for row in played
    ]
    table = build_table(matches, teams)
    names = list(teams)
    points = np.array([table[name].points for name in names])
    gd = np.array([table[name].gd for name in names])
    gf = np.array([table[name].gf for name in names])
    order = rank_indices(points, gd, gf, names)
    return names[int(order[0])]


def load_history() -> pd.DataFrame:
    """Read-only. Never writes matches, features, or predictions."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.id AS match_id, m.match_date, m.kickoff_time,
                       m.home_team_id, m.away_team_id,
                       home.canonical_name AS home_team,
                       away.canonical_name AS away_team,
                       m.home_goals, m.away_goals, m.result, m.result_code,
                       m.is_played, s.start_year, s.name AS season_name
                FROM matches AS m
                JOIN seasons AS s ON s.id = m.season_id
                JOIN teams AS home ON home.id = m.home_team_id
                JOIN teams AS away ON away.id = m.away_team_id
                ORDER BY m.match_date, m.kickoff_time, m.id
                """
            )
            columns = [col.name for col in cur.description]
            rows = cur.fetchall()
    if not rows:
        raise BacktestError("matches is empty")
    frame = pd.DataFrame(rows, columns=columns)
    frame["match_date"] = pd.to_datetime(frame["match_date"]).dt.date
    return frame


def load_training_features(start_year_before: int) -> pd.DataFrame:
    """Stored pre-match features for completed seasons strictly before `start_year_before`.

    Those rows were computed from earlier matches only, so they are safe training
    inputs. Season S itself is excluded even for matches already played at a
    checkpoint, matching production's rule that the live season is never trained on.
    """
    feature_sql = ", ".join(f"f.{name}" for name in FEATURE_COLUMNS)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT m.id AS match_id, m.match_date, m.result_code,
                       {feature_sql}
                FROM matches AS m
                JOIN seasons AS s ON s.id = m.season_id
                JOIN match_features AS f ON f.match_id = m.id
                WHERE s.start_year < %s
                  AND m.is_played
                  AND m.result_code IS NOT NULL
                  AND f.home_prior_n >= %s
                  AND f.away_prior_n >= %s
                ORDER BY m.match_date, m.id
                """,
                (start_year_before, MIN_PRIOR_N, MIN_PRIOR_N),
            )
            columns = [col.name for col in cur.description]
            rows = cur.fetchall()
    if not rows:
        raise BacktestError(f"No training rows before season {start_year_before}")
    return pd.DataFrame(rows, columns=columns)


def fit_logistic(train: pd.DataFrame):
    pipe = logreg_pipeline()
    pipe.fit(feature_matrix(train), target_vector(train))
    return pipe


def _rows(frame: pd.DataFrame) -> list[dict]:
    return frame.to_dict(orient="records")


def evaluate_checkpoint(
    *,
    season_played: list[dict],
    season_all: list[dict],
    teams: list[str],
    n_per_team: int,
    model,
    n_sims: int,
    seed: int,
    history_played: list[dict],
) -> dict:
    meta = cutoff_for_checkpoint(season_played, n_per_team)
    cutoff = meta["cutoff_date"]
    as_of = [row for row in season_all if row["match_date"] <= cutoff and row.get("home_goals") is not None]
    remaining = [row for row in season_all if row["match_date"] > cutoff]
    played_models = [
        PlayedMatch(row["home_team"], row["away_team"], int(row["home_goals"]), int(row["away_goals"]))
        for row in as_of
    ]
    table = build_table(played_models, teams)
    feature_history = history_played + as_of
    feats = remaining_feature_frame(feature_history, remaining)
    remaining_fx: list[RemainingFixture] = []
    if not feats.empty:
        proba = predict_proba_3way(model, feature_matrix(feats))
        for row, probs in zip(feats.itertuples(index=False), proba, strict=True):
            remaining_fx.append(
                RemainingFixture(
                    match_id=int(row.match_id),
                    home=row.home_team,
                    away=row.away_team,
                    p_away=float(probs[0]),
                    p_draw=float(probs[1]),
                    p_home=float(probs[2]),
                )
            )
    sim = simulate_season(
        teams=teams,
        table=table,
        remaining=remaining_fx,
        n_sims=n_sims,
        seed=seed,
    )
    ranked = sim["teams"]
    favorite = ranked[0]
    return {
        **meta,
        "cutoff_date": cutoff.isoformat(),
        "n_remaining": len(remaining_fx),
        "favorite": favorite["team"],
        "favorite_title_pct": float(favorite["title_prob"]),
        "title_table": [
            {
                "team": row["team"],
                "title_prob": float(row["title_prob"]),
                "rank": i + 1,
                "current_points": int(row["current_points"]),
                "current_played": int(row["current_played"]),
            }
            for i, row in enumerate(ranked)
        ],
    }


def score_against_champion(checkpoint: dict, champion: str) -> dict:
    ranks = {row["team"]: row for row in checkpoint["title_table"]}
    champ_row = ranks[champion]
    rank = int(champ_row["rank"])
    return {
        "season": checkpoint["season"],
        "checkpoint": checkpoint["n_per_team"],
        "cutoff_date": checkpoint["cutoff_date"],
        "n_played": checkpoint["n_played"],
        "played_min": checkpoint["played_min"],
        "played_max": checkpoint["played_max"],
        "forecast_favorite": checkpoint["favorite"],
        "favorite_title_pct": checkpoint["favorite_title_pct"],
        "actual_champion": champion,
        "champion_rank": rank,
        "champion_title_pct": float(champ_row["title_prob"]),
        "correct_1": rank == 1,
        "correct_top2": rank <= 2,
        "correct_top3": rank <= 3,
    }


def aggregate_by_checkpoint(rows: list[dict]) -> list[dict]:
    out = []
    for n in CHECKPOINTS:
        subset = [row for row in rows if row["checkpoint"] == n]
        if not subset:
            continue
        k = len(subset)
        out.append(
            {
                "checkpoint": n,
                "n_seasons": k,
                "hit_rate_1": sum(row["correct_1"] for row in subset) / k,
                "hit_rate_top2": sum(row["correct_top2"] for row in subset) / k,
                "hit_rate_top3": sum(row["correct_top3"] for row in subset) / k,
                "mean_champion_title_pct": float(np.mean([row["champion_title_pct"] for row in subset])),
            }
        )
    return out


def run_backtest(
    *,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
    checkpoints: tuple[int, ...] = CHECKPOINTS,
) -> dict:
    history = load_history()
    live = live_season_start_year()
    years = sorted(int(y) for y in history["start_year"].unique() if int(y) < live)
    results: list[dict] = []
    season_notes: list[dict] = []
    for start_year in years:
        season = history.loc[history["start_year"] == start_year]
        played = season.loc[season["is_played"]].copy()
        if len(played) < target_league_matches(max(checkpoints)):
            logger.info("Skip %s: not enough completed matches", short_season_name(start_year))
            continue
        try:
            train = load_training_features(start_year)
        except BacktestError as exc:
            logger.info("Skip %s: %s", short_season_name(start_year), exc)
            continue
        model = fit_logistic(train)
        teams = sorted(set(played["home_team"]).union(played["away_team"]))
        champion = actual_champion(_rows(played), teams)
        prior_played = _rows(
            history.loc[(history["start_year"] < start_year) & (history["is_played"])]
        )
        season_block = {
            "season": short_season_name(start_year),
            "start_year": start_year,
            "champion": champion,
            "n_train": int(len(train)),
            "n_played": int(len(played)),
        }
        logger.info(
            "Season %s champion=%s train_n=%s",
            season_block["season"],
            champion,
            len(train),
        )
        for n_per_team in checkpoints:
            raw = evaluate_checkpoint(
                season_played=_rows(played),
                season_all=_rows(played),
                teams=teams,
                n_per_team=n_per_team,
                model=model,
                n_sims=n_sims,
                seed=seed,
                history_played=prior_played,
            )
            raw["season"] = season_block["season"]
            scored = score_against_champion(raw, champion)
            results.append(scored)
            logger.info(
                "%s after %s: favorite=%s (%.1f%%) champion=%s rank=%s (%.1f%%) hit1=%s",
                scored["season"],
                n_per_team,
                scored["forecast_favorite"],
                100 * scored["favorite_title_pct"],
                scored["actual_champion"],
                scored["champion_rank"],
                100 * scored["champion_title_pct"],
                scored["correct_1"],
            )
        season_notes.append(season_block)

    report = {
        "feature_version": FEATURE_VERSION,
        "algorithm": ALGORITHM,
        "n_sims": int(n_sims),
        "seed": int(seed),
        "checkpoints": list(checkpoints),
        "cutoff_rule": (
            "After N matches per team on average: include every completed match "
            "in the season on or before the date of the (N*10)th chronological "
            "result. Same-day fixtures are not split. Games in hand are allowed; "
            "played_min/max are recorded."
        ),
        "training_rule": (
            "Unweighted logistic regression, the production algorithm, retrained "
            "once per forecast season on stored pre-match features from completed "
            "seasons strictly before that season. The forecast season is never in "
            "the training matrix. The live production artifact is never loaded."
        ),
        "feature_rule": (
            "Remaining-fixture Elo, last-5 form, and H2H use only matches on or "
            "before the cutoff, including earlier seasons. Later results in the "
            "same season are treated as unplayed."
        ),
        "wrote_production_artifacts": False,
        "seasons": season_notes,
        "rows": results,
        "by_checkpoint": aggregate_by_checkpoint(results),
    }
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", REPORT_PATH)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Historical Season Forecast backtest. Does not change production.")
    parser.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    run_backtest(n_sims=args.n_sims, seed=args.seed)


if __name__ == "__main__":
    main()
