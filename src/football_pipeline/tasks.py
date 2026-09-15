"""Callable steps used by the Airflow DAG. No Airflow imports here."""

from __future__ import annotations

from pathlib import Path

from football_pipeline.competitions import DEFAULT_COMPETITION
from football_pipeline.config import (
    INGEST_END_YEAR,
    INGEST_START_YEAR,
    MIN_PRIOR_N,
    TEST_END,
    TRAIN_END,
    VALID_END,
)
from football_pipeline.dataset import assemble_training_rows, parse_iso_date
from football_pipeline.fixtures import ingest_upcoming_fixtures
from football_pipeline.football_data import (
    resolve_competition,
    resolve_from_file,
    resolve_ingest_end_year,
    resolve_ingest_start_year,
)
from football_pipeline.ingest import ingest_season, ingest_seasons
from football_pipeline.load_matches import load_matches
from football_pipeline.seasons import season_windows
from football_pipeline.migrate import apply_migrations
from football_pipeline.quality import (
    check_features,
    check_matches,
    check_predictions,
    check_training_rows,
)
from football_pipeline.summarize import collect_match_summary, write_pipeline_summary


def migrate() -> str:
    apply_migrations()
    write_pipeline_summary({"season_windows": season_windows()})
    return "ok"


def ingest(
    competition: str = DEFAULT_COMPETITION,
    start_year: int | str | None = None,
    end_year: int | str | None = None,
    from_file: str | None = None,
) -> dict:
    code = resolve_competition(competition)
    first = resolve_ingest_start_year(start_year, default=INGEST_START_YEAR)
    local = resolve_from_file(from_file)
    if local:
        summary = ingest_season(code, first, from_file=Path(local))
        write_pipeline_summary({"ingest": summary}, competition=code)
        return summary
    last = resolve_ingest_end_year(end_year, env_default=INGEST_END_YEAR)
    summary = ingest_seasons(code, first, end_year=last, force_download=True)
    write_pipeline_summary({"ingest": summary}, competition=code)
    return summary


def ingest_fixtures(competition: str = DEFAULT_COMPETITION) -> dict:
    code = resolve_competition(competition)
    summary = ingest_upcoming_fixtures(competition=code)
    write_pipeline_summary({"fixtures": summary}, competition=code)
    return summary


def load_core(competition: str = DEFAULT_COMPETITION, start_year: int | None = None) -> dict:
    code = resolve_competition(competition)
    loaded = load_matches(competition=code, start_year=start_year)
    counts = collect_match_summary(competition=code)
    gate = check_matches(counts, competition=code)
    write_pipeline_summary(
        {"load": loaded, "matches": counts, "quality_load": gate},
        competition=code,
    )
    return {**loaded, "played": counts["played"], "unplayed": counts["unplayed"]}


def spark_features(competition: str = DEFAULT_COMPETITION) -> int:
    from football_pipeline.build_features import build_and_store

    code = resolve_competition(competition)
    written = build_and_store(competition=code)
    counts = collect_match_summary(competition=code)
    gate = check_features(expected_rows=written, competition=code)
    write_pipeline_summary(
        {"features": {"rows": written, **counts}, "quality_features": gate},
        competition=code,
    )
    return written


def assemble_dataset(competition: str = DEFAULT_COMPETITION) -> dict:
    code = resolve_competition(competition)
    summary = assemble_training_rows(
        train_end=parse_iso_date(TRAIN_END),
        valid_end=parse_iso_date(VALID_END),
        test_end=parse_iso_date(TEST_END),
        min_prior_n=MIN_PRIOR_N,
        competition=code,
    )
    gate = check_training_rows(summary, competition=code)
    write_pipeline_summary(
        {"training_rows": summary, "quality_training_rows": gate},
        competition=code,
    )
    return summary


def train_models(competition: str = DEFAULT_COMPETITION) -> dict:
    from football_pipeline.train import train_and_evaluate

    code = resolve_competition(competition)
    report = train_and_evaluate(competition=code)
    selected = report.get("selected_by_walkforward_log_loss") or report.get(
        "selected_by_valid_log_loss"
    )
    write_pipeline_summary(
        {
            "train": {
                "competition": code,
                "selected_by_walkforward_log_loss": selected,
                "selected_by_valid_log_loss": selected,
                "selected_sklearn_by_valid_log_loss": report.get("selected_sklearn_by_valid_log_loss"),
                "selection_reason": report.get("selection_reason"),
                "walkforward_summary": report.get("walkforward", {}).get("summary"),
                "production_train": report.get("production_train"),
                "test_holdout": report.get("test_holdout"),
            }
        },
        competition=code,
    )
    return {
        "competition": code,
        "selected_by_walkforward_log_loss": selected,
        "selected_by_valid_log_loss": selected,
        "selection_reason": report.get("selection_reason"),
        "production_train": report.get("production_train"),
        "test_n": report.get("test_holdout", {}).get("n"),
    }


def predict_upcoming_matches(competition: str = DEFAULT_COMPETITION) -> dict:
    from football_pipeline.predict import predict_upcoming

    code = resolve_competition(competition)
    records = predict_upcoming(competition=code)
    payload = {
        "competition": code,
        "n_upcoming": len(records),
        "algorithm": records[0]["algorithm"] if records else None,
    }
    gate = check_predictions(competition=code)
    write_pipeline_summary({"upcoming": payload, "quality_predictions": gate}, competition=code)
    return payload


def simulate_live_season(competition: str = DEFAULT_COMPETITION) -> dict:
    """Rebuild the season-forecast artifact from the current database. Raises on failure."""
    from football_pipeline.season_sim import run_live_forecast

    code = resolve_competition(competition)
    report = run_live_forecast(competition=code)
    favourite = report["teams"][0]["team"] if report.get("teams") else None
    summary = {
        "competition": code,
        "n_sims": report["n_sims"],
        "seed": report["seed"],
        "n_completed": report["n_completed"],
        "n_remaining": report["n_remaining"],
        "generated_at": report["generated_at"],
        "predictions_updated_at": report["predictions_updated_at"],
        "title_favourite": favourite,
    }
    write_pipeline_summary({"season_forecast": summary}, competition=code)
    return summary
