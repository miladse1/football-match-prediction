"""Callable steps used by the Airflow DAG. No Airflow imports here."""

from __future__ import annotations

from pathlib import Path

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
from football_pipeline.migrate import apply_migrations
from football_pipeline.summarize import collect_match_summary, write_pipeline_summary


def migrate() -> str:
    apply_migrations()
    return "ok"


def ingest(
    competition: str = "E0",
    start_year: int | str | None = None,
    end_year: int | str | None = None,
    from_file: str | None = None,
) -> dict:
    code = resolve_competition(competition)
    first = resolve_ingest_start_year(start_year, default=INGEST_START_YEAR)
    local = resolve_from_file(from_file)
    if local:
        summary = ingest_season(code, first, from_file=Path(local))
        write_pipeline_summary({"ingest": summary})
        return summary
    last = resolve_ingest_end_year(end_year, env_default=INGEST_END_YEAR)
    summary = ingest_seasons(code, first, end_year=last, force_download=True)
    write_pipeline_summary({"ingest": summary})
    return summary


def ingest_fixtures(competition: str = "E0") -> dict:
    summary = ingest_upcoming_fixtures(competition=resolve_competition(competition))
    write_pipeline_summary({"fixtures": summary})
    return summary


def load_core(competition: str = "E0", start_year: int | None = None) -> dict:
    loaded = load_matches(competition=resolve_competition(competition), start_year=start_year)
    counts = collect_match_summary()
    write_pipeline_summary({"load": loaded, "matches": counts})
    return {**loaded, "played": counts["played"], "unplayed": counts["unplayed"]}


def spark_features() -> int:
    from football_pipeline.build_features import build_and_store

    written = build_and_store()
    counts = collect_match_summary()
    write_pipeline_summary({"features": {"rows": written, **counts}})
    return written


def assemble_dataset() -> dict:
    summary = assemble_training_rows(
        train_end=parse_iso_date(TRAIN_END),
        valid_end=parse_iso_date(VALID_END),
        test_end=parse_iso_date(TEST_END),
        min_prior_n=MIN_PRIOR_N,
    )
    write_pipeline_summary({"training_rows": summary})
    return summary


def train_models() -> dict:
    from football_pipeline.train import train_and_evaluate

    report = train_and_evaluate()
    selected = report.get("selected_by_walkforward_log_loss") or report.get(
        "selected_by_valid_log_loss"
    )
    write_pipeline_summary(
        {
            "train": {
                "selected_by_walkforward_log_loss": selected,
                "selected_by_valid_log_loss": selected,
                "selected_sklearn_by_valid_log_loss": report.get("selected_sklearn_by_valid_log_loss"),
                "selection_reason": report.get("selection_reason"),
                "walkforward_summary": report.get("walkforward", {}).get("summary"),
                "production_train": report.get("production_train"),
                "test_holdout": report.get("test_holdout"),
            }
        }
    )
    return {
        "selected_by_walkforward_log_loss": selected,
        "selected_by_valid_log_loss": selected,
        "selection_reason": report.get("selection_reason"),
        "production_train": report.get("production_train"),
        "test_n": report.get("test_holdout", {}).get("n"),
    }


def predict_upcoming_matches() -> dict:
    from football_pipeline.predict import predict_upcoming

    records = predict_upcoming()
    payload = {"n_upcoming": len(records), "algorithm": records[0]["algorithm"] if records else None}
    write_pipeline_summary({"upcoming": payload})
    return payload
