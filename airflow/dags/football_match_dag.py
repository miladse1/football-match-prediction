"""Football match ETL + training DAG.

Business logic lives in football_pipeline.tasks. This file only names the steps
and their order so retries and logs are per stage.

Scheduled: 06:00 America/New_York on Mondays and Thursdays. Manual Trigger still
works with the same defaults: Premier League (E0) from 2018/19 through the
current August–July season. Local CSV path is optional and blank. Override the
params only for a backfill or a local CSV.
"""

from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.operators.python import get_current_context

LOCAL_TZ = "America/New_York"


@dag(
    dag_id="football_match_pipeline",
    description="Ingest Premier League seasons + upcoming fixtures → Spark features → walk-forward selection → predict upcoming → season forecast",
    start_date=pendulum.datetime(2023, 8, 1, tz=LOCAL_TZ),
    schedule="0 6 * * 1,4",
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "football",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    },
    params={
        "competition": Param(
            "E0",
            type="string",
            title="Competition",
            description="football-data.co.uk division code. Leave as E0 for Premier League.",
        ),
        "start_year": Param(
            2018,
            type="integer",
            title="Historical start year",
            description="First season start year to ingest (2018 = 2018/19). Change only for a partial backfill.",
        ),
        "end_year": Param(
            "auto",
            type="string",
            title="End season",
            description=(
                "Automatically uses the current Premier League season. "
                "Change only for a historical backfill."
            ),
        ),
        "from_file": Param(
            None,
            type=["null", "string"],
            title="Local CSV path",
            description="Optional. Leave blank to download. Set only to load a local CSV for a single season.",
        ),
    },
    tags=["football", "etl", "ml"],
    doc_md=__doc__,
)
def football_match_pipeline():
    @task
    def migrate_db() -> str:
        from football_pipeline.tasks import migrate

        return migrate()

    @task
    def ingest_raw() -> dict:
        from football_pipeline.tasks import ingest

        params = get_current_context()["params"]
        return ingest(
            competition=params.get("competition"),
            start_year=params.get("start_year"),
            end_year=params.get("end_year"),
            from_file=params.get("from_file"),
        )

    @task
    def ingest_fixtures() -> dict:
        from football_pipeline.tasks import ingest_fixtures as run_fixtures

        params = get_current_context()["params"]
        return run_fixtures(competition=params.get("competition"))

    @task
    def load_core_tables() -> dict:
        from football_pipeline.tasks import load_core

        params = get_current_context()["params"]
        return load_core(competition=params.get("competition"))

    @task
    def spark_features() -> int:
        from football_pipeline.tasks import spark_features as run_features

        return run_features()

    @task
    def assemble_training_table() -> dict:
        from football_pipeline.tasks import assemble_dataset

        return assemble_dataset()

    @task
    def train_evaluate() -> dict:
        from football_pipeline.tasks import train_models

        return train_models()

    @task
    def predict_upcoming() -> dict:
        from football_pipeline.tasks import predict_upcoming_matches

        return predict_upcoming_matches()

    @task
    def season_forecast() -> dict:
        from football_pipeline.tasks import simulate_live_season

        return simulate_live_season()

    (
        migrate_db()
        >> ingest_raw()
        >> ingest_fixtures()
        >> load_core_tables()
        >> spark_features()
        >> assemble_training_table()
        >> train_evaluate()
        >> predict_upcoming()
        >> season_forecast()
    )


football_match_pipeline()
