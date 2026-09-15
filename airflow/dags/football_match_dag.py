"""Football match ETL + training DAG.

Business logic lives in football_pipeline.tasks. This file only names the steps
and their order so retries and logs are per stage.

Scheduled: 06:00 America/New_York on Mondays and Thursdays. One DAG, one
TaskGroup factory: each supported league runs the same eight stages with its
own competition code. Groups are chained so Spark/train never overlap. A
failure in one league is the TaskGroup id (E0, SP1, …) plus the stage name
and does not rewrite another league's rows.

Manual Trigger default is all five leagues. Set competition to E0/SP1/D1/I1/F1
for a single-league backfill. Local CSV path is optional and only valid with a
single competition.
"""

from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.models.param import Param
from airflow.operators.python import get_current_context, PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from football_pipeline.competitions import SUPPORTED_CODES, parse_competition_list

LOCAL_TZ = "America/New_York"
STAGE_TASKS = (
    "ingest_raw",
    "ingest_fixtures",
    "load_core_tables",
    "spark_features",
    "assemble_training_table",
    "train_evaluate",
    "predict_upcoming",
    "season_forecast",
)


def _ensure_selected(competition: str) -> dict:
    params = get_current_context()["params"]
    selected = parse_competition_list(params.get("competition"))
    if competition not in selected:
        raise AirflowSkipException(f"{competition} was not selected for this run")
    if params.get("from_file") and len(selected) > 1:
        raise ValueError("from_file can only be used when competition is a single league")
    return params


def _ingest_raw(competition: str, **_context) -> dict:
    from football_pipeline.tasks import ingest

    params = _ensure_selected(competition)
    return ingest(
        competition=competition,
        start_year=params.get("start_year"),
        end_year=params.get("end_year"),
        from_file=params.get("from_file"),
    )


def _ingest_fixtures(competition: str, **_context) -> dict:
    from football_pipeline.tasks import ingest_fixtures as run_fixtures

    _ensure_selected(competition)
    return run_fixtures(competition=competition)


def _load_core(competition: str, **_context) -> dict:
    from football_pipeline.tasks import load_core

    _ensure_selected(competition)
    return load_core(competition=competition)


def _spark_features(competition: str, **_context) -> int:
    from football_pipeline.tasks import spark_features as run_features

    _ensure_selected(competition)
    return run_features(competition=competition)


def _assemble(competition: str, **_context) -> dict:
    from football_pipeline.tasks import assemble_dataset

    _ensure_selected(competition)
    return assemble_dataset(competition=competition)


def _train(competition: str, **_context) -> dict:
    from football_pipeline.tasks import train_models

    _ensure_selected(competition)
    return train_models(competition=competition)


def _predict(competition: str, **_context) -> dict:
    from football_pipeline.tasks import predict_upcoming_matches

    _ensure_selected(competition)
    return predict_upcoming_matches(competition=competition)


def _forecast(competition: str, **_context) -> dict:
    from football_pipeline.tasks import simulate_live_season

    _ensure_selected(competition)
    return simulate_live_season(competition=competition)


def _league_group(code: str, *, wait_for_previous: bool) -> TaskGroup:
    """One reusable eight-stage pipeline, parameterised by competition code."""
    with TaskGroup(group_id=code) as group:
        ingest_raw = PythonOperator(
            task_id="ingest_raw",
            python_callable=_ingest_raw,
            op_kwargs={"competition": code},
            # Later leagues must still run when an earlier league was skipped
            # (single-league Trigger). A hard failure upstream still blocks
            # the rest of the DAG so a broken migrate/E0 run is obvious.
            trigger_rule=TriggerRule.NONE_FAILED if wait_for_previous else TriggerRule.ALL_SUCCESS,
        )
        ingest_fixtures = PythonOperator(
            task_id="ingest_fixtures",
            python_callable=_ingest_fixtures,
            op_kwargs={"competition": code},
        )
        load_core_tables = PythonOperator(
            task_id="load_core_tables",
            python_callable=_load_core,
            op_kwargs={"competition": code},
        )
        spark_features = PythonOperator(
            task_id="spark_features",
            python_callable=_spark_features,
            op_kwargs={"competition": code},
        )
        assemble_training_table = PythonOperator(
            task_id="assemble_training_table",
            python_callable=_assemble,
            op_kwargs={"competition": code},
        )
        train_evaluate = PythonOperator(
            task_id="train_evaluate",
            python_callable=_train,
            op_kwargs={"competition": code},
        )
        predict_upcoming = PythonOperator(
            task_id="predict_upcoming",
            python_callable=_predict,
            op_kwargs={"competition": code},
        )
        season_forecast = PythonOperator(
            task_id="season_forecast",
            python_callable=_forecast,
            op_kwargs={"competition": code},
        )
        (
            ingest_raw
            >> ingest_fixtures
            >> load_core_tables
            >> spark_features
            >> assemble_training_table
            >> train_evaluate
            >> predict_upcoming
            >> season_forecast
        )
    return group


@dag(
    dag_id="football_match_pipeline",
    description="Ingest Top 5 league seasons + fixtures → Spark features → per-league walk-forward → predict → season forecast",
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
            "all",
            type="string",
            title="Competition",
            description="all = E0, SP1, D1, I1, F1. Or one football-data.co.uk code for a single-league run.",
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
                "Automatically uses the current August–July season. "
                "Change only for a historical backfill."
            ),
        ),
        "from_file": Param(
            None,
            type=["null", "string"],
            title="Local CSV path",
            description="Optional. Leave blank to download. Set only with a single competition.",
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

    migrated = migrate_db()
    upstream = migrated
    # One TaskGroup factory, five sequential groups. Sequential so Spark/train
    # never overlap. NONE_FAILED on later ingest_raw lets a skipped league
    # (unselected on Trigger) pass through without blocking the next one.
    # A failed league still stops later leagues. Writes stay competition-scoped
    # either way, so a SP1 failure cannot rewrite E0 rows.
    for index, code in enumerate(SUPPORTED_CODES):
        group = _league_group(code, wait_for_previous=index > 0)
        upstream >> group
        upstream = group


football_match_pipeline()
