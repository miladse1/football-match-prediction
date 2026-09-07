"""Read curated matches from Postgres, run Spark feature job, upsert match_features."""

from __future__ import annotations

import argparse
import logging
from datetime import date, time
from typing import Any

from football_pipeline.db import connect
from football_pipeline.constants import FEATURE_VERSION
from football_pipeline.dataset import FEATURE_COLUMNS
from football_pipeline.spark_features import build_feature_frame, spark_session

logger = logging.getLogger(__name__)


def fetch_matches() -> list[dict[str, Any]]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, match_date, kickoff_time, home_team_id, away_team_id,
                       home_goals, away_goals, result, is_played
                FROM matches
                ORDER BY match_date, kickoff_time, id
                """
            )
            rows = []
            for rec in cur.fetchall():
                kickoff = rec[2]
                rows.append(
                    {
                        "match_id": rec[0],
                        "match_date": rec[1] if isinstance(rec[1], date) else rec[1],
                        "kickoff_time": kickoff.strftime("%H:%M:%S") if isinstance(kickoff, time) else (str(kickoff) if kickoff else ""),
                        "home_team_id": rec[3],
                        "away_team_id": rec[4],
                        "home_goals": rec[5],
                        "away_goals": rec[6],
                        "result": rec[7],
                        "is_played": rec[8],
                    }
                )
            return rows


def upsert_features(records: list[dict[str, Any]]) -> int:
    cols = ["match_id", *FEATURE_COLUMNS, "home_prior_n", "away_prior_n", "feature_version"]
    col_sql = ", ".join(cols)
    placeholders = ", ".join(f"%({name})s" for name in cols)
    updates = ", ".join(f"{name} = EXCLUDED.{name}" for name in cols if name != "match_id")
    sql = f"""
        INSERT INTO match_features ({col_sql}, computed_at)
        VALUES ({placeholders}, NOW())
        ON CONFLICT (match_id) DO UPDATE SET
            {updates},
            computed_at = NOW()
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, records)
        conn.commit()
    return len(records)


def _row_to_record(row) -> dict[str, Any]:
    data = row.asDict()
    data["feature_version"] = data.get("feature_version") or FEATURE_VERSION
    data["home_prior_n"] = int(data["home_prior_n"] or 0)
    data["away_prior_n"] = int(data["away_prior_n"] or 0)
    return data


def build_and_store() -> int:
    rows = fetch_matches()
    if not any(row.get("is_played") for row in rows):
        raise RuntimeError("No played matches in Postgres. Run ingest + load_matches first.")

    spark = spark_session()
    try:
        frame = build_feature_frame(spark, rows)
        records = [_row_to_record(row) for row in frame.collect()]
    finally:
        spark.stop()

    written = upsert_features(records)
    logger.info("Wrote %s match_features rows (%s)", written, FEATURE_VERSION)
    return written


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    argparse.ArgumentParser(description="Build pre-match features with Spark.").parse_args()
    build_and_store()


if __name__ == "__main__":
    main()
