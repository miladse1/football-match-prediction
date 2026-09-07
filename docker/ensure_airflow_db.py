"""Create the Airflow metadata database if the football Postgres user can."""

from __future__ import annotations

import os

import psycopg


def main() -> None:
    conn = psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "postgres"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        user=os.environ.get("POSTGRES_USER", "football"),
        password=os.environ.get("POSTGRES_PASSWORD", "football"),
        dbname=os.environ.get("POSTGRES_DB", "football_match_prediction"),
        autocommit=True,
    )
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = 'airflow'")
        if cur.fetchone() is None:
            cur.execute("CREATE DATABASE airflow")
            print("Created database airflow")
        else:
            print("Database airflow already exists")
    conn.close()


if __name__ == "__main__":
    main()
