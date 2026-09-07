"""PostgreSQL helpers."""

from __future__ import annotations

import psycopg

from football_pipeline.config import database_url


def connect() -> psycopg.Connection:
    return psycopg.connect(database_url())
