"""Apply SQL files in sql/ once each.

psycopg runs one statement per execute(), so files are split on ';'.
Keep migration files free of semicolons inside procedure bodies.
"""

from __future__ import annotations

import logging
from pathlib import Path

from football_pipeline.config import SQL_DIR
from football_pipeline.db import connect

logger = logging.getLogger(__name__)

MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def _statements(sql_text: str) -> list[str]:
    return [part.strip() for part in sql_text.split(";") if part.strip()]


def apply_migrations() -> None:
    sql_files = sorted(SQL_DIR.glob("*.sql"))
    if not sql_files:
        raise FileNotFoundError(f"No .sql files in {SQL_DIR}")

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(MIGRATIONS_DDL)
            for path in sql_files:
                _apply_file(cur, path)
        conn.commit()
        logger.info("Migrations complete")


def _apply_file(cur, path: Path) -> None:
    cur.execute(
        "SELECT 1 FROM schema_migrations WHERE filename = %s",
        (path.name,),
    )
    if cur.fetchone():
        logger.info("Skip %s (already applied)", path.name)
        return

    logger.info("Apply %s", path.name)
    for statement in _statements(path.read_text(encoding="utf-8")):
        cur.execute(statement)
    cur.execute(
        "INSERT INTO schema_migrations (filename) VALUES (%s)",
        (path.name,),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    apply_migrations()


if __name__ == "__main__":
    main()
