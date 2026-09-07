"""Paths and environment-backed settings.

`.env` is loaded from the repo root so CLI commands work from any cwd.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

SQL_DIR = ROOT / "sql"
RAW_DATA_DIR = ROOT / "data" / "raw"

POSTGRES_USER = os.getenv("POSTGRES_USER", "football")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "football")
POSTGRES_DB = os.getenv("POSTGRES_DB", "football_match_prediction")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# Chronological split (inclusive end dates).
# train 2018/19–2023/24, valid 2024/25, test 2025/26 only. 2026/27 is live.
TRAIN_END = os.getenv("TRAIN_END", "2024-07-31")
VALID_END = os.getenv("VALID_END", "2025-07-31")
TEST_END = os.getenv("TEST_END", "2026-07-31")
MIN_PRIOR_N = int(os.getenv("MIN_PRIOR_N", "5"))
INGEST_START_YEAR = int(os.getenv("INGEST_START_YEAR", "2018"))
_INGEST_END = os.getenv("INGEST_END_YEAR")
INGEST_END_YEAR = int(_INGEST_END) if _INGEST_END else None
FIXTURES_URL = os.getenv(
    "FIXTURES_URL",
    "https://fixturedownload.com/feed/json/epl-2026",
)
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8500"))


def database_url() -> str:
    return (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
