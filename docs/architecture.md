# Architecture

PostgreSQL is the system of record. Airflow orchestrates the Python package. Spark builds pre-match features. The dashboard only reads stored predictions.

```text
football-data.co.uk (results CSVs)
fixturedownload.com (unplayed 2026/27 kickoffs)
        │
        ▼
  ingest → raw_match_payloads (JSONB)
        │
        ▼
  load_matches → competitions / seasons / teams / matches
        │
        ▼
  PySpark windows → match_features (strictly prior matches)
        │
        ▼
  training_rows (chrono split, min prior matches)
        │
        ▼
  walk-forward selection → production retrain → 2025/26 test (once)
        │
        ▼
  predict_upcoming → predictions (frozen once the match is played)
        │
        ▼
  read-only dashboard (127.0.0.1:8500)
```

## Why this shape

- **Raw vs curated.** Re-run transforms from stored payloads without re-downloading blindly.
- **Postgres as source of truth.** SQL, Spark, training, and the dashboard share one schema.
- **Spark for as-of history.** Rolling form, H2H, and Elo are “for each match, look only at earlier matches.”
- **Train only on the past.** Reported scores are forecast metrics, not shuffled-row metrics.

## Data sources

| Source | Role |
|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/data.php) season CSVs | Official results. `FTHG` / `FTAG` / `FTR` only. |
| [fixturedownload.com](https://fixturedownload.com) EPL JSON | Remaining 2026/27 kickoffs. Scores in that feed are ignored. |

Odds, shots, cards, and in-play stats are not used as features.

## Leakage rules

The unit of prediction is a match at time `t`.

Allowed: any match with kickoff **before** `t`.

Forbidden: this match’s goals or result; pandas `rolling()` without a shift; fitting scalers on the test season; random `train_test_split` as the reported score; closing odds.

Spark windows use `rowsBetween(-5, -1)` so the current row is excluded. A unit test plants a fake future result and asserts earlier features do not change.

## Chronological split

| Role | Dates | Seasons | Used for |
|---|---|---|---|
| History | through 2025-07-31 | 2018/19–2024/25 | Walk-forward windows and the final production retrain |
| Untouched test | 2025-08-01 through 2026-07-31 | 2025/26 | Evaluated **once** after model selection |
| Live | after 2026-07-31 | 2026/27 | `predict_upcoming` only |

Rows where either club has fewer than 5 prior matches are dropped from training.

## Airflow DAG

`football_match_pipeline` runs Mondays and Thursdays at 06:00 America/New_York (`schedule="0 6 * * 1,4"`, `catchup=False`, `max_active_runs=1`). Manual Trigger still uses the same defaults: competition `E0`, start year `2018`, end season `auto`, local CSV blank.

1. `migrate_db`
2. `ingest_raw`
3. `ingest_fixtures`
4. `load_core_tables`
5. `spark_features`
6. `assemble_training_table`
7. `train_evaluate`
8. `predict_upcoming`
9. `season_forecast`

Business logic lives in `src/football_pipeline/`, not in the DAG file.

## Local services

| Service | Bind |
|---|---|
| Dashboard | `127.0.0.1:8500` |
| Airflow UI | `127.0.0.1:8080` |
| PostgreSQL | `127.0.0.1:5432` |

Do not publish these on `0.0.0.0` without a reverse proxy and auth. Airflow and Postgres should stay private.
