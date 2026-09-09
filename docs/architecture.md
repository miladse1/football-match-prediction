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

The loader never deletes a match that has been played or that already carries a
stored prediction. An incomplete upstream feed logs a warning and preserves the
row instead, so scored history cannot be destroyed by a bad download.

## Chronological split

Every boundary is derived from today's date by `src/football_pipeline/seasons.py`,
the single source of truth for season logic. Nothing is pinned to a season, so
the whole window shifts by one season on 1 August each year.

| Role | Rule | Today | Used for |
|---|---|---|---|
| History | through the season before the holdout | 2018/19–2024/25 | Walk-forward windows and the final production retrain |
| Untouched test | the most recently completed season | 2025/26 | Evaluated **once** after model selection |
| Live | the season containing today | 2026/27 | `predict_upcoming` only |

Walk-forward folds are the four completed seasons immediately before the
holdout, clamped so a fold never validates on a season with fewer than two
seasons of history behind it. The holdout can never be a validation fold; a test
asserts that invariant at four different points in the calendar.

Rows where either club has fewer than 5 prior matches are dropped from training.

## Data quality gates

`src/football_pipeline/quality.py` runs between stages and fails the Airflow task
rather than letting bad data reach training:

| After | Checks |
|---|---|
| `load_core_tables` | completed seasons have 380 matches, 20 clubs per season, played matches carry a result, plausible draw rate |
| `spark_features` | one feature row per match, Elo inside 1000–2200, prior-match counts in 0–5 |
| `assemble_training_table` | every split populated, no NULL Elo, valid result codes |
| `predict_upcoming` | probabilities in range and summing to 1, `predicted_class` equals the stored argmax |

## Model artifacts and runs

Artifacts are written to `models/<algorithm>-run<NNNNN>.joblib`, keyed by the
`model_runs` id, so a later run can never overwrite an earlier run's model.
Production code resolves the current model through
`src/football_pipeline/registry.py`, which reads `artifact_path` from the run row
itself. A retrain whose inputs and scores are unchanged reuses its existing run
instead of inserting a duplicate run and another full set of holdout predictions.

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
