# Premier League Match Prediction

End-to-end data pipeline and model that estimates **Home / Draw / Away** probabilities for Premier League matches, then serves them in a local dashboard.

The production model is **unweighted logistic regression**. Pre-match probabilities are frozen once a result arrives.

## Problem

Most public football models either leak future information, report shuffled-row accuracy, or hide the Draw class. This project treats match prediction as a **time-series forecasting** problem:

- ingest official results and remaining fixtures
- store a replayable curated schema in PostgreSQL
- build features only from matches that already finished
- select a model with expanding-window walk-forward validation
- score an untouched later season once
- show live 2026/27 probabilities without rewriting history

## Architecture

```text
football-data.co.uk (results)
fixturedownload.com (upcoming kickoffs)
        │
        ▼
  Airflow DAG  →  PostgreSQL
        │              │
        ├─ ingest / load matches
        ├─ PySpark pre-match features
        ├─ walk-forward train / select
        └─ predict remaining fixtures
                       │
                       ▼
              dashboard (read-only)
```

Design notes: [docs/architecture.md](docs/architecture.md)

## Tech stack

Python, PostgreSQL, Apache Airflow 2.10, PySpark 3.5, pandas, scikit-learn, XGBoost, FastAPI, Docker Compose.

## Data sources

| Source | What it supplies |
|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/data.php) | Premier League season CSVs from 2018/19 through the current season. Results only (`FTHG`, `FTAG`, `FTR`). |
| [fixturedownload.com](https://fixturedownload.com) | Remaining 2026/27 kickoffs. Scores in that feed are ignored; official results stay on football-data.co.uk. |

## Database and ETL

PostgreSQL holds `raw_match_payloads`, `competitions`, `seasons`, `teams`, `matches`, `match_features`, `training_rows`, `model_runs`, and `predictions`.

The DAG `football_match_pipeline` runs the same full path on a schedule **and** on demand (Airflow **Trigger**):

1. `migrate_db`
2. `ingest_raw` — download season CSVs
3. `ingest_fixtures` — remaining unplayed kickoffs
4. `load_core_tables` — upsert curated matches (results never lose to a blank fixture row)
5. `spark_features` — leakage-safe rolling form, H2H, Elo
6. `assemble_training_table` — chrono split, drop clubs with fewer than 5 prior matches
7. `train_evaluate` — walk-forward, then one production retrain
8. `predict_upcoming` — score `is_played = false` rows; skip updates once a match is played
9. `season_forecast` — Monte Carlo title/top-4/relegation outlook from frozen probabilities

Schedule: `0 6 * * 1,4` (Monday and Thursday 06:00 **America/New_York**). Catchup is off. At most one run at a time.

## Feature engineering

Feature version `v2-draw-aware`. Every value is computed from **earlier** matches only:

- last-five form (win rate, points, goal difference, goals scored/conceded)
- home-only / away-only splits
- head-to-head win and draw rates
- pre-match Elo and the absolute Elo gap
- last-five draw rates and closeness proxies (PPG and goal differentials)

A unit test plants a fake future result and asserts that earlier features do not change.

## Models tested

Walk-forward compared four classifiers on the same features:

| Model | Mean OOT log loss | Mean accuracy | Mean draw F1 |
|---|---:|---:|---:|
| **Logistic regression (unweighted)** | **0.973** | 0.542 | 0.000 |
| XGBoost | 0.977 | 0.549 | 0.006 |
| Logistic regression (class-weighted) | 0.993 | 0.504 | 0.209 |
| Random forest | 1.024 | 0.490 | 0.264 |

A **Dixon–Coles Poisson** goals model is also implemented and unit-tested as a separate baseline (it predicts 1X2 from a score matrix, not from the rolling-form table). It was not part of the production selection grid.

## Why unweighted logistic regression

Primary selection metric: **mean out-of-time log loss** across four walk-forward folds. Macro-F1 and draw F1 are diagnostics only.

Unweighted logistic regression had the lowest mean log loss (0.973; median 0.978). XGBoost was close but only won the last fold, which is why a single-season validation would have picked it. Class-weighted logistic regression and random forest predict more draws and lose on log loss.

## Walk-forward validation

Expanding window. 2025/26 is **not** used for features, hyperparameters, class weights, thresholds, or model choice. 2026/27 is live-only.

| Fold | Train through | Validate on |
|---|---|---|
| 1 | 2020/21 | 2021/22 |
| 2 | 2021/22 | 2022/23 |
| 3 | 2022/23 | 2023/24 |
| 4 | 2023/24 | 2024/25 |

The winner is retrained on all permitted history through 2024/25, then scored once on 2025/26.

## Untouched 2025/26 test

Production logistic regression, n = 375:

| Metric | Value |
|---|---|
| Accuracy | 50.7% |
| Log loss | 1.029 |
| Macro-F1 | 0.377 |
| Away P / R / F1 | 0.475 / 0.509 / 0.492 |
| Draw P / R / F1 | 0.000 / 0.000 / 0.000 |
| Home P / R / F1 | 0.522 / 0.830 / 0.641 |

Confusion matrix (rows = true away / draw / home): `[[58, 0, 56], [37, 0, 65], [27, 0, 132]]`. Mean predicted P(Draw) was 0.217; argmax never chose Draw.

## Draw calibration

Out-of-sample only (walk-forward folds + 2025/26 test, n = 1,875):

| Class | Mean predicted | Actual | Gap | Brier | Argmax share |
|---|---:|---:|---:|---:|---:|
| Draw | 20.8% | 23.7% | +3.0 pp | 0.182 | 0% |
| Away | 33.3% | 32.1% | −1.2 pp | 0.190 | 36% |
| Home | 46.0% | 44.2% | −1.8 pp | 0.213 | 64% |

Draw probabilities sit in the right neighborhood but are **mildly underpredicted**, not overpredicted. The bulk of matches (predicted Draw around 20–30%) is close; the reliability curve is a bit flat. Predicted Draw never leaves about 4–37%, so Home or Away is almost always the argmax even when P(Draw) is honest.

That is a known limitation of the production model, not a dashboard rounding choice.

## Dashboard

Read-only FastAPI app at [http://127.0.0.1:8500](http://127.0.0.1:8500) (loopback only). It does not train models.

| Page | Contents |
|---|---|
| Overview | Live 2026/27 snapshot, next fixtures, latest completed score |
| Upcoming | Unplayed fixtures with frozen pre-match probabilities |
| Results | Completed 2026/27 matches vs those original probabilities |
| Performance | Live accuracy and log loss |
| About | Walk-forward method, test metrics, draw limitation |

Upcoming and Results paginate (~16 matches) and filter by team or date.

## Run locally

Copy `.env.example` to `.env`. Generate an Airflow Fernet key and fill the placeholders. Never commit `.env`.

```bash
python -c "import base64,os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"
docker compose up -d postgres airflow-init airflow-webserver airflow-scheduler dashboard
```

Then:

- Dashboard: [http://127.0.0.1:8500](http://127.0.0.1:8500)
- Airflow: [http://127.0.0.1:8080](http://127.0.0.1:8080) (credentials from `.env`)

PostgreSQL is bound to `127.0.0.1:5432`. Airflow and the dashboard are loopback-only. Do not publish them on `0.0.0.0` without a reverse proxy and auth.

The first pipeline run needs an Airflow Trigger (see below). Spark feature jobs run in Docker; host Python does not need a local Spark install.

Unit tests (no Docker Spark required):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Refresh data

In the Airflow UI, open `football_match_pipeline` and click **Trigger**. Leave the form unchanged:

- Competition `E0`
- Historical start year `2018`
- End season `auto`
- Local CSV path blank

That ingests 2018/19 through the current August–July season, rebuilds features, retrains with the same walk-forward procedure, and scores remaining unplayed fixtures. Matches that already have a result keep their original probabilities.

CLI equivalent (do not pass `--conf`):

```bash
docker compose exec airflow-scheduler airflow dags trigger football_match_pipeline
```

Reload the dashboard after the run succeeds. The same DAG also runs automatically at 06:00 America/New_York on Mondays and Thursdays (`0 6 * * 1,4`). Catchup is disabled, and `max_active_runs=1` skips a new run while one is already going.

Stop services: `docker compose stop`. Start the dashboard later with `docker compose up -d postgres dashboard`.

## Limitations

- Premier League only (football-data.co.uk code `E0`)
- No expected goals, injuries, or lineups
- Unweighted logistic regression almost never has Draw as the highest-probability class
- Draw probabilities are slightly too low on average (~3 pp on pooled OOS)
- Live 2026/27 accuracy is a small sample until more matches finish
- Local-only; not a public betting product

## Future work

- Expected goals (xG) and shot quality
- Player availability / injuries
- Other leagues with the same leakage-safe pipeline
- League-title and placement simulation from remaining-fixture probabilities
- Recalibration or a Draw-specific head only if it improves out-of-time log loss
