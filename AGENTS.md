# Instructions for AI coding assistants

This repository is **public**. Everything committed is visible to anyone.

## Rules

1. **Never open, print, or commit `.env`.** It holds the PostgreSQL password and Airflow's Fernet and
   webserver keys. Use `.env.example` for variable names.
2. **No data dumps.** Downloaded season CSVs (`data/raw/`), processed outputs and models
   (`data/processed/`), logs, and Playwright output stay out of Git. Test fixtures are small excerpts
   in `tests/fixtures/`.
3. **No leakage.** Features may use only matches that finished before kickoff. The holdout season is
   never used for features, tuning, or model choice; keep the leakage tests passing.
4. **Report real numbers only.** README metrics must come from files the pipeline wrote; never edit
   them by hand or round them favourably.
5. **Commits.** Show the user `git status` and `git diff --cached --stat` first. The pre-commit hook
   must pass; never use `--no-verify` or weaken `.githooks/`, `.gitleaks.toml`, or `.gitignore`
   without the user's explicit approval.
6. **Ask first** before committing, pushing, changing GitHub settings, or rewriting history.

## Commands

```bash
scripts/setup-dev-security.sh        # once per clone: enables the Git hooks
scripts/security-check.sh            # full check before pushing
.venv/bin/python -m pytest -q        # unit tests (RUN_SPARK_TESTS=1 / RUN_DB_TESTS=1 for the rest)
ruff check src tests airflow
```
