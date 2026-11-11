# Secure development workflow

How to change this project, alone or with an AI assistant, without leaking secrets or personal data.

## Once per computer or clone

```bash
brew install gitleaks pip-audit
scripts/setup-dev-security.sh     # enables .githooks, creates ~/.config/git-guard/private-patterns.txt
```

Open `~/.config/git-guard/private-patterns.txt` and add your personal email, phone number (with and
without dashes), student number, and street address, one per line. The hooks block any commit that
contains them. The file never leaves your computer.

## Where things live

| | Location | In Git? |
| --- | --- | --- |
| Source code, SQL, DAGs, small test fixtures | this repository (public) | yes |
| `.env` (PostgreSQL password, Airflow Fernet and webserver keys) | repository root, mode 600, ignored | never |
| Downloaded CSVs, processed outputs, trained models | `data/raw/`, `data/processed/` (ignored) | never |
| Database | PostgreSQL in Docker (`pgdata` volume) | never |

## Everyday loop

1. **Develop.** `docker compose up` (see the README's *Run locally*). Claude Code and Cursor read `AGENTS.md`;
   Claude Code also enforces `.claude/settings.json` (asks before commit/push, denies reading `.env`).
2. **Test.** `.venv/bin/python -m pytest -q` and `ruff check src tests airflow`.
3. **Review.** Before staging, read the diff yourself:
   ```bash
   git status
   git diff                 # what changed
   git add -p               # stage hunk by hunk instead of `git add .`
   git diff --cached --stat # exactly what the commit will contain
   ```
   README numbers must come from files the pipeline wrote in `data/processed/`.
4. **Commit.** `git commit`. The pre-commit hook checks blocked file types, sizes, browser-exposed
   secrets, your private details, and secrets (gitleaks). If it blocks, read the message; do not use
   `--no-verify`.
5. **Check before pushing.** `scripts/security-check.sh` (working tree, full history, `.env`
   permissions, tracked data, dependency audit).
6. **Push.** `git push`. The pre-push hook scans every outgoing commit again.
7. **Watch.** GitHub runs the Security workflow on every push and weekly. Dependabot opens
   grouped update pull requests on Mondays and security fixes immediately; review and merge them
   after CI passes.

## If something goes wrong

- **The hook blocks a file that should be committed** (for example a fictional PDF fixture): move
  it under `tests/fixtures/` or add a narrow, reviewed line to `.githooks/allowed-paths`.
- **The hook reports a false-positive secret:** add its fingerprint (printed by gitleaks) to
  `.gitleaksignore` with a comment saying why.
- **A real secret was committed but not pushed:** `git reset --soft HEAD~1`, remove the secret,
  commit again. Rotate the secret anyway if it was ever pushed or shared.
- **A real secret reached GitHub:** rotate it immediately (new PostgreSQL password; a new
  `AIRFLOW_FERNET_KEY` means re-entering Airflow connections), then clean history only with a deliberate plan; deleting the file in a new commit
  does not remove it from history.

## Publishing

This repository is already public. GitHub secret scanning and push protection are on; keep them on,
and turn on Dependabot alerts and security updates.
