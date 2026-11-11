#!/usr/bin/env bash
# One-time setup on a new computer or a fresh clone:
#   - turns on the Git hooks in .githooks (secret, private-data, and sensitive-file checks)
#   - checks that gitleaks and pip-audit are installed
#   - creates your private-patterns file outside the repository if it does not exist
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

git config core.hooksPath .githooks
chmod +x .githooks/guard.sh .githooks/pre-commit .githooks/pre-push scripts/*.sh
echo "Git hooks enabled (core.hooksPath=.githooks)."

missing=""
command -v gitleaks >/dev/null 2>&1 || missing="$missing gitleaks"
command -v pip-audit >/dev/null 2>&1 || missing="$missing pip-audit"
if [ -n "$missing" ]; then
  echo "Missing tools:$missing"
  echo "Install with: brew install$missing"
fi

PATTERNS="${GIT_GUARD_PRIVATE_PATTERNS:-$HOME/.config/git-guard/private-patterns.txt}"
if [ ! -f "$PATTERNS" ]; then
  mkdir -p "$(dirname "$PATTERNS")"
  cat >"$PATTERNS" <<'EOF'
# Private details that must never appear in any repository (one per line, case-insensitive,
# matched as plain text). This file stays on your computer and is never committed.
# Examples (replace and uncomment):
# your.personal@email.com
# 416-555-0123
# 4165550123
# your student number
# your street address
EOF
  chmod 600 "$PATTERNS"
  echo "Created $PATTERNS: add your email, phone, student number, and address."
else
  echo "Private patterns: $PATTERNS ($(grep -cvE '^[[:space:]]*(#|$)' "$PATTERNS" || true) entries)"
fi
