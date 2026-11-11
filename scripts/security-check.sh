#!/usr/bin/env bash
# Full security check for this public repository. Run before pushing.
#   scripts/security-check.sh          all checks
#   scripts/security-check.sh --quick  skip the dependency audit (no network)
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
fail=0
section() { printf '\n\033[1m%s\033[0m\n' "$*"; }
bad() { printf '\033[31mFAIL\033[0m  %s\n' "$*"; fail=1; }
ok() { printf '\033[32mok\033[0m    %s\n' "$*"; }

section "1. Files, private details, and secrets (working tree and full history)"
if .githooks/guard.sh all; then ok "git-guard"; else bad "git-guard (details above)"; fi

section "2. Secrets stay out of Git"
if [ -f .env ]; then
  git check-ignore -q .env && ok ".env is ignored by Git" || bad ".env is not ignored"
  perms="$(stat -f '%Lp' .env 2>/dev/null || stat -c '%a' .env)"
  [ "$perms" = 600 ] && ok ".env is readable only by you (mode 600)" || bad ".env mode is $perms; run: chmod 600 .env"
fi
tracked_data="$(git ls-files 'data/raw/*' 'data/processed/*' | grep -v '\.gitkeep$' || true)"
[ -z "$tracked_data" ] && ok "no downloaded or processed data tracked" || { printf '%s\n' "$tracked_data"; bad "data files are tracked"; }

if [ "${1:-}" != "--quick" ]; then
  section "3. Known vulnerabilities in installed Python dependencies"
  if command -v pip-audit >/dev/null 2>&1 && [ -x .venv/bin/python ]; then
    if .venv/bin/python -m pip freeze --exclude-editable | pip-audit -r /dev/stdin --no-deps --disable-pip --progress-spinner off >/tmp/pip-audit.$$ 2>&1; then ok "pip-audit"; else cat /tmp/pip-audit.$$; bad "Python dependencies"; fi
    rm -f /tmp/pip-audit.$$
  else
    bad "needs pip-audit (brew install pip-audit) and .venv"
  fi
fi

echo
if [ "$fail" -ne 0 ]; then echo "Security check failed."; exit 1; fi
echo "Security check passed."
