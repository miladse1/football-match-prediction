#!/usr/bin/env bash
# git-guard: keeps secrets, private data, and sensitive files out of Git.
#
#   .githooks/guard.sh staged          what is staged for the next commit (pre-commit hook)
#   .githooks/guard.sh range A..B      commits about to be pushed (pre-push hook)
#   .githooks/guard.sh all             every tracked file, every file `git add .` would add,
#                                      and the full Git history (manual check and CI)
#
# Checks, in order:
#   1. blocked paths   .githooks/blocked-paths (regex per line), minus .githooks/allowed-paths
#   2. file size       over guard.maxFileKb (default 5120 KB); `git config guard.maxFileKb 8192`
#   3. browser-exposed secrets   NEXT_PUBLIC_/VITE_/REACT_APP_/EXPO_PUBLIC_ variables named like secrets
#   4. private details  fixed strings in ~/.config/git-guard/private-patterns.txt (kept outside
#                       every repository: email, phone, student number, address, ...);
#                       .githooks/private-allowed lists reviewed exceptions such as LICENSE
#   5. gitleaks         secret scanner with this repository's .gitleaks.toml
#
# A finding blocks the commit or push. Fix it, or for a reviewed false positive add an entry to
# .githooks/allowed-paths or .gitleaksignore (both are committed, so the exception is visible in
# review). Never bypass with --no-verify.
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel)" || exit 2
cd "$ROOT" || exit 2
HOOKS="$ROOT/.githooks"
PRIVATE_PATTERNS="${GIT_GUARD_PRIVATE_PATTERNS:-$HOME/.config/git-guard/private-patterns.txt}"
MAX_KB="$(git config --get guard.maxFileKb 2>/dev/null || echo 5120)"
MODE="${1:-staged}"
RANGE="${2:-}"
TIP=""
FAIL=0
TMP="$(mktemp -d "${TMPDIR:-/tmp}/git-guard.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

if [ -t 2 ]; then RED=$'\033[31m'; YEL=$'\033[33m'; GRN=$'\033[32m'; DIM=$'\033[2m'; OFF=$'\033[0m'; else RED=""; YEL=""; GRN=""; DIM=""; OFF=""; fi
block() { printf '%sBLOCKED%s  %s\n' "$RED" "$OFF" "$*" >&2; FAIL=1; }
warn() { printf '%sNOTE%s     %s\n' "$YEL" "$OFF" "$*" >&2; }
step() { printf '%s== %s%s\n' "$DIM" "$*" "$OFF" >&2; }

# ---------------------------------------------------------------- file list and content access
case "$MODE" in
  staged) git diff --cached --name-only --diff-filter=ACMR >"$TMP/files" ;;
  range)
    [ -n "$RANGE" ] || { echo "usage: guard.sh range <base>..<tip>" >&2; exit 2; }
    TIP="${RANGE##*..}"
    BASE="${RANGE%%..*}"
    if [ "$BASE" = EMPTY ]; then # first push of a branch with no ancestor on the remote
      git diff --name-only --diff-filter=ACMR 4b825dc642cb6eb9a060e54bf8d69288fbee4904 "$TIP" >"$TMP/files"
      GL_LOG_OPTS="$TIP"
    else
      git diff --name-only --diff-filter=ACMR "$RANGE" >"$TMP/files"
      GL_LOG_OPTS="$RANGE"
    fi ;;
  all) { git ls-files; git ls-files --others --exclude-standard; } | sort -u >"$TMP/files" ;;
  *) echo "usage: guard.sh staged | range <base>..<tip> | all" >&2; exit 2 ;;
esac

content() { # print one file's content as it will be committed or pushed
  case "$MODE" in
    staged) git show ":$1" 2>/dev/null ;;
    range) git show "$TIP:$1" 2>/dev/null ;;
    all) [ -f "$1" ] && cat "$1" ;;
  esac
}
size_bytes() {
  case "$MODE" in
    staged) git cat-file -s ":$1" 2>/dev/null || echo 0 ;;
    range) git cat-file -s "$TIP:$1" 2>/dev/null || echo 0 ;;
    all) if [ -f "$1" ]; then wc -c <"$1" | tr -d ' '; else echo 0; fi ;;
  esac
}
patterns() { [ -f "$1" ] && grep -vE '^[[:space:]]*(#|$)' "$1"; }
is_guard_file() { case "$1" in .githooks/* | .gitleaks.toml | .gitleaksignore) return 0 ;; *) return 1 ;; esac; }

COUNT="$(grep -c . "$TMP/files" || true)"
[ "$COUNT" -eq 0 ] && [ "$MODE" != all ] && exit 0

# ---------------------------------------------------------------- 1. blocked paths
step "paths ($COUNT files)"
patterns "$HOOKS/blocked-paths" >"$TMP/blocked" || : >"$TMP/blocked"
patterns "$HOOKS/allowed-paths" >"$TMP/allowed" || : >"$TMP/allowed"
if [ -s "$TMP/blocked" ]; then
  while IFS= read -r f; do
    if printf '%s\n' "$f" | grep -qiE -f "$TMP/blocked"; then
      if [ -s "$TMP/allowed" ] && printf '%s\n' "$f" | grep -qiE -f "$TMP/allowed"; then continue; fi
      block "$f  (sensitive file type or location; see .githooks/blocked-paths)"
    fi
  done <"$TMP/files"
fi

# ---------------------------------------------------------------- 2. size
step "sizes (limit ${MAX_KB} KB)"
while IFS= read -r f; do
  bytes="$(size_bytes "$f")"
  if [ "${bytes:-0}" -gt $((MAX_KB * 1024)) ]; then
    block "$f  ($((bytes / 1024)) KB is over ${MAX_KB} KB; large files are often data dumps or backups)"
  fi
done <"$TMP/files"

# ---------------------------------------------------------------- 3. secrets in browser-exposed variables
step "browser-exposed variables"
PUBLIC_SECRET='\b(NEXT_PUBLIC|VITE|REACT_APP|EXPO_PUBLIC)_[A-Z0-9_]*(SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE|API_KEY|APIKEY|ACCESS_KEY)'
while IFS= read -r f; do
  is_guard_file "$f" && continue
  case "$f" in *.ts | *.tsx | *.js | *.jsx | *.mjs | *.cjs | *.vue | *.svelte | *.html | .env* | */.env* | *.yml | *.yaml | *.json | *.toml) ;; *) continue ;; esac
  lines="$(content "$f" | grep -nE "$PUBLIC_SECRET" | cut -d: -f1 | tr '\n' ' ')"
  [ -n "$lines" ] && block "$f  line(s) $lines (a secret-named variable with a public prefix ends up in the browser bundle)"
done <"$TMP/files"

# ---------------------------------------------------------------- 4. private details
step "private details"
if [ -f "$PRIVATE_PATTERNS" ] && patterns "$PRIVATE_PATTERNS" >"$TMP/private" && [ -s "$TMP/private" ]; then
  patterns "$HOOKS/private-allowed" >"$TMP/private_allowed" || : >"$TMP/private_allowed"
  while IFS= read -r f; do
    # Reviewed files that may name you on purpose (for example LICENSE). Secrets are still checked.
    if [ -s "$TMP/private_allowed" ] && printf '%s\n' "$f" | grep -qiE -f "$TMP/private_allowed"; then continue; fi
    lines="$(content "$f" | grep -inIF -f "$TMP/private" | cut -d: -f1 | head -20 | tr '\n' ' ')"
    [ -n "$lines" ] && block "$f  line(s) $lines (matches an entry in your private-patterns file; values not shown)"
    printf '%s\n' "$f" | grep -qiF -f "$TMP/private" && block "$f  (file name matches an entry in your private-patterns file)"
  done <"$TMP/files"
elif [ -n "${CI:-}" ]; then
  : # CI has no private list by design; the local hooks cover it.
else
  warn "no private-patterns file at $PRIVATE_PATTERNS (run scripts/setup-dev-security.sh)"
fi

# ---------------------------------------------------------------- 5. gitleaks
step "gitleaks"
GL_ARGS=(--redact --no-banner --log-level error --exit-code 1)
[ -f "$ROOT/.gitleaks.toml" ] && GL_ARGS+=(--config "$ROOT/.gitleaks.toml")
[ -f "$ROOT/.gitleaksignore" ] && GL_ARGS+=(--gitleaks-ignore-path "$ROOT/.gitleaksignore")
if ! command -v gitleaks >/dev/null 2>&1; then
  block "gitleaks is not installed, so secrets cannot be checked. Install it: brew install gitleaks"
else
  case "$MODE" in
    staged) gitleaks git --pre-commit --staged "${GL_ARGS[@]}" -v "$ROOT" >&2 || block "gitleaks found a secret in the staged changes (redacted above)" ;;
    range) gitleaks git --log-opts="$GL_LOG_OPTS" "${GL_ARGS[@]}" -v "$ROOT" >&2 || block "gitleaks found a secret in the commits being pushed (redacted above)" ;;
    all)
      if git rev-parse --verify -q HEAD >/dev/null; then
        gitleaks git "${GL_ARGS[@]}" -v "$ROOT" >&2 || block "gitleaks found a secret in the Git history (redacted above)"
      fi
      # Working tree as `git add .` would see it (tracked and new files, ignored files excluded).
      mkdir -p "$TMP/tree"
      while IFS= read -r f; do
        [ -f "$f" ] || continue
        mkdir -p "$TMP/tree/$(dirname "$f")" && cp "$f" "$TMP/tree/$f"
      done <"$TMP/files"
      gitleaks dir "${GL_ARGS[@]}" -v "$TMP/tree" >&2 || block "gitleaks found a secret in the working tree (redacted above)"
      ;;
  esac
fi

# ---------------------------------------------------------------- reminders and result
if [ "$MODE" = staged ]; then
  images="$(grep -ciE '\.(png|jpe?g|gif|webp|heic|mp4|mov|webm)$' "$TMP/files" || true)"
  [ "${images:-0}" -gt 0 ] && warn "$images image/video file(s) staged: check that screenshots show demo data only (no real names, emails, bank or job data)."
fi

if [ "$FAIL" -ne 0 ]; then
  cat >&2 <<EOF
${RED}git-guard stopped this $( [ "$MODE" = range ] && echo push || echo commit ).${OFF}
  Remove the file from the commit:   git restore --staged <file>
  Keep it out for good:              add it to .gitignore
  Reviewed false positive:           add a narrow entry to .githooks/allowed-paths or .gitleaksignore
  Do not use --no-verify.
EOF
  exit 1
fi
printf '%sgit-guard: ok%s\n' "$GRN" "$OFF" >&2
exit 0
