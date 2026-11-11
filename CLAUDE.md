@AGENTS.md

Claude Code also enforces parts of these rules in `.claude/settings.json`: committing, pushing,
GitHub changes, and edits to the security configuration ask for approval; force-pushes, skipping
hooks, and reading `.env` files or the private backups are denied.
