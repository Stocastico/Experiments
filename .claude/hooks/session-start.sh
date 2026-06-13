#!/bin/bash
# Photo Atlas — SessionStart hook for Claude Code on the web.
# Installs the project (with dev/test deps) so tests and tooling work.
set -euo pipefail

# Only run in the remote (web) environment; local setups manage their own venv.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Editable install pulls in runtime deps (Pillow, numpy, scikit-learn,
# opencv-python-headless, FastAPI, uvicorn, pydantic) plus the [dev] extra
# (pytest, httpx). `pip install -e` is idempotent and benefits from the
# container's cached state on subsequent runs.
python3 -m pip install --quiet --root-user-action=ignore -e '.[dev]'

# pytest already sets pythonpath=["src"], but export it too so ad-hoc
# `python -m ...` invocations resolve the package without reinstalling.
echo 'export PYTHONPATH="${CLAUDE_PROJECT_DIR}/src:${PYTHONPATH:-}"' >> "$CLAUDE_ENV_FILE"

echo "Photo Atlas environment ready."
