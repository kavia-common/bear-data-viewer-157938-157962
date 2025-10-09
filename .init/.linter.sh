#!/bin/bash
set -euo pipefail

cd /home/kavia/workspace/code-generation/bear-data-viewer-157938-157962/backend

# Ensure virtual environment exists
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

# Activate venv
# shellcheck disable=SC1091
source venv/bin/activate

# Ensure pip is up to date and required packages are installed
python -m pip install --upgrade pip >/dev/null 2>&1 || true
if [ -f "requirements.txt" ]; then
  python -m pip install -r requirements.txt >/dev/null
fi

# Run flake8
flake8 .
LINT_EXIT_CODE=$?

if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi
