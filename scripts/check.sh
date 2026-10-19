#!/bin/bash
# Run every quality check without modifying files (exits non-zero on the first failure)
set -e
cd "$(dirname "$0")/.."

echo "==> black (formatting)"
uv run black --check .

echo "==> ruff (lint + import order)"
uv run ruff check .

echo "==> pytest"
uv run pytest "$@"

echo "All checks passed."
