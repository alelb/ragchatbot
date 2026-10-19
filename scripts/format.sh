#!/bin/bash
# Auto-fix code style: sort imports / apply safe lint fixes, then format with black
set -e
cd "$(dirname "$0")/.."

uv run ruff check --fix .
uv run black .
