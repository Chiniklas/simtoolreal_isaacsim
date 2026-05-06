#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e .

python - <<'PY'
import simtoolreal_lab

print(f"simtoolreal_lab import ok: {simtoolreal_lab.__file__}")
PY
