#!/usr/bin/env bash
# Reuse the authorized instance; always stop it on completion or failure.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$task_root"
if [[ "$(id -u)" != "0" ]] || [[ ! -x /usr/bin/shutdown ]]; then
  echo "AutoDL root shutdown facility required." >&2
  exit 2
fi
trap '/bin/bash /usr/bin/shutdown' EXIT
task_python="${DF_PYTHON:-/root/miniconda3/bin/python}"
if [[ ! -x "$task_python" ]]; then exit 2; fi
mkdir -p data/results/exp018
task_log="data/results/exp018/cloud-$(date -u +%Y%m%dT%H%M%SZ).log"
"$task_python" -u experiments/runspecs/EXP-018/run.py --stage full "$@" 2>&1 | tee "$task_log"
