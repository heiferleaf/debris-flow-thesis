#!/usr/bin/env bash
# Run only after fee authorization. Arguments are forwarded to run.py.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$task_root"
if [[ "$(id -u)" != "0" ]] || [[ ! -x /usr/bin/shutdown ]]; then
  echo "Automatic shutdown requires the AutoDL root environment; refusing GPU run." >&2
  exit 2
fi
# An exception, failed probe or timeout must not leave an idle paid GPU running.
trap '/bin/bash /usr/bin/shutdown' EXIT
task_python="${DF_PYTHON:-/root/miniconda3/bin/python}"
if [[ ! -x "$task_python" ]]; then
  echo "Configured Python is unavailable: $task_python" >&2
  exit 2
fi
mkdir -p data/results/exp016
task_log="data/results/exp016/cloud-$(date -u +%Y%m%dT%H%M%SZ).log"
"$task_python" -u experiments/runspecs/EXP-016/run.py --stage full "$@" 2>&1 | tee "$task_log"
