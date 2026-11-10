#!/usr/bin/env bash
# CPU-only check on AutoDL; upload time is separately bounded by platform timer.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$task_root"
if [[ "$(id -u)" != "0" ]] || [[ ! -x /usr/bin/shutdown ]] || ! command -v timeout >/dev/null; then
  echo "AutoDL root/shutdown/timeout required; refusing unattended CPU check." >&2
  exit 2
fi
trap '/bin/bash /usr/bin/shutdown' EXIT
task_python="${DF_PYTHON:-/root/miniconda3/bin/python}"
if [[ ! -x "$task_python" ]]; then
  echo "Configured Python is unavailable: $task_python" >&2
  exit 2
fi
mkdir -p data/process/EXP-016
task_log="data/process/EXP-016/cloud-check-$(date -u +%Y%m%dT%H%M%SZ).log"
timeout --signal=TERM --kill-after=30s 5h \
  "$task_python" -u experiments/runspecs/EXP-016/run.py --stage check "$@" 2>&1 | tee "$task_log"
# Allow the connected client to retrieve the success report before SSH closes.
# Failures still exit immediately through the trap.
sleep 10
