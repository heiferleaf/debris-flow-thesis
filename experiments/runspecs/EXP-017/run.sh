#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$task_root"
if [[ "$(id -u)" != "0" ]] || [[ ! -x /usr/bin/shutdown ]]; then
  echo "Verified AutoDL root shutdown facility required." >&2
  exit 2
fi
task_shutdown() {
  task_status=$?
  trap - EXIT
  /bin/bash /usr/bin/shutdown || {
    echo "Shutdown command failed; verify independent platform timer and instance state." >&2
    if [[ "$task_status" == "0" ]]; then task_status=125; fi
  }
  exit "$task_status"
}
trap task_shutdown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
task_python="${DF_PYTHON:-/root/miniconda3/bin/python}"
[[ -x "$task_python" ]]
mkdir -p data/results/exp017
task_log="data/results/exp017/cloud-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
"$task_python" -u experiments/runspecs/EXP-017/pipeline.py --stage full "$@" 2>&1 | tee "$task_log"
