#!/bin/bash
# The rental owner supplies the actual fee deadline and verified platform timer.
set -euo pipefail
cd "$(dirname "$0")/../../.."
task_out=''
task_previous=''
for task_argument in "$@"; do
    if [[ "$task_previous" == '--out' ]]; then task_out="$task_argument"; fi
    task_previous="$task_argument"
done
finish() {
    task_status=$?
    trap - EXIT
    # Evidence is secondary: neither a missing output nor a full disk may stop
    # the bounded shutdown request. Preserve the original failure status.
    if /usr/bin/timeout --kill-after=2s 15s /bin/bash /usr/bin/shutdown; then
        task_shutdown_status=0
    else
        task_shutdown_status=$?
        if [[ "$task_status" == 0 ]]; then task_status=125; fi
    fi
    printf 'EXP020 shutdown command returncode=%s; final exit=%s\n' "$task_shutdown_status" "$task_status" >&2 || true
    if [[ -n "$task_out" && -d "$task_out" ]]; then
        (set -o noclobber; printf '{"shutdown_command_returncode":%s,"final_exit_code":%s,"actual_instance_state_requires_operator_confirmation":true}\n' \
            "$task_shutdown_status" "$task_status" > "$task_out/shutdown-receipt.json") 2>/dev/null || true
    fi
    exit "$task_status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
/root/miniconda3/bin/python -B experiments/runspecs/EXP-020/pipeline.py "$@"
