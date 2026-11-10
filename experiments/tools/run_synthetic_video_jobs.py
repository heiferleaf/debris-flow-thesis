"""Plan by default; submit/poll/download require explicit --execute.

No automatic retry or long polling. See synthetic_runway.py for the execution
manifest and authorization JSON contracts. The offline preparation manifest is
readable for planning but remains blocked until its missing decisions are made.
For one negative effect preview, use manifest purpose=media_preview and its
preview-only reference checks; it does not require training partition/P controls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video.synthetic_runway import RunwayWorkflow, model_catalog


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "submit", "poll", "download"), nargs="?", default="plan")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--list-models", action="store_true", help="Show verified narrow model profiles and 10-second base prices offline")
    parser.add_argument("--state", type=Path, help="Persistent budget/task state; required for executed actions")
    parser.add_argument("--job-id", action="append", dest="job_ids", help="Repeat to explicitly select jobs")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Force offline metadata output even with --execute")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--max-cost-usd", help="Cumulative local-ledger generation base-cost cap, including unknown receipts")
    parser.add_argument("--out", type=Path, help="Media destination for explicit download")
    args = parser.parse_args(argv)
    if args.list_models:
        print(json.dumps(model_catalog(), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if not args.manifest:
        parser.error("--manifest is required unless --list-models is used")
    execute = args.execute and not args.dry_run
    if execute and args.action != "plan" and (not args.state or not args.job_ids):
        parser.error("executed actions require --state and at least one --job-id")
    if args.action == "download" and not args.out:
        parser.error("download requires --out")
    state = args.state or args.manifest.with_name("runway-task-state.json")
    try:
        workflow = RunwayWorkflow(args.manifest, state)
        if args.action == "plan":
            result = workflow.plan(args.job_ids)
        elif args.action == "submit":
            result = workflow.submit(args.job_ids, execute=execute, authorization_path=args.authorization, max_cost_usd=args.max_cost_usd)
        elif args.action == "poll":
            result = workflow.poll(args.job_ids, execute=execute)
        else:
            result = workflow.download(args.job_ids, args.out, execute=execute)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
