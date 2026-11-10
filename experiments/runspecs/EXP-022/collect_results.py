"""Collect-only EXP022 transport wrapper: helper read waits up to 30 seconds.

The pinned operator still owns SSH, inventory, SHA/prefix checks, .part resume,
copy scheduling and the total deadline. No operator/helper source is edited.
Only the SFTP object's public get_channel view is proxied; Paramiko's internal
raw Channel and its nonblocking calls remain untouched. Root starts one writer.
"""
import argparse
import hashlib
import importlib.util
from importlib.machinery import SourceFileLoader
import json
import math
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[3]
OPERATOR = ROOT / '.codex/logs/evidence/exp022-linear-cv-s7-20261104-v1-operator-sources/exp022_cloud_ops.py.c8b6827d72fcaa03b85d33cabee3c9a308ffbbcdabf66943d44fc1adcadfa5a5.snapshot'
OPERATOR_SHA = 'c8b6827d72fcaa03b85d33cabee3c9a308ffbbcdabf66943d44fc1adcadfa5a5'
READ_WAIT_SECONDS = 30.


def verify_operator(path=OPERATOR):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != OPERATOR_SHA:
        raise ValueError('Frozen EXP022 operator SHA differs; collect wrapper refused')
    return payload


def load_operator():
    verify_operator()
    spec = importlib.util.spec_from_file_location('exp022_collect_only_frozen_operator', OPERATOR, loader=SourceFileLoader('exp022_collect_only_frozen_operator', str(OPERATOR)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HelperChannel:
    """View used by copy_verified, never installed into Paramiko internals."""
    def __init__(self, channel, end, clock=time.time):
        self._channel, self._end, self._clock = channel, end, clock

    def settimeout(self, value):
        remaining = self._end - self._clock()
        if remaining <= 0:
            raise TimeoutError('Collection total deadline reached; partial files preserved')
        # Preserve an explicit nonblocking request even on this public view.
        if value == 0:
            return self._channel.settimeout(0)
        if value is None or not math.isfinite(value) or value < 0 or value > 5:
            raise ValueError('Unexpected timeout request outside the pinned collector helper')
        return self._channel.settimeout(min(READ_WAIT_SECONDS, remaining))

    def __getattr__(self, name):
        return getattr(self._channel, name)


class HelperSFTP:
    def __init__(self, sftp, end, clock=time.time):
        self._sftp = sftp
        self._channel_view = HelperChannel(sftp.get_channel(), end, clock)

    def get_channel(self):
        return self._channel_view

    def __getattr__(self, name):
        # Forward bound methods to the original SFTP object; its own .sock and
        # Channel methods remain original, including internal settimeout(0).
        return getattr(self._sftp, name)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('collect',))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=60)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=25)
    parser.add_argument('--small-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if (not math.isfinite(args.max_seconds) or not 1 <= args.max_seconds <= 21600
            or not math.isfinite(args.poll_seconds) or not 20 <= args.poll_seconds <= 30):
        raise ValueError('Original bounded observation/poll limits must be retained')
    return args


def transport_proof(operator, args):
    """Separate wrapper evidence; original operator proof does not cover this."""
    source = Path(__file__).read_bytes()
    fingerprint = hashlib.sha256(source).hexdigest()
    directory = ROOT / '.codex/logs' / ('exp022-' + operator.RUN_ID + '-collection-transport')
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / ('collect_exp022_results.py.' + fingerprint + '.snapshot')
    if snapshot.exists():
        if snapshot.read_bytes() != source:
            raise ValueError('Transport wrapper snapshot conflict')
    else:
        with snapshot.open('xb') as stream:
            stream.write(source)
    proof = directory / (str(time.time_ns()) + '-' + uuid.uuid4().hex[:8] + '.jsonl')
    def record(value):
        with proof.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(observed_unix=time.time(), **value), allow_nan=False) + '\n')
    record(dict(status='collect_only_transport_wrapper_configured', action='collect', run_id=operator.RUN_ID,
        wrapper_source_sha256=fingerprint, frozen_operator_sha256=OPERATOR_SHA,
        frozen_collector_sha256=hashlib.sha256(Path(operator.COLLECT.__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        helper_positive_read_wait_cap_seconds=READ_WAIT_SECONDS, max_seconds=args.max_seconds,
        global_deadline='Inherited unchanged from the original operator; each wait is capped by its actual end.',
        scope='Additional transport proof; the old operator source proof does not cover this wrapper.',
        no_remote_training_launch_sync_or_shutdown=True, dry_run=args.dry_run))
    return record


def run_collect(operator, args, record):
    if args.action != 'collect':
        raise ValueError('This wrapper permits collect only')
    original_open, original_argv = operator.COLLECT.bounded_sftp, sys.argv
    def bounded_sftp(client, end):
        raw = original_open(client, end)
        record(dict(status='helper_channel_view_installed', actual_global_end_unix=end,
                    helper_read_wait_cap_seconds=READ_WAIT_SECONDS, paramiko_raw_channel_unchanged=True))
        return HelperSFTP(raw, end)
    forwarded = [str(OPERATOR), 'collect', '--config', str(args.config), '--max-seconds', str(args.max_seconds),
                 '--poll-seconds', str(args.poll_seconds)]
    for option, enabled in (('--once', args.once), ('--small-only', args.small_only), ('--dry-run', args.dry_run)):
        if enabled:
            forwarded.append(option)
    operator.COLLECT.bounded_sftp = bounded_sftp
    sys.argv = forwarded
    try:
        return operator.main()
    finally:
        operator.COLLECT.bounded_sftp = original_open
        sys.argv = original_argv


def main(argv=None):
    args = parse_args(argv)
    operator = load_operator()
    record = transport_proof(operator, args)
    try:
        result = run_collect(operator, args, record)
    except BaseException as error:
        record(dict(status='wrapper_exit_original_partial_preservation_retained', error_type=type(error).__name__))
        raise
    record(dict(status='collect_wrapper_returned', remote_control_action='none'))
    return result


if __name__ == '__main__':
    main()
