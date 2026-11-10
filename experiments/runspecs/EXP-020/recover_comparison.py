"""Offline correction of the pinned EXP020 comparison-row construction only.

Requires all six completed/collected fits and the original TypeError/failure
closure. Does not edit the run, model code, plan, metrics or checkpoints. The
original comparison retains all calculations/assertions and streams best-file
SHA; no checkpoint is deserialized and no model is constructed or executed.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'experiments/runspecs/EXP-020/plan.py'
SOURCE_SHA = '42edec46af2471b76af268db14ed13d0093edd8560eba8d4f9c82f430d0d5673'
DEFAULT_RUN = ROOT / 'data/results/exp020/head-s7s8-20261103-v2'
DEFAULT_PLAN = ROOT / 'data/metadata/EXP-020/head-plan-v2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(value, message):
    if not value:
        raise ValueError(message)


def corrected_source(source):
    """Replace one contiguous row-assembly block, not scientific calculations."""
    require(hashlib.sha256(source.encode('utf-8')).hexdigest() == SOURCE_SHA, 'Frozen module source differs')
    begin = '        rows.append(dict(**item, actual_artifact_directory=str(path), selected_epochs=selected,'
    end = '            gate_pass=auc_logit >= .75 and positive >= .5 and negative >= .5))'
    require(source.count(begin) == source.count(end) == 1, 'Unique frozen row block required')
    start, stop = source.index(begin), source.index(end) + len(end)
    old = source[start:stop]
    new = old.replace(begin,
        "        if item['action'] == 'reuse' and item['selected_epochs'] != selected:\n"
        "            raise ValueError('Reused selected E differs from earliest raw BCE minimum')\n"
        "        row = dict(item)\n"
        '        row.update(actual_artifact_directory=str(path), selected_epochs=selected,', 1)
    new = new.replace(end,
        '            gate_pass=auc_logit >= .75 and positive >= .5 and negative >= .5)\n'
        '        rows.append(row)', 1)
    return source[:start] + new + source[stop:]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    p.add_argument('--out', type=Path, default=ROOT / 'data/results/exp020/recovered-comparison-head-s7s8-20261103-v2-v1')
    a = p.parse_args(argv)
    run, directory, out = a.run.resolve(), a.plan.resolve(), a.out.resolve()
    run.relative_to(ROOT); directory.relative_to(ROOT)
    require(not out.exists(), 'Recovery output exists; never overwrite')
    require(out != run and run not in out.parents, 'Recovery must be outside original run')
    require(sha(SOURCE) == SOURCE_SHA, 'Pinned original comparison source changed')
    source = SOURCE.read_text(encoding='utf-8')
    corrected = corrected_source(source)
    failure, closed, execution = [read(run / n) for n in ('pipeline-failure.json', 'closed-run.json', 'execution-inputs.json')]
    require(failure['error_type'] == 'TypeError' and 'selected_epochs' in failure['error'] and 'multiple values' in failure['error'],
            'Only the identified aggregation TypeError is recoverable here')
    require(closed['status'] == 'run_failed_closed' and not (run / 'hard-time-limit.json').exists(),
            'Require original failed-but-closed run, never active/hard-stop output')
    require(execution['plan_sha256'] == sha(directory / 'plan.json'), 'Original execution/plan differs')
    module = types.ModuleType('_exp020_offline_comparison_correction')
    module.__file__ = str(SOURCE)
    # This explicit local derived module is never installed into sys.modules or
    # used by training. Original source bytes remain untouched and hash-pinned.
    exec(compile(corrected, str(SOURCE) + '[offline-row-correction]', 'exec'), module.__dict__)
    plan, schedules, bank, _, _ = module.load(ROOT, directory, with_frames=False)
    require(plan['code_sha256']['experiments/runspecs/EXP-020/plan.py'] == SOURCE_SHA, 'Plan must bind original aggregation source')
    consumed = {}
    def record(path):
        path = Path(path).resolve(); name = path.relative_to(ROOT).as_posix()
        consumed[name] = sha(path)
    for path in (SOURCE, directory / 'plan.json', directory / 'schedules.json', run / 'execution-inputs.json',
                 run / 'pipeline-failure.json', run / 'closed-run.json'):
        record(path)
    new_count = 0
    for item in plan['items']:
        path = ROOT / item['artifact_directory'] if item['action'] == 'reuse' else run / item['item']
        if item['action'] == 'fit':
            new_count += 1
            receipt_path = run / ('closed-' + item['item'] + '.json')
            receipt = read(receipt_path)
            require(receipt['status'] == 'fit_closed' and receipt['item'] == item['item'], 'All six individual fits must be closed')
            listed = {r['path']: r for r in receipt['files']}
            require(len(listed) == len(receipt['files']), 'Duplicate closed artifact')
            for name in ('training-result.json', 'profile.json', 'inputs.json', 'epochs.jsonl', 'best.pt', 'last.pt'):
                key = item['item'] + '/' + name
                require(key in listed and (path / name).is_file() and (path / name).stat().st_size == listed[key]['size_bytes'],
                        'Incomplete collected fit artifact: ' + key)
                if not name.endswith('.pt'):
                    require(sha(path / name) == listed[key]['sha256'], 'Closed small artifact bytes differ')
            result = read(path / 'training-result.json')
            require(result['status'] == 'fit_complete' and result['stage'] == 'inner_select' and result['fold'] == item['fold']
                    and result['completed_epochs'] == 20 and result['optimizer_steps'] == (340 if item['fold'] == 1 else 360),
                    'Incomplete or wrong-role new fit')
            for epoch in range(1, 21):
                name = 'validation-epoch-%02d.json' % epoch
                key = item['item'] + '/' + name
                require(key in listed and sha(path / name) == listed[key]['sha256'], 'Closed raw validation bytes differ')
            record(receipt_path)
        for name in ('training-result.json', 'profile.json', 'inputs.json', 'epochs.jsonl'):
            record(path / name)
        for epoch in range(1, 21):
            record(path / ('validation-epoch-%02d.json' % epoch))
    require(new_count == 6, 'Require exactly six new fits')
    answer = module.comparison(plan, run, ROOT, bank=bank, schedules=schedules)
    require(len(answer['items']) == 8 and len(answer['pairs']) == 4, 'Incomplete derived matrix')
    require(all(sha(ROOT / name) == expected for name, expected in consumed.items()), 'Source changed during recovery')
    answer['derivation'] = dict(kind='offline_recovery_comparison_only', status='aggregation_recovered_offline_pending_independent_review',
        original_run_path=run.relative_to(ROOT).as_posix(), plan_sha256=sha(directory / 'plan.json'),
        original_failure_sha256=sha(run / 'pipeline-failure.json'), original_closed_run_status=closed['status'],
        original_closed_run_sha256=sha(run / 'closed-run.json'), source_sha256=consumed,
        frozen_comparison_source_sha256=SOURCE_SHA,
        corrected_in_memory_source_sha256=hashlib.sha256(corrected.encode()).hexdigest(),
        recovery_script_sha256=sha(Path(__file__).resolve()),
        inherited_accepted_input_sha256=plan['source_fingerprints'],
        no_frozen_source_modified=True, no_run_modified=True, no_refit=True, no_forward=True,
        no_checkpoint_deserialization=True, selected_best_bytes_stream_SHA_retained=True,
        last_checkpoint_identity_scope='Existing collected closure; this tool checks size, not an additional last-file SHA pass',
        argv=sys.argv if argv is None else argv)
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'comparison.json').open('x', encoding='utf-8') as stream:
        json.dump(answer, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    patch = ''.join(difflib.unified_diff(source.splitlines(True), corrected.splitlines(True),
        fromfile='frozen_EXP020_plan.py', tofile='offline_row_assembly_correction'))
    (out / 'aggregation-only.patch').write_text(patch, encoding='utf-8')
    print(json.dumps(dict(status='aggregation_recovered_offline_pending_independent_review',
        comparison_path=str(out / 'comparison.json'), sha256=sha(out / 'comparison.json'))))


if __name__ == '__main__':
    main()
