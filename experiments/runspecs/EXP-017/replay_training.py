"""D0: one bounded CPU eval of the selected-E training exposures; no fitting."""
import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import struct
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = Path('data/results/exp020/head-s7s8-20261103-v2')
INDEX = 'data/process/EXP-017/cv-inputs-v1/verified-cache-files.csv'
INDEX_SHA = 'cc2448fbec65ecab6f20393c1c0049a82d50bcd3039bedd5cb81541b19b6f8ff'
CLOUD = '/root/autodl-tmp/debris-flow-thesis-exp016/'
PLAN = '.codex/docs/research-plans/2026-11-03-EXP021后-负监督包目标路线.md'
PLAN_SHA = 'f173b8cb5b64d87352ac74efff3176740cd34630bd2e1007ebee2fa3ca3f359b'
SPECS = [dict(fold=1, E=10, P=288, N=120,
              closure='70072ca03e94ea0df0bf980a9513714d4a60f1f8d0858e575bf4dc60d494f094',
              checkpoint='b2265c6e3dac6ff04534050d1f55a3ce244bee50c11b951ae1a083dccb838ba8',
              state='7eb913daa8c28b358386337dde4caafa14f23bc509428ec018ed13886926d0fa'),
         dict(fold=4, E=6, P=292, N=120,
              closure='31d9924c4950ad5e2f966d56bbe0114c5a8af8f7aa5dabb3eae75fef7900e04b',
              checkpoint='151c7276723bf0c58cefdf82b9bedeee9732b084238b914f981346a487f37ca9',
              state='3a3df043ab992b08ca28701177ca7fc77f78aadf1bacf7633d1063ddbc4dcdce')]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def physical(row):
    indices, pts = row['frame_indices'], row['source_pts_seconds']
    require(len(indices) == len(pts) == len(row['frames']) == 16, 'Expected 16 real frames')
    require(all(type(i) is int and 0 <= i < 2 ** 31 for i in indices), 'Invalid indices')
    require(all(a < b for a, b in zip(indices, indices[1:])), 'Repeated/unordered indices')
    require(all(math.isfinite(t) for t in pts) and all(a < b for a, b in zip(pts, pts[1:])), 'Invalid PTS')
    return row['unit_id'], tuple(indices)


def select_epoch(inputs, history, epoch, P, N):
    """Only selected train IDs and fixed-val identities are looked up in the bank."""
    require(len(inputs['epoch_clip_ids']) == 20 and history['completed_epochs'] == epoch,
            'Wrong frozen epoch')
    ids = history['train_clip_ids']
    require(ids == inputs['epoch_clip_ids'][epoch - 1] and len(set(ids)) == len(ids),
            'Actual selected-E exposure order differs from inputs')
    val_ids = inputs['validation_clip_ids']
    require(not set(ids).intersection(val_ids), 'Fixed validation clip entered probe')
    bank = inputs['clip_bank']
    val_keys = {physical(bank[cid]) for cid in val_ids}
    rows, keys, quotas, groups = [], set(), {}, {}
    for ordinal, cid in enumerate(ids):
        row = bank[cid]
        key = physical(row)
        require(key not in keys and key not in val_keys, 'Duplicate physical window or validation overlap')
        require(type(row['label']) is int and row['label'] in (0, 1), 'Invalid training label')
        require(row['unit_id'] and row['group_id'], 'Missing source/group identity')
        require(groups.setdefault(row['unit_id'], row['group_id']) == row['group_id'], 'Conflicting video group')
        keys.add(key)
        bucket = (row['unit_id'], row['label'])
        quotas[bucket] = quotas.get(bucket, 0) + 1
        require(quotas[bucket] <= 4, 'Video/class shared quota exceeds four')
        rows.append(dict(ordinal=ordinal, clip_id=cid, unit_id=key[0], group_id=row['group_id'],
                         label=row['label'], frame_indices=row['frame_indices'],
                         source_pts_seconds=row['source_pts_seconds'], frames=row['frames'],
                         frame_tuple_sha256=hashlib.sha256(struct.pack('<16i', *key[1])).hexdigest()))
    require(sum(r['label'] for r in rows) == P and len(rows) == P + N, 'Selected-E P/N counts changed')
    require((history['samples'], history['train_P'], history['train_N']) == (P + N, P, N), 'History counts changed')
    return rows


def summarize(rows):
    """Bags are role-local (video, negative supervision), including positive-video negative clips."""
    require(rows and all(math.isfinite(r['logit']) and math.isfinite(r['score'])
                        and 0 <= r['score'] <= 1 for r in rows), 'Nonfinite/invalid saved predictions')
    bags, groups = {}, {}
    TP = FP = FN = TN = 0
    for row in rows:
        label, pred = row['label'], row['score'] >= .5
        TP += int(label == 1 and pred)
        FN += int(label == 1 and not pred)
        FP += int(label == 0 and pred)
        TN += int(label == 0 and not pred)
        group = groups.setdefault(row['group_id'], dict(P=0, N=0, TP=0, FP=0, FN=0, TN=0))
        group['P' if label else 'N'] += 1
        group[('TP' if pred else 'FN') if label else ('FP' if pred else 'TN')] += 1
        if label == 0:
            bag = bags.setdefault(row['unit_id'], dict(unit_id=row['unit_id'], group_id=row['group_id'], scores=[], clip_ids=[]))
            bag['scores'].append(row['score'])
            bag['clip_ids'].append(row['clip_id'])
    result = []
    for uid, bag in sorted(bags.items()):
        scores = bag.pop('scores')
        require(1 <= len(scores) <= 4, 'Invalid negative supervision bag size')
        result.append(dict(bag, label=0, windows=len(scores), max_score=max(scores),
                           mean_score=sum(scores) / len(scores), positive_windows=sum(s >= .5 for s in scores),
                           bag_false_positive=max(scores) >= .5))
    require(TP + FN > 0 and FP + TN > 0 and result, 'Probe must contain both supervision classes')
    return dict(TP=TP, FP=FP, FN=FN, TN=TN, positive_Recall=TP / (TP + FN),
                negative_Recall=TN / (FP + TN), negative_bags=result,
                negative_bag_count=len(result), false_positive_negative_bags=sum(b['bag_false_positive'] for b in result),
                any_negative_bag_false_positive=any(b['bag_false_positive'] for b in result), groups=groups)


def load_role(spec, identities):
    stage = 'f%02d-linear-s7' % spec['fold']
    directory = ROOT / BASE / stage
    closure_path = directory.parent / ('closed-' + stage + '.json')
    require(sha(closure_path) == spec['closure'], 'Frozen fit closure changed')
    closure = json.loads(closure_path.read_text(encoding='utf-8'))
    require(closure['status'] == 'fit_closed' and closure['item'] == stage, 'Fit not closed')
    files = {r['path']: r for r in closure['files']}
    loaded = {}
    identities[str(closure_path.relative_to(ROOT))] = spec['closure']
    for name in ['profile.json', 'inputs.json', 'epochs.jsonl', 'training-result.json']:
        path = directory / name
        ref = files[stage + '/' + name]
        require(path.stat().st_size == ref['size_bytes'] and sha(path) == ref['sha256'], 'Changed fit metadata: ' + name)
        identities[str(path.relative_to(ROOT))] = ref['sha256']
        if name == 'epochs.jsonl':
            with path.open(encoding='utf-8') as stream:
                for number, line in enumerate(stream, 1):
                    if number == spec['E']:
                        loaded[name] = json.loads(line)
                        break
        else:
            loaded[name] = json.loads(path.read_text(encoding='utf-8'))
    profile, result, history = (loaded[n] for n in ['profile.json', 'training-result.json', 'epochs.jsonl'])
    require((profile['fold'], profile['seed'], profile['head'], profile['mode']) ==
            (spec['fold'], 7, 'linear', 'rgb_zero'), 'Wrong fit role/model')
    require(result['status'] == 'fit_complete' and result['completed_epochs'] == 20 and
            result['selected_epochs'] == spec['E'], 'Wrong selected-E result')
    require(history['profile_sha256'] == result['profile_sha256'] == identities[str((directory / 'profile.json').relative_to(ROOT))], 'Profile identity mismatch')
    require(history['model_state_sha256'] == spec['state'], 'Selected history state mismatch')
    checkpoint = files[stage + '/best.pt']
    require(checkpoint['sha256'] == result['selected_checkpoint']['sha256'] == spec['checkpoint'] and
            result['selected_checkpoint']['completed_epochs'] == spec['E'] and
            (directory / 'best.pt').stat().st_size == checkpoint['size_bytes'], 'Selected checkpoint provenance mismatch')
    require(profile['input_fingerprints'][INDEX] == INDEX_SHA, 'Wrong accepted cache index')
    for absolute, expected in history['code_sha256'].items():
        require(absolute.startswith(CLOUD), 'Unmapped original source path')
        relative = absolute[len(CLOUD):]
        require(sha(ROOT / relative) == expected, 'Reused scientific source changed: ' + relative)
        identities[relative] = expected
    rows = select_epoch(loaded['inputs.json'], history, spec['E'], spec['P'], spec['N'])
    return dict(spec, stage=stage, directory=directory, profile=profile, rows=rows,
                checkpoint_provenance=checkpoint, original_profile_sha256=result['profile_sha256'])


def index_lines(stream, digest):
    for number, line in enumerate(stream):
        digest.update(line)
        yield line.decode('utf-8-sig' if number == 0 else 'utf-8')


def frame_index(roles):
    wanted = {}
    for role in roles:
        for row in role['rows']:
            for index, pts, frame in zip(row['frame_indices'], row['source_pts_seconds'], row['frames']):
                key = (row['unit_id'], index)
                value = (pts, frame)
                require(key not in wanted or wanted[key] == value, 'Conflicting frozen frame identity')
                wanted[key] = value
    found, digest = {}, hashlib.sha256()
    with (ROOT / INDEX).open('rb') as stream:
        for record in csv.DictReader(index_lines(stream, digest)):
            key = (record['unit_id'], int(record['frame_index']))
            if key not in wanted:
                continue
            require(key not in found, 'Duplicate required cache identity')
            pts, frame = wanted[key]
            relative = PurePosixPath(record['cache_path'])
            require(not relative.is_absolute() and '..' not in relative.parts and '\\' not in record['cache_path'], 'Unsafe cache path')
            path = (ROOT / str(relative)).resolve()
            path.relative_to(ROOT)
            require(frame['path'] == CLOUD + record['cache_path'] and
                    frame['jpeg_sha256'] == record['jpeg_sha256'] and
                    frame['rgb112_sha256'] == record['rgb112_sha256'] and
                    abs(pts - float(record['source_pts_seconds'])) <= 1e-8, 'Frozen frame/index mismatch')
            require(path.is_file() and path.stat().st_size == int(record['size_bytes']), 'Missing required JPEG')
            found[key] = dict(path=str(path), jpeg_sha256=record['jpeg_sha256'], rgb112_sha256=record['rgb112_sha256'])
    require(digest.hexdigest() == INDEX_SHA and set(found) == set(wanted), 'Cache index SHA or frame coverage mismatch')
    return found


def worker(out, microbatch):
    import numpy as np
    import torch
    import cv2
    from experiments.common.video.head_comparison_training import LinearVideoClassifier
    from experiments.common.video.r2plus1d_classifier import state_digest
    from experiments.common.video.supervised_runtime import ClipDataset, FrameRef, ClipInput, score_clips
    began = time.perf_counter()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    identities = {str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__)}
    require(sha(ROOT / PLAN) == PLAN_SHA, 'D0 specification changed')
    identities[PLAN] = PLAN_SHA
    roles = [load_role(spec, identities) for spec in SPECS]
    frames = frame_index(roles)
    identities[INDEX] = INDEX_SHA
    for role in roles:
        for row in role['rows']:
            row['local_frames'] = [frames[(row['unit_id'], i)] for i in row['frame_indices']]
    write_json(out / 'selected-exposures.json', dict(scope='selected E only, not cumulative',
        roles=[dict(fold=r['fold'], selected_E=r['E'], rows=r['rows']) for r in roles]))
    write_json(out / 'provenance.json', dict(input_sha256=identities, threshold=.5, device='cpu', precision='FP32',
        threads=4, microbatch=microbatch, checkpoint_byte_SHA_scope='Inherited exact archived closure/collector acceptance; no new full checkpoint byte hash',
        state_SHA_scope='Actual loaded model digest must match selected-E history; final_state (E20) is never used',
        environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__, platform=platform.platform())))
    results = []
    for role in roles:
        start = time.perf_counter()
        # Load only one full archive; immediately release the saved optimizer.
        checkpoint = torch.load(role['directory'] / 'best.pt', map_location='cpu', weights_only=False)
        state = checkpoint['state_dict']
        require(checkpoint['completed_epochs'] == role['E'] and checkpoint['state_sha256'] == role['state'] and
                checkpoint['profile'] == role['profile'], 'Loaded best checkpoint identity mismatch')
        del checkpoint
        gc.collect()
        net = LinearVideoClassifier(pretrained=False).cpu().float().eval()
        net.load_state_dict(state, strict=True)
        del state
        net.requires_grad_(False)
        require(state_digest(net) == role['state'], 'Actual loaded selected model state mismatch')
        transform = ClipInput('rgb_zero').cpu().float().eval()
        ds = ClipDataset([(r['clip_id'], tuple(FrameRef(**frames[(r['unit_id'], i)]) for i in r['frame_indices']),
                           r['label']) for r in role['rows']], cache_bytes=32 * 1024 ** 2)
        predictions = []
        fields = ['ordinal', 'fold', 'selected_E', 'clip_id', 'unit_id', 'group_id', 'label', 'frame_tuple_sha256', 'logit', 'score']
        with (out / (role['stage'] + '-scores.csv')).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for offset in range(0, len(ds), microbatch):
                pixels = np.stack([ds[i][0] for i in range(offset, min(offset + microbatch, len(ds)))])
                with torch.inference_mode():
                    logits = score_clips(net, torch.from_numpy(pixels), transform, microbatch=microbatch)
                    scores = torch.sigmoid(logits.float())
                require(logits.dtype == scores.dtype == torch.float32, 'Expected FP32 predictions')
                for local, (logit, score) in enumerate(zip(logits.tolist(), scores.tolist())):
                    record = {key: role['rows'][offset + local][key] for key in fields[:1] + fields[3:-2]}
                    record.update(fold=role['fold'], selected_E=role['E'], logit=logit, score=score)
                    require(math.isfinite(logit) and math.isfinite(score), 'Nonfinite CPU prediction')
                    writer.writerow(record)
                    predictions.append(record)
                stream.flush()
                if offset % 32 == 0:
                    print('%s %d/%d' % (role['stage'], len(predictions), len(ds)), flush=True)
        require(state_digest(net) == role['state'] and not net.training and
                not any(p.requires_grad for p in net.parameters()), 'Read-only eval mutated model state')
        result = dict(fold=role['fold'], selected_E=role['E'], windows=len(predictions), seconds=time.perf_counter() - start,
                      model_state_sha256=role['state'], inherited_checkpoint_sha256=role['checkpoint'], **summarize(predictions))
        write_json(out / (role['stage'] + '-result.json'), result)
        results.append(result)
        del net, transform, ds, predictions
        gc.collect()
    require(sum(r['windows'] for r in results) == 820, 'Incomplete D0 exposure')
    require(all(sha(ROOT / path) == fingerprint for path, fingerprint in identities.items() if path != INDEX), 'Probe source or small metadata changed')
    decision = ('candidate_eligible_for_design_review' if any(r['any_negative_bag_false_positive'] for r in results)
                else 'no_direct_support_from_this_probe')
    write_json(out / 'result.json', dict(status='D0_selected_training_exposure_probe_complete', decision=decision,
        original_run_id='head-s7s8-20261103-v2', windows=820, roles=results, seconds=time.perf_counter() - began,
        scope='Only these two selected-E training epochs; not all training candidates, dense, independent test, or evidence of temporal utilization',
        action='No automatic training/promotion/threshold search; design review only',
        output_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'data/results/exp020/linear-training-replay-s7-selected-E-v1')
    parser.add_argument('--microbatch', type=int, default=4, choices=range(1, 9))
    parser.add_argument('--max-seconds', type=int, default=900)
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.out.resolve()
    require(0 < args.max_seconds <= 900, 'D0 hard wall limit is at most 900 seconds')
    require(ROOT / 'data/results/exp020' in out.parents and ROOT / BASE not in out.parents and
            out != ROOT / BASE, 'Output must be a new derived EXP020 directory outside original run')
    if args._worker:
        try:
            worker(out, args.microbatch)
            return 0
        except Exception as error:
            write_json(out / 'worker-failure.json', dict(status='failed_or_partial_not_complete',
                       error=repr(error), traceback=traceback.format_exc()))
            raise
    require(not out.exists(), 'Refusing to overwrite output')
    out.mkdir(parents=True, exist_ok=False)
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write_json(out / 'command.json', dict(argv=sys.argv, max_seconds=args.max_seconds, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    try:
        process = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--_worker',
            '--out', str(out), '--microbatch', str(args.microbatch), '--max-seconds', str(args.max_seconds)],
            cwd=str(ROOT), timeout=args.max_seconds, check=False)
        require(process.returncode == 0 and (out / 'result.json').is_file(), 'Probe child failed (exit %s)' % process.returncode)
        print('D0 complete: ' + str(out), flush=True)
        return 0
    except Exception as error:
        write_json(out / 'failure.json', dict(status='failed_or_partial_not_complete', error=repr(error),
                   no_automatic_retry_or_GPU_fallback=True, completed_role_files=sorted(p.name for p in out.glob('f*-result.json'))))
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
