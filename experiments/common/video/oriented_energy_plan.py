"""EXP021 small inherited-input plan and explicit local/cloud A-root mapping."""
from collections import Counter
import csv
import json
import math
from pathlib import Path

from experiments.common.video.cv_supervised_plan import load_cv_plan
from experiments.common.video.cv_supervised_training import _json, _sha
from experiments.common.video.oriented_energy import CARD_SHA, VERSION, read_artifact
from experiments.common.video.oriented_energy_model import RGB_INITIAL_SHA, BACKBONE_INITIAL_SHA
from experiments.common.video.supervised_runtime import FrameRef, PIXEL_VERSION

CARD = 'docs/process/experiments/EXP-021.md'
BASE20 = 'data/metadata/EXP-020/head-plan-v2'
BASE17 = 'data/metadata/EXP-017/role-free-plan-v3'
REFERENCE_LOCAL = 'data/results/exp020/head-s7s8-20261103-v2'
REFERENCE_CLOUD = '/root/exp020-results/head-s7s8-20261103-v2'
EVOLUTION = 'data/metadata/EXP-021/protocol-evolution-v1.json'
KERNEL = 'data/metadata/EXP-021/kernels-v1/kernels.npz'
DENSE = 'data/process/EXP-017/dense-pixel-reference-v1/result.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rooted(root, name):
    path = (root / name).resolve()
    path.relative_to(root)
    if Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Relative rooted metadata identity required')
    return path


def new_sources(root):
    return sorted([p.relative_to(root).as_posix() for p in (root / 'experiments/common/video').glob('oriented_energy*.py')]
        + [p.relative_to(root).as_posix() for p in (root / 'experiments/runspecs/EXP-021').iterdir() if p.is_file()])


def epoch_block(path):
    s = Path(path).read_text(encoding='utf-8')
    return s[s.index('            for epoch, order in enumerate(schedule[:epochs], 1):'):
             s.index('        if completed != epochs or total_updates != planned_steps:')]


def prepare(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.relative_to(root / 'data/metadata/EXP-021')
    if out.exists():
        raise FileExistsError(out)
    if _sha(root / CARD) != CARD_SHA:
        raise ValueError('Approved EXP021 card changed')
    original = read(root / BASE20 / 'plan.json')
    evolution = read(root / EVOLUTION)
    changes = {r['logical_path']: r for r in evolution['documents']}
    fingerprints = {CARD: CARD_SHA, EVOLUTION: _sha(root / EVOLUTION), BASE20 + '/plan.json': _sha(root / BASE20 / 'plan.json')}
    inherited_refs = {}
    for name, sha in original['source_fingerprints'].items():
        if name.startswith('data/results/exp017/') and 'audit-resume-' not in name:
            inherited_refs[name] = sha  # Old deep-head references are not consumed.
            continue
        if name in changes:
            change = changes[name]
            if sha != change['historical_sha256'] or _sha(root / name) != change['current_sha256']:
                raise ValueError('Unregistered current governance evolution')
            for p in change['historical_snapshots']:
                if _sha(root / p) != sha:
                    raise ValueError('Historical governance byte snapshot differs')
            fingerprints[name] = change['current_sha256']
        else:
            if _sha(rooted(root, name)) != sha:
                raise ValueError('Inherited accepted scientific input/source changed: ' + name)
            fingerprints[name] = sha
    for name in (KERNEL, str(Path(KERNEL).with_name('manifest.json')).replace('\\', '/'), DENSE):
        fingerprints[name] = _sha(root / name)
    kernel_manifest = read(root / Path(KERNEL).with_name('manifest.json'))
    if kernel_manifest['artifact_sha256'] != fingerprints[KERNEL] or kernel_manifest['version'] != VERSION:
        raise ValueError('Fixed operator artifact/manifest differs')
    read_artifact(root / KERNEL, fingerprints[KERNEL])
    if epoch_block(root / 'experiments/common/video/oriented_energy_training.py') != epoch_block(root / 'experiments/common/video/cv_supervised_training.py'):
        raise ValueError('Inherited ordinary BCE epoch execution changed')
    for name in new_sources(root):
        fingerprints[name] = _sha(root / name)
    original_execution = 'data/results/exp017/cv-s7-20261102-v1/execution-inputs.json'
    fingerprints[original_execution] = _sha(root / original_execution)
    ffmpeg_sha = read(root / original_execution)['ffmpeg_sha256']
    _, bank, _, _, folds = load_cv_plan(root, root / BASE17, verify_sources=False)
    schedules = read(root / BASE20 / 'schedules.json')
    if _sha(root / BASE20 / 'schedules.json') != original['schedules_sha256']:
        raise ValueError('EXP020 frozen schedule changed')
    fingerprints[BASE20 + '/schedules.json'] = _sha(root / BASE20 / 'schedules.json')
    roles, references, fit_order = {}, {}, []
    for fold, selected, count in ((1, 10, (288, 120)), (4, 6, (292, 120))):
        f = folds[fold - 1]
        if schedules[str(fold)] != dict(epoch_clip_ids=f['inner_train']['epoch_clip_ids'], validation_clip_ids=f['inner_val']['clip_ids']):
            raise ValueError('Inner role/schedule identity differs from original')
        for order in schedules[str(fold)]['epoch_clip_ids']:
            actual = Counter(bank[c]['label'] for c in order)
            if (actual[1], actual[0]) != count:
                raise ValueError('Per-epoch P/N differs')
        roles[str(fold)] = dict(video_ids=f['inner_val']['video_ids'], groups=f['inner_val']['group_ids'])
        if len(roles[str(fold)]['video_ids']) != (23 if fold == 1 else 24):
            raise ValueError('Registered inner dense role size differs')
        item = 'f%02d-linear-s7' % fold
        directory = root / REFERENCE_LOCAL / item
        result, profile = [read(directory / n) for n in ('training-result.json', 'profile.json')]
        if (result['status'] != 'fit_complete' or result['completed_epochs'] != 20 or result['stage'] != 'inner_select'
                or result['fold'] != fold or result['selected_epochs'] != selected or profile['head'] != 'linear'
                or profile['seed'] != 7 or result['initial_state_sha256'] != RGB_INITIAL_SHA):
            raise ValueError('Wrong selected A reference')
        files = {}
        for path in [directory / n for n in ('best.pt', 'training-result.json', 'profile.json', 'epochs.jsonl')] + sorted(directory.glob('validation-epoch-*.json')):
            name = path.relative_to(root / REFERENCE_LOCAL).as_posix()
            files[name] = dict(sha256=_sha(path), size_bytes=path.stat().st_size)
        if files[item + '/best.pt']['sha256'] != result['selected_checkpoint']['sha256']:
            raise ValueError('Selected A checkpoint bytes changed')
        references[str(fold)] = dict(item=item, selected_epochs=selected, files=files)
        fit_order.extend(['f%02d-B-s7' % fold, 'f%02d-C-s7' % fold])
    ids1, ids4 = set(roles['1']['video_ids']), set(roles['4']['video_ids'])
    if (len(ids1 | ids4), len(ids1 & ids4)) != (42, 5):
        raise ValueError('Expected42 unique/5 shared role videos')
    with (root / BASE17 / 'sources.csv').open(encoding='utf-8-sig', newline='') as stream:
        sources = {r['unit_id']: r for r in csv.DictReader(stream)}
    for fold in (1, 4):
        videos = roles[str(fold)]['video_ids']
        labels = Counter(sources[u]['video_label'] for u in videos)
        if (labels['pos'], labels['neg']) != ((18, 5) if fold == 1 else (18, 6)) or len({sources[u]['group_id'] for u in videos if sources[u]['video_label'] == 'neg'}) != 2:
            raise ValueError('Frozen dense video-label/negative-session counts differ')
    for uid in ids1 | ids4:
        for kind in ('coord', 'qc'):
            name, expected = sources[uid][kind + '_path'], sources[uid][kind + '_sha256']
            if _sha(root / name) != expected:
                raise ValueError('Consumed dense coordinate/QC identity changed')
            fingerprints[name] = expected
    out.mkdir(parents=True, exist_ok=False)
    _json(out / 'schedules.json', schedules)
    plan = dict(spec_id='EXP-021', version=1, status='prepared_not_training_authorization', seed=7, epochs=20,
        base17_path=BASE17, base20_path=BASE20, schedules_sha256=_sha(out / 'schedules.json'),
        source_fingerprints=fingerprints, old_unconsumed_deep_reference_fingerprints=inherited_refs,
        reference_roots=dict(local=REFERENCE_LOCAL, cloud=REFERENCE_CLOUD), A_references=references,
        reference_mapping='A paths are relative to explicit reference_root; old profile absolute provenance stays unchanged',
        RGB_expected_initial_sha256=RGB_INITIAL_SHA, backbone_expected_initial_sha256=BACKBONE_INITIAL_SHA,
        ffmpeg_sha256=ffmpeg_sha,
        kernel_path=KERNEL, kernel_sha256=fingerprints[KERNEL], descriptor_version=VERSION,
        dense_reference=DENSE, roles=roles, unique_role_videos=42, shared_role_videos=5,
        fit_order=fit_order, dense_order=['dense-f%02d-%s' % (fold, arm) for fold in (1, 4) for arm in ('A', 'B', 'C')],
        new_fit_epochs=80, new_fit_updates=1400, dense_role_models=6, dataloader_generator_seed=100007,
        pixel_acceptance='inherits actual EXP017 acceptance; selected JPEG/RGB checked while consumed; no new global pixel scan',
        descriptor_cache=False, shared_dense_B_C_descriptor=True, independent_test=False)
    _json(out / 'plan.json', plan)
    for name, sha in fingerprints.items():
        if _sha(root / name) != sha:
            raise ValueError('Source changed during prepare')
    return plan


def load(root, plan_dir, reference_root, *, with_dense=False):
    root, plan_dir, reference_root = Path(root).resolve(), Path(plan_dir).resolve(), Path(reference_root).resolve()
    plan_dir.relative_to(root)
    plan = read(plan_dir / 'plan.json')
    if plan['spec_id'] != 'EXP-021' or plan['epochs'] != 20 or plan['descriptor_version'] != VERSION:
        raise ValueError('Unregistered energy plan')
    allowed = {(root / plan['reference_roots']['local']).resolve(), Path(plan['reference_roots']['cloud']).resolve()}
    if reference_root not in allowed:
        raise ValueError('Explicit reference_root differs from frozen environment mapping')
    for name, sha in plan['source_fingerprints'].items():
        if _sha(rooted(root, name)) != sha:
            raise ValueError('Current inherited/EXP021 source fingerprint changed: ' + name)
    if _sha(plan_dir / 'schedules.json') != plan['schedules_sha256']:
        raise ValueError('Energy schedule SHA differs')
    _, bank, _, _, folds = load_cv_plan(root, root / plan['base17_path'], verify_sources=False)
    schedules = read(plan_dir / 'schedules.json')
    fingerprints = dict(plan['source_fingerprints'])
    for n in ('plan.json', 'schedules.json'):
        fingerprints[(plan_dir / n).relative_to(root).as_posix()] = _sha(plan_dir / n)
    with (root / plan['base17_path'] / 'sources.csv').open(encoding='utf-8-sig', newline='') as f:
        sources = {r['unit_id']: r for r in csv.DictReader(f)}
    context = dict(root=root, plan_dir=plan_dir, plan=plan, schedules=schedules, bank=bank, folds=folds,
        sources=sources, fingerprints=fingerprints, reference_root=reference_root)
    if with_dense:
        ref = read(root / plan['dense_reference'])
        uids = {u for role in plan['roles'].values() for u in role['video_ids']}
        if ref['status'] != 'cv_dense_pixels_verified' or not uids <= set(ref['videos']):
            raise ValueError('Missing existing canonical pixel reference')
        grouped = {u: [] for u in uids}
        with (root / ref['grid_path']).open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                if row['unit_id'] in grouped:
                    grouped[row['unit_id']].append(row)
        if any(len(grouped[u]) != ref['videos'][u]['windows'] for u in uids):
            raise ValueError('Original label-free physical dense grid differs')
        context['dense'] = dict(reference=ref, windows=grouped)
        acceptance = read(root / 'data/metadata/EXP-017/cv-cache-acceptance-v1/acceptance.json')
        known = {u: {} for u in uids}
        with (root / acceptance['verified_cache_index']).open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                if row['unit_id'] in known:
                    known[row['unit_id']][int(row['frame_index'])] = row
        context['known'] = known
    return context


def frame_refs(context, fold):
    """Stream one accepted index; retain only this fit's physical frame union."""
    root, plan, bank = context['root'], context['plan'], context['bank']
    schedule = context['schedules'][str(fold)]
    clips = set(c for row in schedule['epoch_clip_ids'] for c in row) | set(schedule['validation_clip_ids'])
    required = {(bank[c]['unit_id'], i) for c in clips for i in bank[c]['frame_indices']}
    acceptance = read(root / 'data/metadata/EXP-017/cv-cache-acceptance-v1/acceptance.json')
    if (acceptance['status'] != 'cv_supervised_cache_ready' or acceptance['pixel_version'] != PIXEL_VERSION
            or not acceptance['actual_JPEG_byte_SHA_checked'] or not acceptance['actual_RGB112_pixel_SHA_checked']):
        raise ValueError('Prior actual pixel acceptance unavailable')
    refs = {}
    with (root / acceptance['verified_required_frames_index']).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            key = row['unit_id'], int(row['frame_index'])
            if key not in required:
                continue
            if (key in refs or row['status'] != 'verified' or row['actual_jpeg_sha256'] != row['jpeg_sha256']
                    or row['actual_rgb112_sha256'] != row['rgb112_sha256'] or row['rgb112_shape'] != '112x112x3'):
                raise ValueError('Selected accepted frame identity differs')
            refs[key] = FrameRef(str(rooted(root, row['cache_path'])), row['jpeg_sha256'], row['rgb112_sha256'])
    if set(refs) != required:
        raise ValueError('Selected accepted physical frame missing')
    return refs
