"""Prepare prediction/selection-role-blinded sheets for a limited visibility review.

No video decoder, model inference, training, label changes or content adjudication.
Only blind/*.png may be given to the image reviewer; mapping.json reveals roles.
Native OSD is retained. This is not identity blinding or truth adjudication.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys

from PIL import Image, ImageDraw, ImageFont

SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[3]
DEFAULT_SEED = 180208
ROLES = ('new_FN', 'stable_correct')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Inputs:
    def __init__(self):
        self.records = {}

    def read(self, path, expected=None):
        path = Path(path).resolve()
        relative = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError('Frozen input SHA mismatch: ' + relative)
        self.records[relative] = dict(path=relative, sha256=digest, size_bytes=len(raw))
        return raw.decode('utf-8-sig')

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected))

    def csv(self, path, expected=None):
        return list(csv.DictReader(io.StringIO(self.read(path, expected))))

    def unchanged(self):
        for row in self.records.values():
            if sha(ROOT / row['path']) != row['sha256']:
                raise ValueError('Input changed during preparation: ' + row['path'])


def sorted_clips(values):
    return sorted(values, key=lambda r: (r['group_id'], r['unit_id'], r['clip_id']))


def select_cases(pools, seed):
    """Select by role sign only, group coverage and seeded choice; never score rank."""
    rng = random.Random(seed)
    pools = {role: sorted_clips(pools[role]) for role in ROLES}
    for role in ROLES:
        if len(pools[role]) < 6 or len({r['group_id'] for r in pools[role]}) < 3:
            raise ValueError('Insufficient six clips/three groups for ' + role)
    groups = {role: sorted({r['group_id'] for r in pools[role]}) for role in ROLES}
    shared = sorted(set(groups[ROLES[0]]) & set(groups[ROLES[1]]))
    # This task was authorized with two common groups, not a fully matched design.
    if len(shared) != 2:
        raise ValueError('Expected exactly two common candidate groups; stop for review')
    selected = {role: [] for role in ROLES}
    used_videos, matched_pairs = set(), []
    ordered_shared = list(shared)
    rng.shuffle(ordered_shared)
    for group in ordered_shared:
        pairs = [(a, b) for a in pools[ROLES[0]] for b in pools[ROLES[1]]
                 if a['group_id'] == b['group_id'] == group
                 and a['unit_id'] != b['unit_id']
                 and a['unit_id'] not in used_videos and b['unit_id'] not in used_videos]
        if not pairs:
            raise ValueError('Cannot preserve unique videos for a common-group pair')
        a, b = pairs[rng.randrange(len(pairs))]
        selected[ROLES[0]].append(a)
        selected[ROLES[1]].append(b)
        used_videos.update((a['unit_id'], b['unit_id']))
        matched_pairs.append(dict(group_id=group, new_FN_clip_id=a['clip_id'],
                                  stable_correct_clip_id=b['clip_id']))
    # Cover every remaining available group once before adding further clips.
    for role in ROLES:
        order = [g for g in groups[role] if g not in shared]
        rng.shuffle(order)
        for group in order:
            if len(selected[role]) == 6:
                break
            options = [r for r in pools[role] if r['group_id'] == group
                       and r['unit_id'] not in used_videos]
            if not options:
                raise ValueError('Cannot cover remaining group with a unique video: ' + role)
            choice = options[rng.randrange(len(options))]
            selected[role].append(choice)
            used_videos.add(choice['unit_id'])
    # Seeded group round-robin fills spare slots, preserving global video uniqueness.
    for role in ROLES:
        order = list(groups[role])
        rng.shuffle(order)
        while len(selected[role]) < 6:
            before = len(selected[role])
            for group in order:
                options = [r for r in pools[role] if r['group_id'] == group
                           and r['unit_id'] not in used_videos]
                if options:
                    choice = options[rng.randrange(len(options))]
                    selected[role].append(choice)
                    used_videos.add(choice['unit_id'])
                if len(selected[role]) == 6:
                    break
            if before == len(selected[role]):
                raise ValueError('Cannot reach twelve globally unique videos; stop, no relaxation')
    chosen = [dict(r, role=role) for role in ROLES for r in selected[role]]
    if (len(chosen) != 12 or len({r['clip_id'] for r in chosen}) != 12
            or len({r['unit_id'] for r in chosen}) != 12
            or any(len({r['group_id'] for r in selected[role]}) < 3 for role in ROLES)):
        raise ValueError('Final count/group/unique-video requirement failed')
    # An independent seed stream randomizes anonymous presentation order.
    anonymous = sorted_clips(chosen)
    random.Random(seed + 1).shuffle(anonymous)
    for index, record in enumerate(anonymous, 1):
        record['anonymous_id'] = 'C%02d' % index
    return anonymous, shared, matched_pairs


def prepare(seed, out):
    recover_empty = False
    if out.exists():
        # An interrupted run may have created only these two empty directories.
        # No existing output file, including a partial figure, is ever overwritten.
        if (out.is_dir() and {p.name for p in out.iterdir()} == {'blind'}
                and (out / 'blind').is_dir() and not any((out / 'blind').iterdir())):
            recover_empty = True
        else:
            raise FileExistsError('Refuse to overwrite an existing review: ' + str(out))
    inputs = Inputs()
    inputs.read(SOURCE)
    pair_dir = ROOT / 'data/metadata/EXP-018/paired-plan-v1'
    pair = inputs.json(pair_dir / 'plan.json')
    if pair['spec_id'] != 'EXP-018' or pair['seed'] != 7 or pair['validation_count'] != 136:
        raise ValueError('Unexpected registered EXP-018 plan')
    base_dir = ROOT / pair['base_plan_path']
    base = inputs.json(base_dir / 'plan.json', pair['base_plan_sha256'])
    resource = ROOT / base['resource']
    refs = {r['path']: r['sha256'] for r in base['sources']}
    def expected(path):
        return refs[Path(path).relative_to(ROOT).as_posix()]
    windows_path, segments_path = resource / 'windows.jsonl', resource / 'segments.csv'
    windows = [json.loads(line) for line in inputs.read(windows_path, expected(windows_path)).splitlines() if line]
    segments = inputs.csv(segments_path, expected(segments_path))
    videos_path = resource / 'video-inputs.csv'
    videos = inputs.csv(videos_path, expected(videos_path))
    assignments = inputs.csv(base_dir / 'assignments.csv', base['files']['assignments.csv'])
    bank = {w['clip_id']: w for w in windows}
    segment_bank = {(s['unit_id'], s['segment_id']): s for s in segments}
    video_bank = {v['unit_id']: v for v in videos}
    if len(bank) != len(windows) or len(segment_bank) != len(segments) or len(video_bank) != len(videos):
        raise ValueError('Duplicate input identity')
    validation_ids = [a['clip_id'] for a in assignments if a['split'] == 'validation']
    if validation_ids != pair['validation_clip_ids'] or len(set(validation_ids)) != 136:
        raise ValueError('Validation assignment differs from frozen pair plan')
    run = ROOT / 'data/results/exp018/paired-s7-20261101-v1/bce'
    profile = inputs.json(run / 'profile.json')
    if (profile['objective_mode'] != 'bce' or profile['seed'] != 7 or profile['smoke']
            or profile['pair_plan_sha256'] != sha(pair_dir / 'plan.json')
            or profile['plan_sha256'] != pair['base_plan_sha256']
            or profile['score_threshold'] != .5):
        raise ValueError('Not the registered ordinary BCE run')
    predictions = {}
    for epoch in (2, 8):
        rows = inputs.json(run / ('validation-epoch-%02d.json' % epoch))
        if [r['clip_id'] for r in rows] != validation_ids:
            raise ValueError('Prediction coverage/order mismatch')
        if any(type(r['label']) is not int or r['label'] not in (0, 1)
               or type(r['logit']) not in (int, float) or not math.isfinite(r['logit']) for r in rows):
            raise ValueError('Invalid prediction label/logit')
        predictions[epoch] = {r['clip_id']: r for r in rows}
    pools = {role: [] for role in ROLES}
    for cid in validation_ids:
        w = bank[cid]
        s = segment_bank[(w['unit_id'], w['segment_id'])]
        v = video_bank[w['unit_id']]
        label = int(w['kind'] == 'pos')
        if (w['split'] != 'validation' or s['split'] != 'validation' or v['split'] != 'validation'
                or w['kind'] != s['kind'] or w['group_id'] != s['group_id']
                or w['group_id'] != v['group_id']
                or any(predictions[e][cid]['label'] != label for e in (2, 8))):
            raise ValueError('Frozen metadata/prediction identity mismatch')
        ix, pts = w['frame_indices'], w['source_pts_seconds']
        if (len(ix) != 16 or len(set(ix)) != 16 or len(pts) != 16
                or any(type(i) is not int for i in ix) or not all(math.isfinite(p) for p in pts)
                or any(a >= b for a, b in zip(ix, ix[1:]))
                or any(a >= b for a, b in zip(pts, pts[1:]))
                or not 2.9 - 1e-9 <= pts[-1] - pts[0] <= 3.1 + 1e-9
                or not float(s['sampling_start_sec']) <= pts[0] < pts[-1] < float(s['sampling_end_sec'])):
            raise ValueError('Malformed source window')
        if not label:
            continue
        a, b = float(s['annotation_start_sec']), float(s['annotation_end_sec'])
        if not math.isfinite(a) or not math.isfinite(b) or not a <= pts[0] < pts[-1] < b:
            raise ValueError('Missing/incompatible original annotation; no boundary fallback')
        elapsed, remaining = pts[0] - a, b - pts[-1]
        if elapsed < 10 or remaining < 3 or predictions[2][cid]['logit'] < 0:
            continue
        role = 'new_FN' if predictions[8][cid]['logit'] < 0 else 'stable_correct'
        pools[role].append(dict(clip_id=cid, unit_id=w['unit_id'], group_id=w['group_id'],
            segment_id=w['segment_id'], original_annotation_start_sec=a,
            original_annotation_end_sec=b, from_annotation_start_sec=elapsed,
            to_annotation_end_sec=remaining))
    chosen, shared, pairs = select_cases(pools, seed)
    completion_path = ROOT / 'data/process/EXP-016/metadata-v4/cache-completion.json'
    completion = inputs.json(completion_path, expected(completion_path))
    cache_path = ROOT / completion['verified_cache_index']
    cache_rows = inputs.csv(cache_path, completion['verified_cache_index_sha256'])
    if expected(cache_path) != completion['verified_cache_index_sha256']:
        raise ValueError('Base plan and completion cache-index hash disagree')
    wanted = {(c['unit_id'], i) for c in chosen for i in bank[c['clip_id']]['frame_indices']}
    frames = {}
    for row in cache_rows:
        key = (row['unit_id'], int(row['frame_index']))
        if key in wanted:
            if key in frames:
                raise ValueError('Duplicate selected cache key')
            path = ROOT / row['cache_path']
            raw = path.read_bytes()
            if len(raw) != int(row['size_bytes']) or hashlib.sha256(raw).hexdigest() != row['jpeg_sha256']:
                raise ValueError('Selected JPEG SHA/size mismatch')
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                if image.format != 'JPEG' or image.size != (256, 256) or image.mode != 'RGB':
                    raise ValueError('Selected JPEG format/shape/mode mismatch')
            frames[key] = row
    if set(frames) != wanted:
        raise ValueError('Missing selected cache frames')
    for case in chosen:
        w = bank[case['clip_id']]
        for index, pts in zip(w['frame_indices'], w['source_pts_seconds']):
            if abs(float(frames[(case['unit_id'], index)]['source_pts_seconds']) - pts) > 1e-8:
                raise ValueError('Cache PTS differs from frozen window')
    inputs.unchanged()
    out.mkdir(parents=True, exist_ok=recover_empty)
    blind = out / 'blind'
    blind.mkdir(exist_ok=recover_empty)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    small = ImageFont.truetype(str(font_path), 18)
    title_font = ImageFont.truetype(str(font_path), 28)
    figures = []
    for case in chosen:
        cid, uid, anonymous_id = case['clip_id'], case['unit_id'], case['anonymous_id']
        w = bank[cid]
        sheet = Image.new('RGB', (1072, 1230), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 8), anonymous_id, fill='black', font=title_font)
        frame_records = []
        for j, (index, pts) in enumerate(zip(w['frame_indices'], w['source_pts_seconds'])):
            ref = frames[(uid, index)]
            with Image.open(ROOT / ref['cache_path']) as image:
                tile = image.convert('RGB')
            x, y = 12 + (j % 4) * 264, 52 + (j // 4) * 290
            sheet.paste(tile, (x, y))
            relative_time = pts - w['source_pts_seconds'][0]
            draw.text((x, y + 259), '%02d | +%.3fs' % (j + 1, relative_time), fill='black', font=small)
            frame_record = dict(ref)
            frame_record.update(position=j + 1, relative_time_seconds=relative_time,
                                source_frame_index=index, source_pts_seconds=pts)
            frame_records.append(frame_record)
        target = blind / (anonymous_id + '.png')
        # Fresh RGB canvas, no JPEG EXIF, comments, sidecars or identity-bearing PNG metadata.
        sheet.save(target, format='PNG')
        case.update(window=w, original_segment=segment_bank[(uid, w['segment_id'])],
                    source_video_metadata=video_bank[uid], frames=frame_records,
                    epoch2_prediction=predictions[2][cid], epoch8_prediction=predictions[8][cid],
                    blind_image='blind/' + target.name, blind_image_sha256=sha(target))
        figures.append(dict(anonymous_id=anonymous_id, path='blind/' + target.name,
                            sha256=sha(target), size_bytes=target.stat().st_size))
    if {p.name for p in blind.iterdir()} != {'C%02d.png' % i for i in range(1, 13)}:
        raise ValueError('Blind directory contains unexpected identity-bearing files')
    for path in blind.iterdir():
        with Image.open(path) as image:
            if image.info:
                raise ValueError('Unexpected embedded PNG metadata')
    (out / 'generator-source.py').write_bytes(SOURCE.read_bytes())
    result = dict(review_id='EXP018-drift-blind-review-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        status='prepared_pending_independent_blind_visibility_review', seed=seed,
        no_inference=True, no_video_decoding=True, no_training=True, no_label_changes=True,
        content_truth_adjudicated=False, selection=dict(
            roles={'new_FN':'original positive, ep2 logit>=0 and ep8 logit<0',
                   'stable_correct':'original positive, ep2 and ep8 logit>=0'},
            decision_threshold='registered sigmoid(logit)>=0.5, equivalent logit>=0',
            internal_rule='first real source PTS - original annotation start >=10s; original annotation end - last real source PTS >=3s; strict original bounds, no QC-boundary substitute',
            sort_key=['group_id','unit_id','clip_id'], random_algorithm='Python random.Random(seed); sorted candidate choices then randrange; seeded group shuffle and round-robin',
            stages=['one different-video pair from each of the two shared groups',
                    'cover each remaining group once per role',
                    'seeded group round-robin to six per role without any reused video',
                    'sort combined chosen clips then independent Random(seed+1).shuffle into C01..C12'],
            score_magnitude_used_for_ranking=False, all_candidates={r:sorted_clips(pools[r]) for r in ROLES},
            pool_counts={r:dict(windows=len(pools[r]), videos=len({x['unit_id'] for x in pools[r]}),
                               groups=len({x['group_id'] for x in pools[r]})) for r in ROLES},
            selected_counts={r:dict(windows=sum(x['role']==r for x in chosen),
                                    groups=len({x['group_id'] for x in chosen if x['role']==r})) for r in ROLES},
            unique_videos=len({x['unit_id'] for x in chosen}), common_candidate_groups=shared,
            designated_same_group_pairs=pairs,
            limitation='Only two candidate groups contain both roles. These selected post-hoc examples are not a fully matched or causal control comparison; no general prevalence estimate.'),
        blinding_scope='prediction_and_selection_role_blinded', identity_blind=False,
        native_OSD_retained=True, added_OSD_masking=False,
        blind_contract='Give reviewer only blind/*.png, never mapping.json. Generated text contains C identifier, position1..16 and time relative to first frame; no added role, prediction, label, unit/group, source path or absolute PTS. Source image pixels and their native site/date OSD are retained unchanged. Identity is not blinded. The reviewer has no selection-role/model-score mapping.',
        input_fingerprints=list(inputs.records.values()), selected_unique_verified_jpegs=len(frames),
        font=dict(path=str(font_path), sha256=sha(font_path)), environment=dict(python=sys.version, pillow=Image.__version__),
        generator_source_sha256=sha(SOURCE), figures=figures, cases=chosen)
    with (out / 'mapping.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    inputs.unchanged()
    return dict(output=out.relative_to(ROOT).as_posix(), clips=12,
                unique_videos=len({x['unit_id'] for x in chosen}),
                groups_by_role=result['selection']['selected_counts'], verified_jpegs=len(frames),
                mapping_sha256=sha(out / 'mapping.json'), generator_source_sha256=sha(SOURCE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    out = ROOT / 'data/process/EXP-018/drift-blind-review-v1'
    print(json.dumps(prepare(args.seed, out), ensure_ascii=True))


if __name__ == '__main__':
    main()
