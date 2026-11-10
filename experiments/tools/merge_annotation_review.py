"""Build a new editable N/P/T review without overwriting a live or historical copy.

The proposal is {'items': [{'unit_id': ..., 'recommended_intervals': [
{'start_sec': ..., 'end_sec': ..., 'state': 'N'|'P'|'T'|'cut'}]}]}.
All intervals for a proposed video are explicit; missing time never becomes N.
This is review bookkeeping, not training-label acceptance or visual validation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KINDS = {'N': 'neg', 'P': 'pos', 'T': 'transition', 'cut': 'cut'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def encode(document):
    return (json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def ref(path):
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def support(intervals):
    """Union of intervals, allowing adjacent original rows to be merged."""
    merged = []
    for start, end in sorted(intervals):
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError('Invalid interval')
        if merged and start < merged[-1][1] - 1e-6:
            raise ValueError('Overlapping intervals')
        if merged and abs(start - merged[-1][1]) <= 1e-6:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged


def replace_unit(originals, intervals):
    if not originals or not intervals:
        raise ValueError('Missing original or proposed intervals')
    candidates = sorted(intervals, key=lambda x: float(x['start_sec']))
    for row in candidates:
        if row['state'] not in KINDS:
            raise ValueError('Only N/P/T and preserved cut are supported')
    new_support = support([(float(r['start_sec']), float(r['end_sec'])) for r in candidates])
    old_support = support([(float(r['start_sec']), float(r['end_sec'])) for r in originals])
    if new_support != old_support:
        raise ValueError('Proposal must retain original explicit time support; no gap filling')
    old_cut = support([(float(r['start_sec']), float(r['end_sec'])) for r in originals if r['kind'] == 'cut'])
    new_cut = support([(float(r['start_sec']), float(r['end_sec'])) for r in candidates if r['state'] == 'cut'])
    if old_cut != new_cut:
        raise ValueError('User cut coverage must remain unchanged')
    result = []
    for proposal in candidates:
        start, end = float(proposal['start_sec']), float(proposal['end_sec'])
        parent = max(originals, key=lambda r: max(0., min(end, float(r['end_sec'])) - max(start, float(r['start_sec']))))
        row = dict(parent)
        row.update(start_sec=format(start, '.12g'), end_sec=format(end, '.12g'),
                   kind=KINDS[proposal['state']], confidence='real')
        result.append(row)
    return result


def merge(baseline, current, proposals, output, execute=False):
    baseline, current, proposals, output = map(lambda p: Path(p).resolve(), (baseline, current, proposals, output))
    if output.exists():
        raise FileExistsError('New review directory must not already exist')
    if output in (baseline, current) or baseline in output.parents or current in output.parents:
        raise ValueError('Output must be separate from both inputs')
    base_index, current_index = load(baseline / 'sources.json'), load(current / 'sources.json')
    proposal_raw = proposals.read_bytes()
    proposal_doc = json.loads(proposal_raw.decode('utf-8-sig'))
    by_unit = {}
    for item in proposal_doc['items']:
        if item['unit_id'] in by_unit:
            raise ValueError('Duplicate proposed video')
        by_unit[item['unit_id']] = item
    captured, files, sources, used = {}, {}, {}, set()
    # A live user save invalidates this merge; never silently overwrite it.
    for key, item in current_index['sources'].items():
        path = current / item['annotations_csv']
        raw = path.read_bytes()
        if sha(raw) != item.get('initial_annotations_sha256'):
            raise ValueError('Live review has new edits; reconcile first: ' + str(path))
        captured[path] = raw
    for key, source in base_index['sources'].items():
        path = baseline / source['annotations_csv']
        raw = path.read_bytes()
        captured[path] = raw
        table = list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))
        fieldnames, rows = table[0], []
        for values in table[1:]:
            row = dict(zip(fieldnames, values))
            # Preserve legacy extra cells without silently dropping them or
            # interpreting them as semantic fields.
            row['_legacy_extra'] = values[len(fieldnames):]
            rows.append(row)
        units = defaultdict(list)
        for row in rows:
            units[row['video_key']].append(row)
        replacement = {}
        for unit, item in by_unit.items():
            if unit in units:
                if unit in used:
                    raise ValueError('Ambiguous video across sources')
                replacement[unit] = replace_unit(units[unit], item['recommended_intervals'])
                used.add(unit)
        merged, seen = [], set()
        for row in rows:
            unit = row['video_key']
            if unit not in replacement:
                merged.append(row)
            elif unit not in seen:
                merged.extend(replacement[unit])
                seen.add(unit)
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(fieldnames)
        for row in merged:
            writer.writerow([row.get(field, '') for field in fieldnames] + row.get('_legacy_extra', []))
        data = stream.getvalue().encode('utf-8-sig') if replacement else raw
        files[source['annotations_csv']] = data
        sources[key] = {**source, 'parent_annotation_csv': ref(path), 'parent_annotation_sha256': sha(raw),
                        'initial_annotations_sha256': sha(data)}
    if used != set(by_unit):
        raise ValueError('Proposed videos missing from baseline: ' + str(sorted(set(by_unit) - used)))
    report = {'schema_version': 1, 'status': 'editable_reaudit_candidates_not_training_truth',
              'baseline': ref(baseline), 'previous_live_review': ref(current), 'output': ref(output),
              'proposal_path': ref(proposals), 'proposal_sha256': sha(proposal_raw),
              'proposed_units': sorted(used), 'user_cut_preserved': True,
              'live_edits_detected': False, 'training_ready': False,
              'input_fingerprints': [{'path': ref(p), 'sha256': sha(raw)} for p, raw in captured.items()],
              'current_pointer_changed': False, 'images_copied': 0}
    files['sources.json'] = encode({'schema_version': 1, 'sources': sources, 'editable_review': True,
                                    'training_ready': False, 'revision_status': report['status'],
                                    'audit_proposals': ref(proposals), 'provenance_sidecar': 'merge-provenance.json'})
    files['merge-provenance.json'] = encode(report)
    if execute:
        for path, raw in captured.items():
            if path.read_bytes() != raw:
                raise ValueError('Input changed during merge; no output written')
        if proposals.read_bytes() != proposal_raw:
            raise ValueError('Proposals changed during merge')
        output.mkdir(parents=True, exist_ok=False)
        for name, raw in files.items():
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(raw)
    return {**report, 'dry_run': not execute}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'current', 'proposals', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    result = merge(args.baseline, args.current, args.proposals, args.out, args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
