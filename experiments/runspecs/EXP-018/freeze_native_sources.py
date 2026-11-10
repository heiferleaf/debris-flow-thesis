"""Fingerprint current originals for fixed cases; never infer raw-frame alignment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import time

SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[3]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    out.relative_to(ROOT / 'data/metadata/EXP-018')
    if out.exists():
        raise FileExistsError(out)
    mapping_path = ROOT / 'data/process/EXP-018/drift-blind-review-v1/mapping.json'
    videos_path = ROOT / 'data/metadata/EXP-016/metadata-v4/video-inputs.csv'
    mapping_sha, videos_sha = digest(mapping_path), digest(videos_path)
    cases = json.loads(mapping_path.read_text(encoding='utf-8'))['cases']
    if len(cases) != 12 or len({r['unit_id'] for r in cases}) != 12:
        raise ValueError('Only the twelve fixed unique-video cases are allowed')
    with videos_path.open(encoding='utf-8-sig', newline='') as stream:
        videos = {row['unit_id']: row for row in csv.DictReader(stream)}
    raw_root = (ROOT / 'data/raw/video_dataset_1').resolve()
    candidates, seen = [], set()
    for case in sorted(cases, key=lambda row: row['anonymous_id']):
        uid = case['unit_id']
        log = ROOT / ('data/process/EXP-008-units-src256/log/' + uid + '.txt')
        line = log.read_text(encoding='utf-8').splitlines()[0]
        match = re.fullmatch(r"Input #0, .+, from '(.+)':", line)
        if not match:
            raise ValueError('Exact source path not found in first log line: ' + uid)
        path = Path(match.group(1))
        relative = path.relative_to(ROOT).as_posix()
        resolved = path.resolve(strict=True)
        resolved.relative_to(raw_root)
        if not resolved.is_file() or path.stem != videos[uid]['video_id']:
            raise ValueError('Log source/stem differs from frozen video identity: ' + uid)
        if resolved in seen:
            raise ValueError('Two fixed videos refer to one original')
        seen.add(resolved)
        candidates.append((case, path, relative, log, digest(log)))
    if sum(path.stat().st_size for _, path, _, _, _ in candidates) > 15 * 1024 ** 3:
        raise ValueError('Unexpected current-source size; refuse broad inventory')
    records = []
    started = time.monotonic()
    for case, path, relative, log, log_sha in candidates:
        before = path.stat()
        source_sha = digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source changed while fingerprinting: ' + case['unit_id'])
        if digest(log) != log_sha:
            raise ValueError('Source locator log changed')
        records.append(dict(case_id=case['anonymous_id'], unit_id=case['unit_id'],
                            source_path=relative, source_bytes=after.st_size,
                            source_sha256=source_sha,
                            locator_log=log.relative_to(ROOT).as_posix(), locator_log_sha256=log_sha,
                            native_alignment_status='unverified',
                            fingerprint_scope='current original; not a historical-transcode source hash'))
        print(json.dumps(dict(unit_id=case['unit_id'], completed=len(records), total=12)), flush=True)
    if digest(mapping_path) != mapping_sha or digest(videos_path) != videos_sha:
        raise ValueError('Fixed-case or metadata input changed')
    report = dict(status='current_original_fingerprints_complete_alignment_unverified',
                  source_files=records, total_bytes=sum(r['source_bytes'] for r in records),
                  seconds=time.monotonic() - started, video_decoded=False, model_executed=False,
                  labels_changed=False, historical_source_fingerprint_claimed=False,
                  input_sha256={mapping_path.relative_to(ROOT).as_posix(): mapping_sha,
                                videos_path.relative_to(ROOT).as_posix(): videos_sha},
                  generator_sha256=digest(SOURCE),
                  command=[sys.executable, str(SOURCE), *sys.argv[1:]])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'sources.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'generator-source.py').write_bytes(SOURCE.read_bytes())
    print(json.dumps(dict(status=report['status'], files=12, total_bytes=report['total_bytes'])), flush=True)


if __name__ == '__main__':
    main()
