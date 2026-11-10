"""EXP-017 CPU input preparation; default check never decodes media.

extract writes only41848 missing JPEGs; dense writes119 new canonical pixel
references and reuses31 old label-free references. Pixel acceptance is separate.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.cv_cache_preparation import context, extract_missing, prepare_dense, record_cli_failure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check','extract','dense'), default='check')
    parser.add_argument('--plan', type=Path, default=ROOT/'data/metadata/EXP-017/role-free-plan-v3')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--ffmpeg', type=Path, default=Path('D:/CodexCache/debris-flow-thesis/tools/ffmpeg-9.0.2/ffmpeg.exe'))
    parser.add_argument('--producer', type=Path, default=ROOT/'data/process/EXP-017/cv-inputs-v1')
    parser.add_argument('--acceptance', type=Path, default=ROOT/'data/metadata/EXP-017/cv-cache-acceptance-v1/acceptance.json')
    parser.add_argument('--resume', action='store_true')
    a = parser.parse_args(argv)
    if a.stage == 'check':
        c = context(ROOT, a.plan)
        result = dict(status='metadata_inventory_only_pixels_unverified', plan_sha256=c['plan_sha256'],
            required_frames=len(c['required']), reusable_indexed_JPEGs=len(c['reused']), missing_JPEGs=len(c['missing']),
            no_media_decode=True, no_JPEG_pixel_read=True, no_training=True)
    else:
        target = a.out or ROOT/'data/process/EXP-017'/('cv-inputs-v1' if a.stage == 'extract' else 'dense-pixel-reference-v1')
        try:
            if a.stage == 'extract': result = extract_missing(ROOT, a.plan, target, a.ffmpeg, a.resume)
            else: result = prepare_dense(ROOT, a.plan, a.producer, a.acceptance, target, a.ffmpeg, a.resume)
        except Exception as error:
            record_cli_failure(ROOT, target, a.stage, error)
            raise
    print(json.dumps({k:v for k,v in result.items() if k not in ('videos','completed_video_receipts')}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
