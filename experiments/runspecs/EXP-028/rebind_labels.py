"""Rebind a frozen physical grid to new labels without decoding or repeating QC.

Writes a compact semantic overlay, role counts and training-side coefficients.
The parent remains the source of frame identities and prior technical checks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.common.video.full_coverage_sampling import validate_release, integrate_release, diagnostic_loss_coefficients


def fingerprint(path):
    return {'path':Path(path).as_posix(),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def dump(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def csv_dump(path,rows):
    rows=list(rows)
    with path.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def bind(row,intervals):
    composition=None
    if row.get('frame_indices') is not None:
        composition=integrate_release(row['actual_start_seconds'],row['actual_end_seconds'],intervals)
    legal=bool(composition and composition['semantic_existence_complete']
               and row['legacy_guard_pass'] and row['gray_QC_pass'] and row['PTS_pass']
               and row.get('retained_unique_physical_window',False))
    return composition,legal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--roles',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    release=json.loads(args.release.read_text(encoding='utf-8-sig'))
    grouped=defaultdict(list)
    for row in release['rows']:
        grouped[row['unit_id']].append(row)
    for unit,rows in grouped.items():
        grouped[unit]=validate_release(rows)
    roles=json.loads(args.roles.read_text(encoding='utf-8-sig'))
    camera_roles=roles['camera_roles']
    if set(camera_roles.values())!={'train','inner_validation','outer_evaluation'}:
        raise ValueError('Exactly three camera roles are required')
    output=args.out
    if output.exists():
        raise FileExistsError('Use a new overlay version')
    stats=defaultdict(Counter)
    unit_camera={}
    members=[]
    overlays=[]
    parent_hash=hashlib.sha256()
    seen=set()
    for line_number,raw in enumerate(args.parent.open('rb'),1):
        parent_hash.update(raw)
        row=json.loads(raw)
        unit,camera=row['unit_id'],row['source_camera']
        if unit not in grouped or camera not in camera_roles:
            raise ValueError('Frozen pool/camera is missing from the selected release or role map')
        if unit in unit_camera and unit_camera[unit]!=camera:
            raise ValueError('One unit cannot cross camera roles')
        unit_camera[unit]=camera
        role=camera_roles[camera]
        composition,legal=bind(row,grouped[unit])
        q=composition['q_N_T_P'] if composition else None
        status=('NTP'[q.index(1.0)] if 1.0 in q else 'MIX') if q is not None else 'incomplete'
        tally=stats[unit]
        tally['grid_points']+=1
        tally['matched_points']+=int(composition is not None)
        tally['old_legal_K']+=int(row['conservative_legal'])
        tally['legal_K']+=int(legal)
        tally['cache_complete_K']+=int(legal and row.get('cache_complete_for_matched_window',False))
        tally['legal_'+status]+=int(legal)
        tally['legal_with_T']+=int(legal and q[1]>0)
        tally['legal_with_P']+=int(legal and q[2]>0)
        if legal:
            key=row['physical_window_key']
            if key in seen:
                raise ValueError('Legal physical identity repeated in parent grid')
            seen.add(key)
            members.append({'unit_id':unit,'source_camera':camera,'role':role,
                            'physical_video_id':row['physical_video_id'],'physical_window_key':key,
                            'composition':composition,'motion_state':'unknown'})
        overlays.append({'parent_line':line_number,'unit_id':unit,'source_camera':camera,'role':role,
                         'grid_ordinal':row['grid_ordinal'],'physical_window_key':row.get('physical_window_key'),
                         'actual_start_seconds':row.get('actual_start_seconds'),'actual_end_seconds':row.get('actual_end_seconds'),
                         'semantic_status':status,'q_N':q[0] if q is not None else '',
                         'q_T':q[1] if q is not None else '', 'q_P':q[2] if q is not None else '',
                         'r_existence':composition['r_existence'] if composition and composition['r_existence'] is not None else '',
                         'conservative_legal':legal,'cache_complete_legal':bool(legal and row.get('cache_complete_for_matched_window',False)),
                         'legacy_guard_pass':row['legacy_guard_pass'],'gray_QC_pass':row['gray_QC_pass'],'PTS_pass':row['PTS_pass']})
    if len(unit_camera)!=150 or set(unit_camera)!=set(grouped) or set(unit_camera.values())!=set(camera_roles):
        raise ValueError('The complete frozen 150-unit/11-camera pool must be represented')
    output.mkdir(parents=True,exist_ok=False)
    per_unit=[{'unit_id':u,'source_camera':unit_camera[u],'role':camera_roles[unit_camera[u]],**stats[u]} for u in sorted(stats)]
    csv_dump(output/'window-label-overlay.csv',overlays)
    csv_dump(output/'per-unit.csv',({**r,**{k:r.get(k,0) for k in ['legal_N','legal_T','legal_P','legal_MIX']}} for r in per_unit))
    role_summaries={}
    for role in ['train','inner_validation','outer_evaluation','outer_refit']:
        accepted={'train','inner_validation'} if role=='outer_refit' else {role}
        units=[r for r in per_unit if r['role'] in accepted]
        legal=[r for r in members if r['role'] in accepted]
        k=len(legal)
        cameras=sorted({r['source_camera'] for r in units})
        role_summaries[role]={'cameras':cameras,'pool_units':len(units),'units_with_legal_windows':sum(r['legal_K']>0 for r in units),
                              'legal_K':k,'cache_complete_K':sum(r['cache_complete_K'] for r in units),
                              'legal_N':sum(r.get('legal_N',0) for r in units),'legal_T':sum(r.get('legal_T',0) for r in units),
                              'legal_P':sum(r.get('legal_P',0) for r in units),'legal_MIX':sum(r.get('legal_MIX',0) for r in units),
                              'one_traversal_updates':math.ceil(k/24),'Emax_under_10000_updates':10000//math.ceil(k/24) if k else 0,
                              'training_ready':False}
        if role in {'train','outer_refit'}:
            coefficients,camera_report=diagnostic_loss_coefficients(legal)
            csv_dump(output/(role+'-loss-coefficients.csv'),coefficients)
            dump(output/(role+'-camera-coefficients.json'),camera_report)
            if abs(math.fsum(x['loss_coefficient_w'] for x in coefficients)-k)>max(1e-6,k*1e-10):
                raise ValueError('Loss coefficient normalization failed')
    common_epochs=min(role_summaries[r]['Emax_under_10000_updates'] for r in ['train','outer_refit'])
    summary={'schema_version':1,'training_ready':False,'release':fingerprint(args.release),'roles':fingerprint(args.roles),
             'parent_grid':{'path':args.parent.as_posix(),'sha256':parent_hash.hexdigest()},
             'known_pool_units':150,'camera_count':len(camera_roles),'full_grid_count':len(overlays),
             'legal_K':len(members),'cache_complete_K':sum(r['cache_complete_K'] for r in per_unit),
             'zero_K_units':[r['unit_id'] for r in per_unit if not r['legal_K']],
             'roles_summary':role_summaries,'common_complete_traversal_cap':common_epochs,
             'budget_spec_feasible_at_least_two_traversals':common_epochs>=2,
             'motion_policy':'unknown remains undiscounted; scoped flowing_N checks do not justify pure-static tagging',
             'technical_policy':'Reuse frozen parent guard/gray/PTS/cache facts; no semantic relabel clears QC.',
             'performed':{'video_decodes':0,'JPEG_reads':0,'new_QC_scans':0,'model_fits':0,'paid_requests':0},
             'limits':'A semantic overlay of inherited technical checks, not new raw-media/QC certification or training permission. Parent grid holds frame identities; weights are calculated separately for actual training and refit roles.'}
    dump(output/'summary.json',summary)
    print(json.dumps({'legal_K':len(members),'cache_complete_K':summary['cache_complete_K'],'zero_K_units':summary['zero_K_units'],
                      'roles':{k:{a:v[a] for a in ['pool_units','legal_K','one_traversal_updates']} for k,v in role_summaries.items()},
                      'common_complete_traversal_cap':common_epochs},ensure_ascii=False))


if __name__=='__main__':
    main()
