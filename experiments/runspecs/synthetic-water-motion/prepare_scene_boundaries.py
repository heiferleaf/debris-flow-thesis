"""Refine coarse river polygons locally; preserve source markings and history.

Produces review candidates, not automatically accepted masks. The generated
unclipped reference supplies edge evidence; protected rocks remain excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def prepare(row, output, band=24):
    if (output/'boundary-record.json').exists():raise ValueError('Preserve existing boundary review')
    output.mkdir(parents=True,exist_ok=True)
    source=Path(row['source']);original=cv2.imread(str(source));h,w=original.shape[:2]
    generated=cv2.imread(str(source.parent/'generated-full.png'))
    generated=cv2.resize(generated,(w,h),interpolation=cv2.INTER_CUBIC)
    dry_path=Path('data/process/synthetic-video/dry-scenes-20261108-v2')/row['scene_id']/'dry-reference.png'
    dry=cv2.imread(str(dry_path));dry=cv2.resize(dry,(w,h),interpolation=cv2.INTER_CUBIC)
    geometry_path=Path(row['geometry']);geometry=json.loads(geometry_path.read_text('utf-8-sig'))
    draft=geometry['draft'];mask=np.zeros((h,w),np.uint8)
    polygon=np.rint(np.array(draft['polygon'])*[w-1,h-1]).astype(np.int32)
    cv2.fillPoly(mask,[polygon],255)
    protected=np.zeros_like(mask)
    for ex in draft.get('exclusions',[]):
        cv2.fillPoly(protected,[np.rint(np.array(ex)*[w-1,h-1]).astype(np.int32)],255)
    inside=cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,5)
    outside=cv2.distanceTransform((mask==0).astype(np.uint8),cv2.DIST_L2,5)
    labels=np.where(mask>0,cv2.GC_PR_FGD,cv2.GC_PR_BGD).astype(np.uint8)
    labels[outside>band]=cv2.GC_BGD;labels[inside>band]=cv2.GC_FGD
    path=np.rint(np.array(draft['flow'])*[w-1,h-1]).astype(np.int32)
    seeds=np.zeros_like(mask);cv2.polylines(seeds,[path],False,255,3)
    labels[(seeds>0)&(mask>0)]=cv2.GC_FGD
    labels[protected>0]=cv2.GC_BGD
    cv2.grabCut(generated,labels,None,np.zeros((1,65),float),np.zeros((1,65),float),4,cv2.GC_INIT_WITH_MASK)
    refined=np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)*255
    refined[outside>band]=0;refined[inside>band]=255;refined[protected>0]=0
    # Remove isolated classifications, without filling protected holes.
    count,components,stats,_=cv2.connectedComponentsWithStats(refined)
    for i in range(1,count):
        if stats[i,cv2.CC_STAT_AREA]<25:refined[components==i]=0
    distance=cv2.distanceTransform((refined>0).astype(np.uint8),cv2.DIST_L2,5)
    alpha=np.clip(distance/1.5,0,1)[:,:,None]
    replacement=np.rint(generated*alpha+dry*(1-alpha)).clip(0,255).astype(np.uint8)
    edge_band=(inside<=band)&(outside<=band)
    result=original.copy();result[edge_band]=replacement[edge_band]
    # Existing foreground objects retain the original, not a regenerated rock.
    result[protected>0]=original[protected>0]
    overlay=result.copy();cv2.polylines(overlay,[polygon],True,(0,150,255),2)
    contours,_=cv2.findContours(refined,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay,contours,-1,(40,220,70),1)
    cv2.imwrite(str(output/'wet-first-frame.png'),result)
    cv2.imwrite(str(output/'water-mask.png'),refined)
    cv2.imwrite(str(output/'boundary-overlay.jpg'),overlay)
    cv2.imwrite(str(output/'original-mask.png'),mask)
    (output/'user-region.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2),encoding='utf8')
    record={'scene_id':row['scene_id'],'name':row['name'],'source':str(source),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'geometry_source':str(geometry_path),
        'geometry_sha256':hashlib.sha256(geometry_path.read_bytes()).hexdigest(),'band_pixels':band,
        'method':'GrabCut in local uncertainty band with original interior, flow seeds and protected-object exclusions',
        'user_geometry_modified':False,'changed_mask_pixels':int(np.count_nonzero(mask!=refined)),
        'outside_edge_band_pixel_difference':int(np.abs(result.astype(int)-original.astype(int))[~edge_band].max()),
        'protected_object_pixel_difference':int(np.abs(result.astype(int)-original.astype(int))[protected>0].max()) if protected.any() else 0,
        'status':'requires_visual_review','user_accepted':False,'training_ready':False}
    (output/'boundary-record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scene');p.add_argument('--band',type=int,default=24)
    a=p.parse_args()
    for row in json.loads(a.manifest.read_text('utf8')):
        if row['scene_id'] in ('camera-e3e5e0945cbc','camera-c9c07aee7f28'):continue
        if a.scene and row['scene_id']!=a.scene:continue
        prepare(row,a.output/row['scene_id']/'input',a.band)
