"""Assemble image_gen water pixels with unchanged source background.

This does not invent or animate water. The only replacement pixels come from
the built-in image edit. The mask bounds a local data-composition operation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.common.video.synthetic_wan import atomic_json,utc_now
from experiments.common.video.minimax_preview_workflow import digest,read_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generated',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--region',type=Path,help='Frozen user-region JSON; use exact polygon without dilation.')
    args=parser.parse_args()
    source=ROOT/'data/process/synthetic-video/dry-scenes-20261108-v2/camera-c9c07aee7f28/dry-reference.png'
    prior_mask=source.with_name('fluid-mask.png')
    args.output.mkdir(parents=True,exist_ok=True)
    target=args.output/'wet-first-frame.png'
    if target.exists():raise ValueError('Existing candidate must not be overwritten.')
    base=np.asarray(Image.open(source).convert('RGB'))
    generated=np.asarray(Image.open(args.generated).convert('RGB').resize((base.shape[1],base.shape[0]),Image.Resampling.LANCZOS if hasattr(Image,'Resampling') else Image.LANCZOS))
    if args.region:
        region=read_json(args.region)['draft']['polygon']
        mask=np.zeros(base.shape[:2],np.uint8)
        cv2.fillPoly(mask,[np.array([[min(base.shape[1]-1,round(x*base.shape[1])),min(base.shape[0]-1,round(y*base.shape[0]))] for x,y in region],np.int32)],255)
    else:
        mask=np.asarray(Image.open(prior_mask).convert('L'))
        mask=cv2.dilate(mask,np.ones((49,49),np.uint8))
    mask[:230,:]=0
    mask[390:540,535:960]=0  # Independent pond and its liner are protected.
    mask[448:534,206:220]=0  # Existing foreground fixed poles.
    mask[440:482,436:450]=0
    alpha=np.minimum(cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,3)/6.,1.)
    mixed=np.rint(generated*alpha[:,:,None]+base*(1-alpha[:,:,None])).astype(np.uint8)
    Image.fromarray(mixed).save(target)
    Image.fromarray(mask).save(args.output/'edit-region.png')
    delta=np.any(mixed!=base,axis=2)
    outside=int(delta[mask==0].sum())
    pond=int(delta[390:540,535:960].sum())
    assert outside==0 and pond==0
    record={'created_at':utc_now(),'source':source.relative_to(ROOT).as_posix(),'source_sha256':digest(source),
            'generated':args.generated.resolve().relative_to(ROOT).as_posix(),'generated_sha256':digest(args.generated),
            'output':target.resolve().relative_to(ROOT).as_posix(),'output_sha256':digest(target),
            'source_dimensions':[base.shape[1],base.shape[0]],'local_region_margin_pixels':0 if args.region else 24,'edge_blend_inside_region_pixels':6,
            'user_region':args.region.resolve().relative_to(ROOT).as_posix() if args.region else None,
            'user_region_sha256':digest(args.region) if args.region else None,
            'outside_edit_region_changed_pixels':outside,'independent_pond_changed_pixels':pond,
            'creative_water_pixels':'built-in image_gen only','composition':'source pixels retained outside bounded river edit region',
            'status':'assistant_spatial_review_pending','training_ready':False}
    atomic_json(args.output/'composition.json',record)
    print(target.resolve());print('outside-region changed pixels:',outside,'pond changed pixels:',pond)


if __name__=='__main__':main()
