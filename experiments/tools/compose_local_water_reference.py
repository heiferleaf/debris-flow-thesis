"""Prepare and compose the two local water first-frame demonstrations.

Replacement pixels must come from the built-in image_gen tool. This script only
constructs bounded masks, crops, copies source pixels and writes provenance.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'data/process/synthetic-video/dry-scenes-20261108-v2/camera-c9c07aee7f28/dry-reference.png'
OLD_MASK=ROOT/'data/process/synthetic-video/dry-scenes-20261108-v2/camera-c9c07aee7f28/fluid-mask.png'
OUTPUT=ROOT/'data/process/synthetic-video/water-reference-demo-20261108-v3/camera-c9c07aee7f28'
CROP=(0,200,960,540)

COMMON='''Use case: precise-object-edit. Image 1 is the EDIT TARGET, a crop of an accepted dry-bed photograph from a fixed surveillance camera. Image 2 is a SUPPORTING LOCATION GUIDE: its magenta tint identifies the only thin channel footprints to receive water. The magenta is annotation, never a desired water color. Put the new water precisely inside the guided footprints and retain every grey gravel island between them. Preserve the exact terrain, channel banks, road, slopes, fields, plants, poles, text and camera geometry. The separate WHITE-LINED POND at lower right must keep exactly its original green still water and white liner. It is not part of the stream. Same overcast hazy light and low-resolution surveillance appearance, no cinematic lighting, no oversharpening. Do not move shorelines, dig channels, reshape banks, flood the gravel islands, or alter the pond. No yellow or ochre flow, no tropical blue/green river, no magenta, no muddy solid clumps, no debris-flow lobes, no transported gravel or stones, no new objects. Preserve image framing and crop dimensions exactly. '''
PROMPTS={
 'ordinary':COMMON+'''Add a NORMAL SMALL SHALLOW STREAM, not a flood. Use narrow interconnecting braided rivulets occupying only the magenta-guided threads, separated by the existing exposed grey gravel. Clear colorless water with natural grey reflections of the overcast sky; the underlying dark sand and pebbles are visible THROUGH the water. A thin transparent surface with coherent small ripples and only very sparse small white foam flecks. Fluid water with normal low viscosity. No beige slurry, no yellow tint, no opaque muddy paste, no large waves. The many unmarked parts of the original channel must remain dry.''',
 'flood':COMMON+'''Add a NATURAL RIVER FLOOD confined to the guided low braided channel. This has more depth and a wider, more continuous water surface than a small ordinary stream while the original exposed gravel islands remain visible. Natural neutral GREY turbid water carrying fine suspended silt; do not use a uniform yellow/brown/ochre color. The surface is unmistakably fluid: coherent current-aligned ripples, small turbulent waves and irregular thin white foam streaks on a continuous liquid surface. The surface can be opaque with fine suspended sediment but never contains granular mud clumps, rolling rocks, debris masses or a rough solid front. No debris-flow movement. Do not raise the water above the original banks or into the independent pond.'''
}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rel(p): return Path(p).resolve().relative_to(ROOT).as_posix()

def masks():
    allowed=Image.open(OLD_MASK).convert('L')
    narrow=Image.new('L',allowed.size);d=ImageDraw.Draw(narrow)
    # Each line follows a subdivision of the observed original low braided route.
    paths=[
      ([(0,505),(45,510),(87,501),(127,483),(160,453),(180,437),(216,415),(255,400),(288,381),(328,368),(366,364),(404,357),(444,350),(486,345),(528,335),(560,321),(580,305),(598,289)],9),
      ([(0,483),(39,478),(75,472),(104,460),(130,441),(159,432),(183,420),(216,409),(250,393),(285,381),(314,369),(355,359),(397,354),(444,350)],5),
      ([(173,441),(213,439),(248,426),(281,418),(312,407),(341,392),(365,378),(404,365),(444,350)],5),
      ([(598,289),(634,291),(674,291),(711,293),(734,284),(756,273),(790,272),(831,272),(871,275),(914,280),(960,283)],4),
      ([(711,293),(749,294),(789,297),(830,295),(872,289),(918,291),(960,294)],3),
    ]
    for pts,width in paths:d.line(pts,fill=255,width=width,joint='curve')
    ordinary=Image.fromarray(np.minimum(np.asarray(narrow),np.asarray(allowed)))
    flood=allowed.copy()
    # Exact source retention for the independent reservoir and fixed objects.
    protected=[[(535,390),(960,390),(960,540),(535,540)],[(206,448),(219,448),(219,533),(206,533)],[(436,440),(449,440),(449,481),(436,481)]]
    for m in [ordinary,flood]:
      dd=ImageDraw.Draw(m)
      for polygon in protected:dd.polygon(polygon,fill=0)
    return {'ordinary':ordinary,'flood':flood},paths,protected

def run(kind,generated=None):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source=Image.open(SOURCE).convert('RGB');all_masks,paths,protected=masks();mask=all_masks[kind]
    source.save(OUTPUT/'accepted-dry-source.png')
    source.crop(CROP).save(OUTPUT/'generation-input.png')
    mask.save(OUTPUT/f'{kind}-mask.png')
    overlay=np.asarray(source).copy();selected=np.asarray(mask)>0
    overlay[selected]=(overlay[selected]*.4+np.array([255,0,170])*.6).astype('uint8')
    Image.fromarray(overlay).save(OUTPUT/f'{kind}-mask-overlay.png')
    Image.fromarray(overlay).crop(CROP).save(OUTPUT/f'{kind}-generation-guide.png')
    save(OUTPUT/f'{kind}-prompt.json',{'kind':kind,'prompt':PROMPTS[kind],'crop_xyxy':CROP,'mask_paths_ordinary':paths,'protected_polygons':protected})
    if generated is None:return
    raw_target=OUTPUT/f'{kind}-raw-generation.png'
    if generated.resolve()!=raw_target.resolve():shutil.copyfile(generated,raw_target)
    patch=Image.open(raw_target).convert('RGB').resize((CROP[2]-CROP[0],CROP[3]-CROP[1]),Image.Resampling.LANCZOS)
    canvas=source.copy();canvas.paste(patch,(CROP[0],CROP[1]))
    # Only soften the inward edge; every pixel outside the mask stays identical.
    alpha=Image.fromarray(np.minimum(np.asarray(mask),np.asarray(mask.filter(ImageFilter.GaussianBlur(.45)))))
    alpha.save(OUTPUT/f'{kind}-alpha.png')
    final=Image.composite(canvas,source,alpha);dest=OUTPUT/f'{kind}-reference.png';final.save(dest)
    pair=Image.new('RGB',(source.width*2,source.height));pair.paste(source);pair.paste(final,(source.width,0));pair.save(OUTPUT/f'{kind}-comparison.png')
    delta=np.abs(np.asarray(final).astype('int16')-np.asarray(source).astype('int16'))
    outside=~selected
    record={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),'scene_id':'camera-c9c07aee7f28','camera':'鸡冠山下游','kind':kind,'status':'candidate_pending_root_and_user_review','source_path':rel(SOURCE),'source_sha256':sha(SOURCE),'source_acceptance':'User confirmed all 11 dry references in the current session, as relayed by root. This derivative has no user acceptance.','prompt':PROMPTS[kind],'generation_tool':'built-in image_gen','built_in_artifact_path':str(generated),'crop_xyxy':CROP,'mask_path':rel(OUTPUT/f'{kind}-mask.png'),'mask_sha256':sha(OUTPUT/f'{kind}-mask.png'),'alpha_path':rel(OUTPUT/f'{kind}-alpha.png'),'alpha_sha256':sha(OUTPUT/f'{kind}-alpha.png'),'raw_generation_path':rel(raw_target),'raw_generation_sha256':sha(raw_target),'output_path':rel(dest),'output_sha256':sha(dest),'mask_method':'Ordinary: hand-traced thin subdivisions intersected with original fluid-support mask. Flood: original fluid-support footprint, preserving exposed islands. Protected reservoir and fixed poles excluded in both.','pixel_checks':{'mask_pixels':int(selected.sum()),'mask_fraction':float(selected.mean()),'outside_mask_changed_pixels':int(np.any(delta[outside]>0,axis=1).sum()),'outside_mask_max_difference':int(delta[outside].max(initial=0)),'reservoir_rectangle_changed_pixels':int(np.any(delta[390:540,535:960]>0,axis=2).sum())},'physical_flow_verified':False,'video_api_calls':0,'paid_video_api_calls':0,'user_accepted':False,'training_ready':False,'assistant_visual_review':'pending','command':['.venv/Scripts/python.exe',rel(Path(__file__)),'--kind',kind,'--generated',rel(raw_target)],'code_sha256':sha(__file__)}
    save(OUTPUT/f'{kind}-metadata.json',record)
    print(json.dumps(record['pixel_checks']))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=list(PROMPTS),required=True);p.add_argument('--generated',type=Path);a=p.parse_args();run(a.kind,a.generated)
