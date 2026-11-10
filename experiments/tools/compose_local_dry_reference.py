"""Build explicit fluid masks and composite built-in generated patches onto sources.

Only mask construction, cropping, deterministic compositing and evidence reporting
are performed here. All replacement texture must come from image_gen.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def make_mask(source, config, generated_canvas=None):
    a=np.asarray(source).astype(np.int16)
    region=Image.new('L', source.size)
    d=ImageDraw.Draw(region)
    for p in config['polygons']:
        d.polygon([tuple(x) for x in p], fill=255)
    for stroke in config.get('region_strokes',[]):
        d.line([tuple(x) for x in stroke['points']], fill=255, width=stroke['width'], joint='curve')
    selection=np.asarray(region)>0
    if config.get('color') == 'tan':
        selection &= (a[:,:,0]-a[:,:,2]>config.get('r_b_min',10)) & (a[:,:,1]-a[:,:,2]>config.get('g_b_min',5)) & (a[:,:,0]>75)
    if config.get('color') == 'tan_or_bright':
        selection &= (((a[:,:,0]-a[:,:,2]>config.get('r_b_min',4)) & (a[:,:,1]-a[:,:,2]>config.get('g_b_min',2))) | (a.mean(axis=2)>config.get('brightness_min',165)))
    if config.get('color') == 'bright':
        selection &= (a.mean(axis=2)>config.get('brightness_min',145)) & (a.max(axis=2)-a.min(axis=2)<config.get('chroma_max',65))
    if config.get('generation_difference_min') is not None and generated_canvas is not None:
        smooth_source=np.asarray(source.filter(ImageFilter.GaussianBlur(2))).astype('float32').mean(axis=2)
        smooth_generated=np.asarray(generated_canvas.filter(ImageFilter.GaussianBlur(2))).astype('float32').mean(axis=2)
        selection &= (smooth_source-smooth_generated)>config['generation_difference_min']
    if config.get('min_component_pixels',0):
        from scipy.ndimage import label
        components,count=label(selection)
        sizes=np.bincount(components.ravel())
        keep=sizes>=config['min_component_pixels'];keep[0]=False
        selection=keep[components]
    mask=Image.fromarray((selection*255).astype('uint8'))
    if config.get('close_radius',0):
        size=config['close_radius']*2+1
        mask=mask.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
    if config.get('dilate',0):
        mask=mask.filter(ImageFilter.MaxFilter(config['dilate']*2+1))
    # Additional small fluid areas can be manually traced where color is ambiguous.
    d=ImageDraw.Draw(mask)
    for stroke in config.get('include_strokes',[]):
        d.line([tuple(x) for x in stroke['points']], fill=255, width=stroke['width'], joint='curve')
    for p in config.get('include_polygons',[]):
        d.polygon([tuple(x) for x in p], fill=255)
    for extra in config.get('include_color_regions',[]):
        area=Image.new('L',source.size)
        ImageDraw.Draw(area).polygon([tuple(x) for x in extra['polygon']],fill=255)
        hit=(np.asarray(area)>0) & (a[:,:,0]-a[:,:,1]>extra.get('r_g_min',3)) & (a[:,:,0]-a[:,:,2]>extra.get('r_b_min',15)) & (a[:,:,1]-a[:,:,2]>extra.get('g_b_min',8))
        extra_mask=Image.fromarray((hit*255).astype('uint8')).filter(ImageFilter.MaxFilter(3))
        mask=Image.fromarray(np.maximum(np.asarray(mask),np.asarray(extra_mask)))
        d=ImageDraw.Draw(mask)
    for p in config.get('exclude_polygons',[]):
        d.polygon([tuple(x) for x in p], fill=0)
    if config.get('preserve_white_overlay',False):
        m=np.array(mask)
        # Restrict this to known text rectangles, never remove white water globally.
        for x0,y0,x1,y1 in config.get('overlay_rectangles',[]):
            b=a[y0:y1,x0:x1]
            white=(b.min(axis=2)>190) & (b.max(axis=2)-b.min(axis=2)<30)
            m[y0:y1,x0:x1][white]=0
            if config.get('preserve_black_overlay',False):
                black=(b.max(axis=2)<90) & (b.max(axis=2)-b.min(axis=2)<35)
                m[y0:y1,x0:x1][black]=0
        mask=Image.fromarray(m)
    return mask

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    p.add_argument('--generated',type=Path)
    args=p.parse_args()
    args.config=args.config.resolve()
    c=json.loads(args.config.read_text(encoding='utf-8-sig'))
    out=ROOT/c['directory'];out.mkdir(parents=True,exist_ok=True)
    src_path=ROOT/c['source_path']; src=Image.open(src_path).convert('RGB')
    crop=tuple(c['crop'])
    generated_canvas=None
    if args.generated:
        gen=Image.open(args.generated).convert('RGB')
        patch=gen.resize((crop[2]-crop[0],crop[3]-crop[1]),Image.Resampling.LANCZOS)
        if c.get('tone_offset_rgb'):
            patch=Image.fromarray(np.clip(np.asarray(patch).astype('int16')+np.array(c['tone_offset_rgb']),0,255).astype('uint8'))
        generated_canvas=src.copy();generated_canvas.paste(patch,(crop[0],crop[1]))
    mask=make_mask(src,c,generated_canvas)
    src.save(out/'source.png');mask.save(out/'fluid-mask.png')
    crop=tuple(c['crop']);src.crop(crop).save(out/'generation-input.png')
    overlay=np.asarray(src).copy();selected=np.asarray(mask)>0
    overlay[selected]=(0.4*overlay[selected]+0.6*np.array([255,0,130])).astype('uint8')
    Image.fromarray(overlay).save(out/'mask-overlay.png')
    if not args.generated:return
    gen=Image.open(args.generated).convert('RGB')
    if args.generated.resolve() != (out/'raw-generation.png').resolve():
        shutil.copyfile(args.generated,out/'raw-generation.png')
    patch=gen.resize((crop[2]-crop[0],crop[3]-crop[1]),Image.Resampling.LANCZOS)
    if c.get('tone_offset_rgb'):
        patch=Image.fromarray(np.clip(np.asarray(patch).astype('int16')+np.array(c['tone_offset_rgb']),0,255).astype('uint8'))
    generated_canvas=src.copy();generated_canvas.paste(patch,(crop[0],crop[1]))
    alpha=mask
    if c.get('feather_inward',0):
        # Feather only inside the declared mask, so outside-mask pixels remain exact.
        alpha=Image.fromarray(np.minimum(np.asarray(mask),np.asarray(mask.filter(ImageFilter.GaussianBlur(c['feather_inward'])))))
    alpha.save(out/'composition-alpha.png')
    final=Image.composite(generated_canvas,src,alpha);final.save(out/'dry-reference.png')
    delta=np.abs(np.asarray(final).astype('int16')-np.asarray(src).astype('int16'))
    outside=~selected
    comparison=Image.new('RGB',(src.width*2,src.height));comparison.paste(src);comparison.paste(final,(src.width,0));comparison.save(out/'comparison.png')
    stats={'mask_pixels':int(selected.sum()),'mask_fraction':float(selected.mean()),'outside_mask_changed_pixels':int(np.any(delta[outside]>0,axis=1).sum()),'outside_mask_max_difference':int(delta[outside].max(initial=0)),'inside_mask_mean_absolute_difference':float(delta[selected].mean())}
    record={'schema_version':2,'status':c.get('status','generated_candidate_pending_user_acceptance'),'needs_revision':c.get('needs_revision',False),'residual_fluid':c.get('residual_fluid'),'root_review':c.get('root_review'),'accepted':False,'scene_id':c['scene_id'],'camera':c['camera'],'source_path':c['source_path'],'source_sha256':digest(src_path),'mask_path':(out/'fluid-mask.png').relative_to(ROOT).as_posix(),'mask_sha256':digest(out/'fluid-mask.png'),'alpha_path':(out/'composition-alpha.png').relative_to(ROOT).as_posix(),'alpha_sha256':digest(out/'composition-alpha.png'),'raw_generation_path':(out/'raw-generation.png').relative_to(ROOT).as_posix(),'raw_generation_sha256':digest(out/'raw-generation.png'),'output_path':(out/'dry-reference.png').relative_to(ROOT).as_posix(),'output_sha256':digest(out/'dry-reference.png'),'method':'Built-in image_gen local patch edit, then exact source retention outside an explicit fluid mask. Replacement bed is a synthetic hypothesis, not recovered ground truth.','mask_config_path':args.config.relative_to(ROOT).as_posix(),'crop_xyxy':crop,'tone_offset_rgb':c.get('tone_offset_rgb',[0,0,0]),'generation_prompt':c['prompt'],'generation_tool':'built-in image_gen','built_in_artifact_path':c.get('built_in_artifact_path',str(args.generated)),'mask_note':c.get('mask_note'),'tone_matching_method':c.get('tone_matching_method'),'temporal_review':c.get('temporal_review'),'limitations':c.get('limitations',[]),'revision_note':c.get('revision_note'),'command':['.venv/Scripts/python.exe','experiments/tools/compose_local_dry_reference.py',args.config.relative_to(ROOT).as_posix(),'--generated',(out/'raw-generation.png').relative_to(ROOT).as_posix()],'pixel_checks':stats,'assistant_visual_review':c.get('assistant_visual_review','pending'),'user_acceptance':'pending','camera_role':None,'training_ready':False}
    write_json(out/'composition.json',record)
    print(json.dumps(stats))

if __name__=='__main__':main()
