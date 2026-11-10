"""Recompose existing built-in water pixels through one continuous channel envelope."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
import numpy as np
from PIL import Image,ImageDraw,ImageFilter

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'data/process/synthetic-video/dry-scenes-20261108-v2').is_dir())
D=ROOT/'data/process/synthetic-video/ordinary-water-20261108-v4/camera-936ae8691cee';R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def smooth_closed(points,n=12):
 q=[];pts=[np.asarray(x,dtype=float) for x in points]
 for i in range(len(pts)):
  p0,p1,p2,p3=[pts[j%len(pts)] for j in [i-1,i,i+1,i+2]]
  for t in np.linspace(0,1,n,endpoint=False):
   v=.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t*t*t)
   q.append(tuple(v))
 return q

sourcep=ROOT/'data/process/synthetic-video/dry-scenes-20261108-v2/camera-936ae8691cee/dry-reference.png'
src=Image.open(sourcep).convert('RGB');rawp=D/'ordinary-water-raw-generation.png'
raw=Image.open(rawp).convert('RGB').resize((1090,660),Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(.75))
canvas=src.copy();canvas.paste(raw,(190,60))
# Only the generated-caption footprint receives the single local cleanup edit.
# Original caption glyphs are restored from src below; channel geometry is reused.
cleanup=Image.open(R/'caption-cleanup-raw-generation.png').convert('RGB').resize((330,160),Image.Resampling.LANCZOS)
clean_canvas=canvas.copy();clean_canvas.paste(cleanup,(950,560));text_area=Image.new('L',src.size)
ImageDraw.Draw(text_area).rounded_rectangle((1032,648,1180,698),radius=4,fill=255)
text_area=text_area.filter(ImageFilter.GaussianBlur(2.0));canvas=Image.composite(clean_canvas,canvas,text_area)
# Envelope is deliberately continuous and slightly OUTSIDE the natural water banks
# visible in the original built-in raw; the water/gravel boundary comes from the raw,
# rather than clipping it with the rejected fragmented fluid support.
main=[(630,82),(579,99),(535,119),(526,151),(554,182),(611,212),(675,244),(730,279),(794,313),(856,347),(918,383),(975,422),(1027,462),(1087,503),(1158,543),(1226,583),(1310,627),(1320,755),(1206,755),(1144,704),(1082,664),(1017,625),(958,588),(909,551),(861,515),(817,474),(770,440),(722,413),(674,379),(627,348),(576,323),(524,295),(470,261),(426,231),(395,204),(393,165),(410,136),(455,111),(502,99),(549,90)]
branch=[(671,350),(716,373),(684,411),(644,447),(614,483),(584,526),(571,572),(571,617),(578,663),(572,747),(456,747),(457,679),(466,635),(473,590),(482,544),(510,499),(544,458),(566,417),(593,386),(629,363)]
scale=4;hi=Image.new('L',(src.width*scale,src.height*scale));draw=ImageDraw.Draw(hi)
for points in [main,branch]:draw.polygon([(round(x*scale),round(y*scale)) for x,y in smooth_closed(points)],fill=255)
mask=hi.resize(src.size,Image.Resampling.LANCZOS)
# Remove tiny anti-alias ringing from support so outside-mask proof is meaningful.
ma=np.asarray(mask).copy();ma[ma<4]=0;mask=Image.fromarray(ma)
alpha=Image.fromarray(np.minimum(ma,np.asarray(mask.filter(ImageFilter.GaussianBlur(3.0)))))
caption=np.zeros((src.height,src.width),dtype='uint8');a=np.asarray(src).astype('int16');x0,y0,x1,y1=(1042,655,1156,684);b=a[y0:y1,x0:x1]
glyph=((b.min(axis=2)>190)&(b.max(axis=2)-b.min(axis=2)<30))|((b.max(axis=2)<90)&(b.max(axis=2)-b.min(axis=2)<35))
caption[y0:y1,x0:x1]=glyph.astype('uint8')*255
cm=Image.fromarray(caption).filter(ImageFilter.MaxFilter(3));cp=np.asarray(cm)>0
aa=np.asarray(alpha).copy();mm=np.asarray(mask).copy();aa[cp]=0;mm[cp]=0;alpha=Image.fromarray(aa);mask=Image.fromarray(mm)
mask.save(D/'ordinary-water-mask.png');alpha.save(D/'ordinary-water-alpha.png');cm.save(R/'protected-caption-glyphs.png')
out=Image.composite(canvas,src,alpha);out.save(D/'ordinary-water-reference.png')
pair=Image.new('RGB',(src.width*2,src.height));pair.paste(src);pair.paste(out,(src.width,0));pair.save(D/'ordinary-water-comparison.png')
box=(390,250,1280,720);p=Image.new('RGB',((box[2]-box[0])*2,box[3]-box[1]));p.paste(src.crop(box));p.paste(out.crop(box),(box[2]-box[0],0));p.save(D/'ordinary-water-detail-comparison.png')
out.crop((550,300,1280,720)).save(R/'revised-channel-detail.png')
ov=np.asarray(src).copy();sel=mm>0;ov[sel]=(ov[sel]*.35+np.array([255,0,170])*.65).astype('uint8');Image.fromarray(ov).save(D/'ordinary-water-mask-overlay.png')
delta=np.abs(np.asarray(out).astype('int16')-a)
checks={'mask_pixels':int(sel.sum()),'mask_fraction':float(sel.mean()),'outside_mask_changed_pixels':int(np.any(delta[~sel]>0,axis=1).sum()),'outside_mask_max_difference':int(delta[~sel].max(initial=0)),'protected_caption_glyph_changed_pixels':int(np.any(delta[cp]>0,axis=1).sum()),'protected_mountain_vegetation_timestamp_changed_pixels':int(np.any(delta[:80]>0,axis=2).sum())}
assert checks['outside_mask_changed_pixels']==0 and checks['protected_caption_glyph_changed_pixels']==0
record={'revision':'continuous-channel-20261109','reason':'root rejected fragmented_mask_edges','prior_output_path':rel(D/'revision/failed-20261109-fragmented-mask/ordinary-water-reference.png'),'prior_output_sha256':sha(D/'revision/failed-20261109-fragmented-mask/ordinary-water-reference.png'),'raw_generation_reused':rel(rawp),'raw_generation_sha256':sha(rawp),'new_image_generation_calls':1,'caption_cleanup':{'tool':'built-in image_gen','input_path':rel(R/'caption-cleanup-input.png'),'input_sha256':sha(R/'caption-cleanup-input.png'),'prompt_path':rel(R/'caption-cleanup-prompt.json'),'raw_path':rel(R/'caption-cleanup-raw-generation.png'),'raw_sha256':sha(R/'caption-cleanup-raw-generation.png'),'built_in_artifact_path':'C:/Users/20443/.codex/generated_images/01a1bbdd-12c1-7430-82d9-e8780e761ead/exec-60818aed-b3f6-4679-96b1-fcd19e1b0bec.png','applied_footprint_xyxy':[1032,648,1180,698],'purpose':'Remove generated-caption ghost before restoring original exact glyph pixels.'},'source_sha256':sha(sourcep),'output_sha256':sha(D/'ordinary-water-reference.png'),'main_envelope_points':main,'branch_envelope_points':branch,'envelope_smoothing':'Closed Catmull-Rom, 12 samples per segment; 4x supersampled rasterization','alpha_feather_px':3,'patch_blur_px':.75,'old_fluid_mask_used':False,'method':'Existing natural liquid pixels composited through a continuous hand-bounded channel envelope; no intersection with the rejected fragmented support. Broad dry bars, slopes, meadow, road and camera text remain source pixels outside this envelope.','pixel_checks':checks,'created_at':datetime.now(timezone.utc).isoformat(),'script_path':rel(__file__),'script_sha256':sha(__file__),'status':'repair_pending_root_review'}
save(R/'revision-record.json',record)
m=json.loads((D/'ordinary-water-metadata.json').read_text(encoding='utf-8'))
m.update(status='assistant_repair_pending_root_review',needs_revision=True,ref_issue='fragmented_mask_edges',revision_pending_root_review=True,training_ready=False,user_accepted=False,pixel_checks=checks,mask_method=record['method'],mask_sha256=sha(D/'ordinary-water-mask.png'),alpha_sha256=sha(D/'ordinary-water-alpha.png'),output_sha256=sha(D/'ordinary-water-reference.png'),code_sha256=sha(__file__),command=['.venv/Scripts/python.exe',rel(__file__)])
m['composition_revision']=record;m['assistant_visual_review']={'status':'repair_image_pending_visual_inspection','full_frame_viewed':False,'channel_detail_viewed':False};m['root_review']={'status':'recheck_pending','previous_status':'rejected','previous_issue':'fragmented_mask_edges'}
m['limitations']=['Static synthetic frame does not verify physical flow or direction.','Source is out of focus; source background remains unsharpened.','Continuous envelope contains some adjacent synthetic bed pixels as well as water; it was drawn from the original natural-water generation, not the rejected fragmented support.','User acceptance remains pending; training_ready=false.']
for n in ['composition.json','ordinary-water-metadata.json']:save(D/n,m)
print(json.dumps({'output':str(D/'ordinary-water-reference.png'),'old_sha256':record['prior_output_sha256'],'new_sha256':record['output_sha256'],'pixel_checks':checks},ensure_ascii=False))
