"""Bounded, local controlled-fluid inference; never calls a paid service.

Uses the released SLR-SFS *baseline* encoder/decoder, with explicit Eulerian
motion and a CPU bilinear forward-splat. This is not the unpublished SFS solver
and is not a physically validated training-data generator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
COMMIT = 'fa8d61046dfa19d80c5605be4a64e71f499b58f7'
SOURCE = ROOT / 'data/results/synthetic-video/pretrained/slr-sfs-fa8d610/source' / ('SLR-SFS-' + COMMIT)
WEIGHTS = ROOT / 'data/results/synthetic-video/pretrained/slr-sfs-fa8d610/baseline2-motion2-fixedMotionFinetune.pth'
INPUT = ROOT / 'data/process/synthetic-video/user-water-region-20261109-v2/camera-c9c07aee7f28'
OUTPUT = ROOT / 'data/process/synthetic-video/controlled-flow-20261109-v1/camera-c9c07aee7f28'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(output, size=256, speed=1.0, highres_reference=False, input_dir=INPUT, boundary_tangent=False):
    """Reuse the user's polygon/path unchanged; speeds are preview assumptions."""
    output.mkdir(parents=True, exist_ok=True)
    input_dir=Path(input_dir).resolve()
    generic=input_dir != INPUT.resolve()
    source = input_dir / 'wet-first-frame.png'
    region_path = input_dir / 'user-region.json'
    region = json.loads(region_path.read_text(encoding='utf-8-sig'))['draft']
    image = cv2.imread(str(source))
    hd_path=input_dir/'generated-full.png'
    if highres_reference:
        hd=cv2.imread(str(hd_path))
        hd_h,hd_w=hd.shape[:2]
        hd_h-=hd_h%2;hd_w-=hd_w%2
        hd=cv2.resize(hd,(hd_w,hd_h))
        image=cv2.resize(image,(hd_w,hd_h),interpolation=cv2.INTER_CUBIC)
    h, w = image.shape[:2]
    if h%2 or w%2:
        image=cv2.resize(image,(w-w%2,h-h%2));h,w=image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    boundary = np.rint(np.array(region['polygon']) * [w-1, h-1]).astype(np.int32)
    cv2.fillPoly(mask, [boundary], 255)
    if highres_reference:
        blend=np.minimum(cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,3)/(6*w/960),1)[:,:,None]
        image=np.rint(hd*blend+image*(1-blend)).astype(np.uint8)
    # Conservative fixed-object exclusions, visually located on the 960x540
    # input. These are independent draft controls, not edits to user geometry.
    exclusions = [
        [[201,450],[211,446],[224,539],[202,539]],
        [[430,445],[447,443],[447,480],[431,480]],
        [[411,380],[426,376],[434,390],[414,396]],
        [[365,468],[389,466],[409,519],[362,526]],
    ]
    if generic:
        exclusions=[(np.asarray(p)*[960,540]).tolist() for p in region.get('exclusions',[])]
    for polygon in exclusions:
        cv2.fillPoly(mask, [np.rint(np.array(polygon)*[w/960.,h/540.]).astype(np.int32)], 0)
    local_mask = cv2.resize(mask, (size,size), interpolation=cv2.INTER_NEAREST) / 255.
    path = np.array(region['flow'], dtype=np.float32) * (size-1)
    yy, xx = np.mgrid[:size,:size].astype(np.float32)
    accumulated = np.zeros((size,size,2),np.float32)
    denominator = np.zeros((size,size),np.float32)
    hints = []
    for i in range(len(path)-1):
        point = (path[i]+path[i+1])*.5
        tangent = path[i+1]-path[i]
        tangent /= max(float(np.linalg.norm(tangent)),1e-6)
        perspective_profile=not generic or region.get('flow_profile')=='original_perspective'
        magnitude = ((.10 + .35*(1-i/max(len(path)-2,1))) if perspective_profile else .275) * speed * size/256.
        if region.get('flow_profile')=='image_y_perspective':
            near_fraction=float(np.clip((point[1]-path[:,1].min())/max(float(np.ptp(path[:,1])),size*.15),0,1))
            magnitude=(.10+.35*near_fraction)*speed*size/256.
        vector = tangent*magnitude
        weights = np.exp(-((xx-point[0])**2+(yy-point[1])**2)/(size*.13)**2)
        accumulated += weights[:,:,None]*vector
        denominator += weights
        hints.append({'xy_normalized':(point/(size-1)).tolist(),
                      'velocity_pixels_per_frame_at_input_size':vector.tolist()})
    flow = accumulated/np.maximum(denominator[:,:,None],1e-12)
    # Slow the imposed motion at the boundary. This is an artistic constraint,
    # not an inferred physical velocity or an incompressibility solution.
    edge = cv2.distanceTransform((local_mask>0).astype(np.uint8),cv2.DIST_L2,3)
    if boundary_tangent:
        # Geometry constraint only: remove velocity through the local bank or
        # obstacle boundary. This is not a mass/momentum-conserving simulation.
        smooth=cv2.GaussianBlur(edge,(0,0),1)
        ny,nx=np.gradient(smooth)
        norm=np.sqrt(nx*nx+ny*ny)+1e-6
        nx/=norm;ny/=norm
        strength=np.clip(1-edge/(size/32),0,1)**2
        dot=flow[:,:,0]*nx+flow[:,:,1]*ny
        flow[:,:,0]-=strength*dot*nx
        flow[:,:,1]-=strength*dot*ny
    flow *= (local_mask*np.minimum(edge/2.,1))[:,:,None]
    np.save(output/'motion.npy',flow)
    cv2.imwrite(str(output/'mask.png'),mask)
    preview = image.copy()
    for y in range(0,size,12):
        for x in range(0,size,12):
            if local_mask[y,x] and np.linalg.norm(flow[y,x])>.02:
                start=(round(x*w/size),round(y*h/size))
                end=(round((x+flow[y,x,0]*20)*w/size),round((y+flow[y,x,1]*20)*h/size))
                cv2.arrowedLine(preview,start,end,(10,200,255),1,tipLength=.25)
    cv2.polylines(preview,[boundary],True,(255,200,0),1)
    cv2.imwrite(str(output/'motion-controls.jpg'),preview)
    spec = {'method':'explicit Eulerian flow + pretrained learned feature renderer',
            'source':str(source.relative_to(ROOT)), 'source_sha256':sha(source),
            'user_geometry':str(region_path.relative_to(ROOT)), 'user_geometry_sha256':sha(region_path),
            'user_polygon_modified':False, 'preview_exclusions_pixels_960x540':exclusions,
            'preview_exclusions_author':region.get('exclusions_author','user drawn') if generic else 'assistant visual estimate; not user confirmed',
            'input_size':[size,size], 'speed_multiplier':speed, 'hints':hints,
            'upstream_commit':COMMIT, 'model':'released baseline2+motion2+fixedMotionFinetune encoder/projector only',
            'motion_regressor_used':False, 'SFS_solver_used':False,
            'physical_velocity_measured':False,'physical_simulation':False,
            'boundary_tangent_projection':boundary_tangent,
            'highres_water_reference':str(hd_path.relative_to(ROOT)) if highres_reference else None,
            'highres_water_sha256':sha(hd_path) if highres_reference else None,
            'background_upscaled':highres_reference,
            'training_ready':False, 'paid_requests':0}
    save_json(output/'input.json',spec)
    return image,mask,flow,spec


def splat(values, displacement):
    """Bilinear forward sum-splat, equivalent operation to CUDA sum-splat."""
    b,c,h,w=values.shape
    y,x=torch.meshgrid(torch.arange(h,device=values.device),torch.arange(w,device=values.device),indexing='ij')
    px=x[None]+displacement[:,0]
    py=y[None]+displacement[:,1]
    x0,y0=px.floor().long(),py.floor().long()
    result=torch.zeros_like(values).flatten(2)
    for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
        xi,yi=x0+dx,y0+dy
        valid=(xi>=0)&(xi<w)&(yi>=0)&(yi<h)
        weight=(1-(px-xi).abs())*(1-(py-yi).abs())*valid
        index=(yi.clamp(0,h-1)*w+xi.clamp(0,w-1)).flatten(1)
        result.scatter_add_(2,index[:,None].expand(b,c,-1),(values*weight[:,None]).flatten(2))
    return result.view_as(values)


def displacements(flow, count, keep=None):
    """Nearest-coordinate Euler integration, matching the released baseline."""
    _,_,h,w=flow.shape
    y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    original=torch.stack([x,y]).float()
    current=original.clone()
    invalid=torch.zeros((h,w),dtype=torch.bool)
    results={0:torch.zeros_like(flow)} if keep is not None else [torch.zeros_like(flow)]
    for step in range(1,count+1):
        index=current.round().long()
        current+=flow[0,:,index[1],index[0]]
        invalid |= (current[0]<0)|(current[0]>w-1)|(current[1]<0)|(current[1]>h-1)
        current[:,invalid]=original[:,invalid]
        displacement=current-original
        displacement[:,invalid]=max(h,w)+1
        if keep is None:results.append(displacement[None].clone())
        elif step in keep:results[step]=displacement[None].clone()
    return results


class DisplacementBlocks:
    """Exact integration checkpoints with one small cached block, not 289 fields."""
    def __init__(self,flow,count,block=16):
        self.flow=flow;self.count=count;self.block=block
        _,_,self.h,self.w=flow.shape
        y,x=torch.meshgrid(torch.arange(self.h),torch.arange(self.w),indexing='ij')
        self.original=torch.stack([x,y]).float()
        current=self.original.clone();invalid=torch.zeros((self.h,self.w),dtype=torch.bool)
        self.checkpoints={0:(current.clone(),invalid.clone())}
        for step in range(1,count+1):
            self.advance(current,invalid)
            if step%block==0:self.checkpoints[step]=(current.clone(),invalid.clone())
        self.cache={};self.cache_start=None

    def advance(self,current,invalid):
        index=current.round().long()
        current+=self.flow[0,:,index[1],index[0]]
        invalid |= (current[0]<0)|(current[0]>self.w-1)|(current[1]<0)|(current[1]>self.h-1)
        current[:,invalid]=self.original[:,invalid]

    def __getitem__(self,step):
        if not 0<=step<=self.count:raise IndexError(step)
        start=(step//self.block)*self.block
        if start!=self.cache_start:
            current,invalid=(x.clone() for x in self.checkpoints[start])
            cache={}
            for position in range(start,min(start+self.block,self.count+1)):
                if position>start:self.advance(current,invalid)
                delta=current-self.original;delta[:,invalid]=max(self.h,self.w)+1
                cache[position]=delta[None].clone()
            self.cache=cache;self.cache_start=start
        return self.cache[step]


class Options(argparse.Namespace):
    def __contains__(self,key):
        return hasattr(self,key)


def load_renderer(size):
    sys.path.insert(0,str(SOURCE))
    from models.networks.utilities import get_encoder,get_decoder
    torch.serialization.add_safe_globals([argparse.Namespace])
    checkpoint=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    opts=Options(**vars(checkpoint['opts']))
    opts.W=size
    opts.bn_noise_misc=True
    assert 'skip_warp' not in opts.refine_model_type
    encoder=get_encoder(opts).eval()
    decoder=get_decoder(opts).eval()
    loaded={}
    for name,module in [('encoder',encoder),('projector',decoder)]:
        prefix='model.module.'+name+'.'
        state={k[len(prefix):]:v for k,v in checkpoint['state_dict'].items() if k.startswith(prefix)}
        if not state:
            raise ValueError('Unexpected checkpoint prefix for '+name)
        module.load_state_dict(state,strict=True)
        loaded[name]=len(state)
    return encoder,decoder,opts,loaded


def self_test():
    a=torch.arange(20,dtype=torch.float32).view(1,1,4,5)
    zero=torch.zeros(1,2,4,5)
    torch.testing.assert_close(splat(a,zero),a)
    shift=zero.clone();shift[:,0]=1
    torch.testing.assert_close(splat(a,shift)[:,:,:,1:],a[:,:,:,:-1])
    half=zero.clone();half[:,0]=.5
    torch.testing.assert_close(splat(a,half)[:,:,:,1:],(a[:,:,:,:-1]+a[:,:,:,1:])*.5)
    d=displacements(shift,2)
    assert d[2][0,0,1,1]==2 and d[2][0,0,1,4]>5
    print('PASS: zero, integer, fractional splat and integration boundary',flush=True)


def render(args):
    if not args.prepare_only and (args.output/'candidate.mp4').exists():
        raise ValueError('Use a new output directory for another preview.')
    torch.set_num_threads(args.threads)
    torch.manual_seed(7)
    image,mask,field,spec=prepare(args.output,args.size,args.speed,args.highres_reference,args.input_dir,args.boundary_tangent)
    if getattr(args,'water_mask',None):
        refined=cv2.imread(str(args.water_mask),0)
        if refined is None or refined.shape!=mask.shape:
            raise ValueError('Water mask must match the full reference image')
        mask=np.where(refined>127,255,0).astype(np.uint8)
        cv2.imwrite(str(args.output/'mask.png'),mask)
        spec.update(water_mask_override=str(args.water_mask),water_mask_sha256=sha(args.water_mask))
    if getattr(args,'flow_field',None):
        supplied=np.load(args.flow_field,allow_pickle=False)
        if supplied.shape!=(*image.shape[:2],2) or not np.isfinite(supplied).all():
            raise ValueError('Flow must be finite full-image HxWx2, in pixels/frame')
        field=cv2.resize(supplied.astype(np.float32),(args.size,args.size))
        field*=np.array([args.size/image.shape[1],args.size/image.shape[0]],np.float32)
        field*=cv2.resize(mask,(args.size,args.size),interpolation=cv2.INTER_NEAREST)[:,:,None]/255.
        np.save(args.output/'motion.npy',field)
        spec.update(flow_field_override=str(args.flow_field),flow_field_sha256=sha(args.flow_field),
                    flow_field_units='full-reference pixels per output frame',hints=[],
                    external_flow=True,physical_simulation='external field provenance, not asserted by renderer')
        # Replace the otherwise stale arrow preview produced by prepare().
        preview=image.copy()
        for y in range(0,image.shape[0],24):
            for x in range(0,image.shape[1],24):
                if mask[y,x]:
                    end=np.rint(np.array([x,y])+supplied[y,x]*12).astype(int)
                    cv2.arrowedLine(preview,(x,y),tuple(end),(10,200,255),1,tipLength=.25)
        cv2.imwrite(str(args.output/'motion-controls.jpg'),preview)
    save_json(args.output/'input.json',spec)
    if args.prepare_only:
        print(json.dumps({'prepared':str(args.output),'size':args.size}),flush=True)
        return
    start=time.time()
    full_h,full_w=image.shape[:2]
    if args.water_crop:
        ys,xs=np.where(mask>0)
        x0=max(0,int(xs.min())-16);x1=min(full_w,int(xs.max())+17)
        y0=max(0,int(ys.min())-16);y1=min(full_h,int(ys.max())+17)
        render_scale=args.size/max(x1-x0,y1-y0)
        render_w=max(16,round((x1-x0)*render_scale/4)*4)
        render_h=max(16,round((y1-y0)*render_scale/4)*4)
        native_flow=cv2.resize(field,(full_w,full_h))
        native_flow*=np.array([full_w/field.shape[1],full_h/field.shape[0]],np.float32)
        field=cv2.resize(native_flow[y0:y1,x0:x1],(render_w,render_h))
        field*=np.array([render_w/(x1-x0),render_h/(y1-y0)],np.float32)
        field*=cv2.resize(mask[y0:y1,x0:x1],(render_w,render_h),interpolation=cv2.INTER_NEAREST)[:,:,None]/255.
    else:
        x0,y0,x1,y1=0,0,full_w,full_h
        render_w=render_h=args.size
    spec.update(render_crop_xyxy=[x0,y0,x1,y1],render_dimensions=[render_w,render_h],
                detail_residual_strength=args.detail_strength,
                native_appearance_blend=args.native_appearance,
                native_detail_transport='backtrace bilinear sampling' if args.detail_backtrace else 'forward bilinear sum-splat',
                water_only_encoding=args.water_only,
                detail_method='native-reference high-pass carried by the same bidirectional flow; not sharpening static pixels' if args.detail_strength else 'none')
    encoder,decoder,opts,loaded=load_renderer(args.size)
    render_source=image[y0:y1,x0:x1]
    render_mask=mask[y0:y1,x0:x1]
    source_valid=cv2.erode(render_mask,np.ones((7,7),np.uint8))>0
    if args.water_only:
        # This gray-water reference has green vegetation leaking into the drawn
        # bank edge. Exclude chromatic source pixels without changing geometry.
        hsv=cv2.cvtColor(render_source,cv2.COLOR_BGR2HSV)
        if args.input_dir.resolve()==INPUT.resolve() or args.source_neutral:
            source_valid &= hsv[:,:,1] < 55
        if not source_valid.any():
            raise ValueError('Water region is too narrow after excluding its edge; enlarge the region.')
    encoder_source=render_source
    if args.water_only:
        # Extend *water* behind excluded areas for the convolutional receptive
        # field, so encoded banks/vegetation cannot be advected into the river.
        # This is an internal source layer; the visible background is unchanged.
        _,nearest=cv2.distanceTransformWithLabels((~source_valid).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
        y_valid,x_valid=np.where(source_valid)
        encoder_source=render_source[y_valid[nearest-1],x_valid[nearest-1]]
        cv2.imwrite(str(args.output/'internal-water-source.png'),encoder_source)
    image_square=cv2.resize(cv2.cvtColor(encoder_source,cv2.COLOR_BGR2RGB),(render_w,render_h))
    tensor=torch.from_numpy(image_square.copy()).permute(2,0,1)[None].float()/127.5-1
    flow=torch.from_numpy(field).permute(2,0,1)[None]
    count=round(args.duration*args.fps)
    frame_stride=getattr(args,'frame_stride',1)
    if frame_stride<1:raise ValueError('Frame stride must be positive')
    timeline=sorted(set([0,count//4,count//2,3*count//4,count-1])) if args.probe else sorted(set(range(0,count,frame_stride))|{count-1})
    spec.update(neural_frame_stride=frame_stride,
                intermediate_frames='bidirectional flow-compensated interpolation' if frame_stride>1 else 'each frame decoded')
    keep=set(timeline)|{count-t for t in timeline} if args.probe else None
    forward=displacements(flow,count,keep) if args.probe else DisplacementBlocks(flow,count)
    backward=displacements(-flow,count,keep) if args.probe else DisplacementBlocks(-flow,count)
    spec['trajectory_cache']='sparse probes' if args.probe else 'exact checkpointed 16-frame blocks'
    alpha=np.minimum(cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,3)/5,1)[:,:,None]
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    target=args.output/'candidate.mp4'
    if target.exists():
        raise ValueError('Existing output is not overwritten: '+str(target))
    h,w=image.shape[:2]
    writer=subprocess.Popen([ffmpeg,'-nostdin','-v','error','-f','rawvideo','-pix_fmt','bgr24',
        '-s',f'{w}x{h}','-r',str(args.fps),'-i','pipe:0','-an','-c:v','libx264',
        '-pix_fmt','yuv420p','-crf','18','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
    colors=[];samples=[];background_max=0
    native_source=encoder_source.astype(np.float32)
    native_residual=native_source-cv2.GaussianBlur(native_source,(0,0),1.5)
    if args.native_appearance:
        native_residual=native_source
    source_weight=np.ones(native_residual.shape[:2],np.float32)
    native_values=torch.from_numpy(np.concatenate([native_residual*source_weight[:,:,None],source_weight[:,:,None]],2)).permute(2,0,1)[None]
    native_yy,native_xx=np.mgrid[:render_source.shape[0],:render_source.shape[1]].astype(np.float32)
    def transport_details(displacement):
        native_displacement=torch.nn.functional.interpolate(displacement,size=render_source.shape[:2],mode='bilinear',align_corners=False)
        native_displacement*=torch.tensor([(x1-x0)/render_w,(y1-y0)/render_h]).view(1,2,1,1)
        if args.detail_backtrace:
            delta=native_displacement[0].numpy()
            map_x=native_xx+delta[0];map_y=native_yy+delta[1]
            invalid=(map_x<0)|(map_x>native_xx.shape[1]-1)|(map_y<0)|(map_y>native_xx.shape[0]-1)
            map_x[invalid]=native_xx[invalid];map_y[invalid]=native_yy[invalid]
            sampled=cv2.remap(native_residual,map_x,map_y,cv2.INTER_LINEAR)
            sampled=np.concatenate([sampled,np.ones((*sampled.shape[:2],1),np.float32)],2)
            return torch.from_numpy(sampled).permute(2,0,1)[None]
        return splat(native_values,native_displacement)
    sample_indices=set(np.linspace(0,count-1,12).round().astype(int))
    interp_flow=cv2.resize(field,(x1-x0,y1-y0))
    interp_flow*=np.array([(x1-x0)/render_w,(y1-y0)/render_h],np.float32)
    previous_crop=None;previous_t=None
    def emit_crop(crop,index):
        generated=image.copy();generated[y0:y1,x0:x1]=np.rint(crop).clip(0,255).astype(np.uint8)
        final=np.rint(generated*alpha+image*(1-alpha)).astype(np.uint8)
        delta=int(np.abs(final.astype(int)-image.astype(int))[mask==0].max())
        writer.stdin.write(final.tobytes())
        if args.probe:cv2.imwrite(str(args.output/f'probe-{index:04d}.png'),final)
        colors.append(final[mask>0].mean(axis=0).tolist())
        if index in sample_indices:
            tile=cv2.resize(final,(480,270));cv2.putText(tile,f'{index/args.fps:.2f}s',(8,22),0,.6,(255,255,255),2)
            samples.append(tile)
            if index==0:cv2.imwrite(str(args.output/'first-frame.png'),final)
        return delta
    try:
        with torch.inference_mode():
            features,z=encoder(tensor)
            if getattr(opts,'use_softmax_splatter_v2',False):
                raise ValueError('Per-target maximum splat normalization is not ported.')
            z=z if getattr(opts,'use_softmax_splatter_v1',False) else z-z.max()
            if not hasattr(opts,'no_clamp_Z'):z=z.clamp(-20,20)
            weighted=torch.cat([features*z.exp(),z.exp()],1)
            print('renderer_loaded '+json.dumps(loaded),flush=True)
            for t in timeline:
                weight=1-t/count
                combined=splat(weighted,forward[t])*weight+splat(weighted,backward[count-t])*(1-weight)
                fused=combined[:,:-1]/combined[:,-1:].clamp_min(1e-8)
                fused=torch.where(combined[:,-1:]>1e-8,fused,features)
                pred=decoder(fused.contiguous()).tanh()[0]
                pred=((pred.permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
                generated_crop=cv2.resize(cv2.cvtColor(pred,cv2.COLOR_RGB2BGR),(x1-x0,y1-y0)).astype(np.float32)
                if args.detail_strength or args.native_appearance:
                    transported=transport_details(backward[t] if args.detail_backtrace else forward[t])*weight+transport_details(forward[count-t] if args.detail_backtrace else backward[count-t])*(1-weight)
                    details=(transported[:,:3]/transported[:,3:].clamp_min(1e-8))[0].permute(1,2,0).numpy()
                    if args.native_appearance:
                        coverage=transported[0,3].numpy()
                        blend=args.native_appearance*np.minimum(coverage,1)[:,:,None]
                        generated_crop=np.clip((1-blend)*generated_crop+blend*details,0,255)
                    else:
                        generated_crop=np.clip(generated_crop+args.detail_strength*details,0,255)
                if args.probe or frame_stride==1:
                    background_max=max(background_max,emit_crop(generated_crop,t))
                elif previous_crop is not None:
                    span=t-previous_t
                    for index in range(previous_t,t):
                        elapsed=index-previous_t;fraction=elapsed/span
                        if not elapsed:intermediate=previous_crop
                        else:
                            left=cv2.remap(previous_crop,native_xx-interp_flow[:,:,0]*elapsed,native_yy-interp_flow[:,:,1]*elapsed,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT101)
                            right=cv2.remap(generated_crop,native_xx+interp_flow[:,:,0]*(span-elapsed),native_yy+interp_flow[:,:,1]*(span-elapsed),cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT101)
                            intermediate=(1-fraction)*left+fraction*right
                        background_max=max(background_max,emit_crop(intermediate,index))
                previous_crop=generated_crop;previous_t=t
                if t%12==0:print(f'frame {t+1}/{count} elapsed {time.time()-start:.1f}s',flush=True)
            if not args.probe and frame_stride>1:
                background_max=max(background_max,emit_crop(previous_crop,count-1))
    finally:
        writer.stdin.close()
        result=writer.wait()
    if result:raise RuntimeError('ffmpeg failed')
    if not args.probe:
        cv2.imwrite(str(args.output/'inspection12.jpg'),np.vstack([np.hstack(samples[i:i+3]) for i in range(0,12,3)]))
    spec.update(status='diagnostic_probe' if args.probe else 'rendered_pending_visual_review',frames=len(timeline) if args.probe else count,fps=args.fps,
                duration_seconds=(len(timeline) if args.probe else count)/args.fps,simulation_horizon_seconds=count/args.fps,
                neural_decoded_frames=len(timeline),
                rendered_frame_indices=list(timeline) if args.probe else None,
                water_source_saturation_limit=55 if args.water_only and (args.input_dir.resolve()==INPUT.resolve() or args.source_neutral) else None,
                elapsed_seconds=time.time()-start,
                weights_sha256=sha(WEIGHTS),strict_weight_keys=loaded,
                torch_version=torch.__version__,device='cpu',threads=args.threads,
                outside_mask_max_pixel_difference_before_encoding=background_max,
                water_mean_bgr_by_frame=colors,video_sha256=sha(target),
                numerical_backend='CPU torch scatter_add bilinear sum-splat; CUDA parity not measured',
                notes='Learned baseline rendering, imposed motion, original background compositing. No color correction. Not full SLR/SFS reproduction.')
    save_json(args.output/'result.json',spec)
    print(json.dumps({'video':str(target),'elapsed_seconds':spec['elapsed_seconds'],'training_ready':False}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--input-dir',type=Path,default=INPUT)
    p.add_argument('--boundary-tangent',action='store_true',help='Project near-boundary velocity along banks and protected obstacles; geometry constraint only.')
    p.add_argument('--flow-field',type=Path,help='External HxWx2 .npy flow, full-reference pixels per output frame; overrides imposed arrow speeds.')
    p.add_argument('--water-mask',type=Path,help='Explicit full-reference binary water mask; keeps supplied geometry separate from the coarse polygon.')
    p.add_argument('--source-neutral',action='store_true',help='For gray-water references only: exclude saturated bank pixels from the internal source.')
    p.add_argument('--size',type=int,default=256)
    p.add_argument('--duration',type=float,default=4)
    p.add_argument('--fps',type=int,default=24)
    p.add_argument('--frame-stride',type=int,default=1,help='Decode every Nth frame and interpolate intervening frames with the supplied flow; default decodes all frames.')
    p.add_argument('--speed',type=float,default=1)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--water-crop',action='store_true',help='Spend model pixels on the river instead of the fixed mountains.')
    p.add_argument('--detail-strength',type=float,default=0,help='Add advected native high-frequency detail; 0 disables.')
    p.add_argument('--water-only',action='store_true',help='Exclude static scenery from both learned features and advected detail.')
    p.add_argument('--native-appearance',type=float,default=0,help='0..1 blend of transported full native texture instead of high-pass residual.')
    p.add_argument('--highres-reference',action='store_true',help='Reuse existing 1672px generated water before its previous downsample; background is upscaled, not newly generated.')
    p.add_argument('--detail-backtrace',action='store_true',help='Advect native detail by backward characteristic sampling with OpenCV.')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--probe',action='store_true',help='Render five diagnostic frames across the requested horizon; MP4 is not a duration deliverable.')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:self_test()
    else:render(args)


if __name__=='__main__':main()
