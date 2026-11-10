"""Experimental photo rendering driven by the scene shallow-water solution.

Macroscopic advection uses the computed velocities. Fine gravity-wave detail is
an explicit appearance approximation; it is not resolved by the hydraulic grid.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import imageio_ffmpeg
import numpy as np


def transform(matrix, x, y):
    den = matrix[2,0]*x+matrix[2,1]*y+matrix[2,2]
    return ((matrix[0,0]*x+matrix[0,1]*y+matrix[0,2])/den,
            (matrix[1,0]*x+matrix[1,1]*y+matrix[1,2])/den)


def transport_detail(field, bx, by, fx, fy):
    """Limited forward/backward error correction, without coordinate resets."""
    def sample(a,x,y):
        return cv2.remap(a,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101)
    forward=sample(field,bx,by)
    reverse=sample(forward,fx,fy)
    corrected=forward+.5*(field-reverse)
    lower=sample(cv2.erode(field,np.ones((3,3),np.uint8)),bx,by)
    upper=sample(cv2.dilate(field,np.ones((3,3),np.uint8)),bx,by)
    return np.clip(corrected,lower,upper)


def run(simulation, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'candidate.mp4').exists():raise ValueError('Existing video is preserved')
    spec=json.loads((simulation/'simulation.json').read_text('utf8'))
    inp=Path(spec['source']);image=cv2.imread(str(inp/'wet-first-frame.png'))
    mask=cv2.imread(str(inp/'water-mask.png'),0);h,w=mask.shape
    data=np.load(simulation/'simulation.npz');states=data['states'];wet=data['mask']
    xx,yy=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
    # The ground plane has a projective horizon outside the river. Do not
    # evaluate its singular coordinates on mountains/sky that are never rendered.
    water_y,water_x=np.where(mask>0)
    sample_x=np.where(mask>0,xx,float(np.median(water_x)))
    sample_y=np.where(mask>0,yy,float(np.median(water_y)))
    gx,gy=transform(data['img_to_grid'],sample_x,sample_y)
    wx,wy=transform(data['img_to_world'],sample_x,sample_y)
    world_to_img=np.linalg.inv(data['img_to_world'])
    px,py=transform(world_to_img,wx+.01,wy)
    qx,qy=transform(world_to_img,wx,wy+.01)
    jxx,jyx=(px-sample_x)/.01,(py-sample_y)/.01
    jxy,jyy=(qx-sample_x)/.01,(qy-sample_y)/.01
    _,indices=cv2.distanceTransformWithLabels((~wet).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    sy,sx=np.where(wet)
    def project(values):
        fill=values[sy[indices-1],sx[indices-1]].astype(np.float32)
        return cv2.remap(fill,gx.astype(np.float32),gy.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    # Water-only texture support prevents an interpolated sample from bringing
    # bank/stone pixels into the moving water. Visible solids stay in place.
    source_valid=cv2.erode(mask,np.ones((3,3),np.uint8))>0
    _,near=cv2.distanceTransformWithLabels((~source_valid).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    yv,xv=np.where(source_valid);texture=image[yv[near-1],xv[near-1]]
    low=cv2.GaussianBlur(texture.astype(np.float32),(0,0),18)
    residual=cv2.cvtColor(texture,cv2.COLOR_BGR2GRAY).astype(np.float32)-cv2.cvtColor(low,cv2.COLOR_BGR2GRAY)
    detail= residual.copy()
    detail_rms=np.sqrt(cv2.GaussianBlur(residual**2,(0,0),14)).clip(5,28)
    rng=np.random.default_rng(20261110)
    distance=cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,5)
    outside=cv2.distanceTransform((mask==0).astype(np.uint8),cv2.DIST_L2,5)
    normal_y,normal_x=np.gradient(cv2.GaussianBlur(distance-outside,(0,0),1))
    length=np.maximum(np.hypot(normal_x,normal_y),1e-6);normal_x/=length;normal_y/=length
    edge_strength=np.clip(1-distance/3,0,1)
    alpha=np.clip(distance/1.5,0,1)[:,:,None]
    fps=24
    # Persist one evolving appearance field through the whole clip. There are
    # no phase resets, frame loops or crossfades back to the starting photograph.
    detail_decay=np.exp(-.15/fps)
    _,image_nearest=cv2.distanceTransformWithLabels((mask==0).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    first=states[0];depth=project(first[...,0]);dx=float(data['dx'])
    # Deterministic centimetre/sub-centimetre gravity-wave appearance. Their
    # amplitudes and lighting are assumptions, not measurements of this site.
    wave_parameters=[(.30,.002,1.15),(.47,.003,1.8),(.75,.004,1.45),(1.1,.005,2.1)]
    phases=[]
    for wavelength,amp,angle in wave_parameters:
        k=2*np.pi/wavelength;kx,ky=k*np.cos(angle),k*np.sin(angle)
        screen_wave_pixels=wavelength*np.sqrt((jxx*np.cos(angle)+jxy*np.sin(angle))**2+(jyx*np.cos(angle)+jyy*np.sin(angle))**2)
        visibility=np.clip((screen_wave_pixels-2)/3,0,1)
        phases.append(dict(k=k,kx=kx,ky=ky,amp=amp,phase=(kx*wx+ky*wy).astype(np.float32),visibility=visibility))
    writer=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r','24','-i','pipe:0','-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'candidate.mp4')],stdin=subprocess.PIPE)
    started=time.time();speeds=[];static_max=0
    near_bank=(distance>=3)&(distance<=20);middle=distance>40
    try:
        for t,state in enumerate(states):
            grid_h=np.maximum(state[...,0],1e-5)
            u=project(state[...,1]/grid_h);v=project(state[...,2]/grid_h)
            depth=project(grid_h)
            # 1.1 is a documented appearance approximation from depth-average
            # to surface transport, not a measured or universal conversion.
            vx=(jxx*u+jxy*v)*1.1;vy=(jyx*u+jyy*v)*1.1
            normal_speed=vx*normal_x+vy*normal_y
            vx-=edge_strength*normal_speed*normal_x;vy-=edge_strength*normal_speed*normal_y
            vx[mask==0]=0;vy[mask==0]=0
            # Extend transport through the image boundary, where water enters
            # the camera view. An identity map outside the photograph would
            # pin incoming coordinates and stretch them into long straight lines.
            ex=vx[water_y[image_nearest-1],water_x[image_nearest-1]]
            ey=vy[water_y[image_nearest-1],water_x[image_nearest-1]]
            if t:
                bx=xx-ex/fps;by=yy-ey/fps
                detail=transport_detail(detail,bx,by,xx+ex/fps,yy+ey/fps)
                # Small nonperiodic, correlated spatial innovations represent
                # unresolved surface appearance, not additional hydraulic force.
                noise=rng.standard_normal((h,w)).astype(np.float32)
                noise=cv2.GaussianBlur(noise,(0,0),.8)-cv2.GaussianBlur(noise,(0,0),3)
                noise/=max(float(noise[mask>0].std()),1e-5)
                detail=detail_decay*detail+np.sqrt(1-detail_decay**2)*detail_rms*.65*noise
                detail=np.clip(detail,-4*detail_rms,4*detail_rms)
            slope_x=np.zeros_like(xx);slope_y=np.zeros_like(xx)
            for wave in phases:
                omega=np.sqrt(9.81*wave['k']*np.tanh(wave['k']*depth))
                if t:wave['phase']-=(omega+wave['kx']*u+wave['ky']*v)/fps
                factor=wave['amp']*np.cos(wave['phase'])*wave['visibility']
                slope_x+=factor*wave['kx'];slope_y+=factor*wave['ky']
            # Small surface-normal distortions, with an explicit 1.5-pixel cap.
            ox=np.clip((jxx*slope_x+jxy*slope_y)*.12,-1.5,1.5)
            oy=np.clip((jyx*slope_x+jyy*slope_y)*.12,-1.5,1.5)
            details=cv2.remap(detail,xx+ox,yy+oy,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT101)[:,:,None]
            lighting=np.clip(30*(-.6*slope_x+.8*slope_y),-9,9)[:,:,None]
            water=low+details+lighting
            if not np.isfinite(water[mask>0]).all():raise ValueError('Invalid projected water appearance')
            final=np.rint(water*alpha+image*(1-alpha)).clip(0,255).astype(np.uint8)
            static_max=max(static_max,int(np.abs(final.astype(np.int16)-image.astype(np.int16))[mask==0].max()))
            if static_max:raise ValueError('Protected background changed')
            writer.stdin.write(final.tobytes())
            if t in [0,12,24,72,144,216,287]:cv2.imwrite(str(output/f'review-{t:04d}.png'),final)
            speed=np.hypot(vx,vy)
            speeds.append([float(np.median(speed[near_bank])),float(np.median(speed[middle]))])
            if t%72==0:print(f'frame {t+1}/{len(states)}',flush=True)
    finally:
        writer.stdin.close();code=writer.wait()
    if code:raise RuntimeError('Encoder failed')
    report=dict(simulation=str(simulation),frames=len(states),fps=24,duration_seconds=len(states)/24,
        surface_velocity_factor_assumption=1.1,wave_parameters=wave_parameters,
        texture_cycle_seconds=None,appearance_method='continuous limited scalar transport with stochastic detail replenishment',
        appearance_seed=20261110,detail_decay_rate_per_second=.15,
        near_bank_centre_projected_median_speed_pixels_per_second=speeds,
        outside_water_max_difference_before_encoding=static_max,
        limitations=['unmeasured hydraulic and camera geometry parameters','surface-wave and photometric appearance approximations',
            'not a site-validated free-surface optical simulation'],
        user_accepted=False,training_ready=False,elapsed_seconds=time.time()-started)
    (output/'render.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='near_bank_centre_projected_median_speed_pixels_per_second'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--simulation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.simulation,a.output)
