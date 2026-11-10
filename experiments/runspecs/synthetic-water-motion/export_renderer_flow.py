"""Project the temporal mean simulated velocity for the fixed-flow renderer."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from render_physical_scene import transform


def run(simulation, output):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'flow.npy').exists():raise ValueError('Preserve existing flow')
    spec=json.loads((simulation/'simulation.json').read_text('utf8'))
    mask=cv2.imread(str(Path(spec['source'])/'water-mask.png'),0)
    h,w=mask.shape;y,x=np.where(mask>0)
    xx,yy=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
    sx=np.where(mask>0,xx,float(np.median(x)));sy=np.where(mask>0,yy,float(np.median(y)))
    data=np.load(simulation/'simulation.npz');wet=data['mask'];s=data['states']
    velocity=(s[...,1:]/np.maximum(s[...,:1],1e-5)).mean(0)
    gx,gy=transform(data['img_to_grid'],sx,sy);wx,wy=transform(data['img_to_world'],sx,sy)
    inverse=np.linalg.inv(data['img_to_world'])
    px,py=transform(inverse,wx+.01,wy);qx,qy=transform(inverse,wx,wy+.01)
    _,labels=cv2.distanceTransformWithLabels((~wet).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    wy0,wx0=np.where(wet);fill=velocity[wy0[labels-1],wx0[labels-1]]
    v=cv2.remap(fill,gx.astype(np.float32),gy.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    flow=np.stack([((px-sx)*v[...,0]+(qx-sx)*v[...,1]),((py-sy)*v[...,0]+(qy-sy)*v[...,1])],-1)/.01*1.1/24
    distance=cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,5)
    outer=cv2.distanceTransform((mask==0).astype(np.uint8),cv2.DIST_L2,5)
    ny,nx=np.gradient(cv2.GaussianBlur(distance-outer,(0,0),1))
    length=np.maximum(np.hypot(nx,ny),1e-6);nx/=length;ny/=length
    dot=flow[...,0]*nx+flow[...,1]*ny;strength=np.clip(1-distance/3,0,1)
    flow[...,0]-=strength*dot*nx;flow[...,1]-=strength*dot*ny;flow[mask==0]=0
    np.save(output/'flow.npy',flow.astype(np.float32))
    record={'simulation':str(simulation),'method':'temporal mean of simulated depth-averaged velocity, projected to reference',
        'fps':24,'surface_factor_assumption':1.1,'units':'full-reference pixels per output frame',
        'time_variation_retained':False,'site_calibrated':False,'shape':list(flow.shape),
        'max_speed_pixels_per_frame':float(np.linalg.norm(flow,axis=2).max())}
    (output/'flow.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(json.dumps(record))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--simulation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.simulation,a.output)
