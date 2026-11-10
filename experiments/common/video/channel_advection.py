"""Generic image-space flow-map animation. This is not hydrodynamics."""
import cv2
import numpy as np


class ChannelAdvection:
    def __init__(self,image,polygon,flow,allowed_mask=None,period=1.6):
        self.base=image;self.h,self.w=image.shape[:2];self.period=period
        polygon_mask=np.zeros((self.h,self.w),np.uint8)
        cv2.fillPoly(polygon_mask,[np.rint(np.array(polygon)*[self.w,self.h]).astype(np.int32)],255)
        self.mask=polygon_mask if allowed_mask is None else cv2.bitwise_and(polygon_mask,allowed_mask)
        interior=cv2.erode(self.mask,np.ones((9,9),np.uint8))>0
        dark=((cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)<72)&interior).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(dark)
        fixed=np.zeros_like(self.mask)
        for i in range(1,count):
            if 6<=stats[i,cv2.CC_STAT_AREA]<=1800:fixed[labels==i]=255
        fixed=cv2.dilate(fixed,np.ones((3,3),np.uint8))
        holes=((polygon_mask>0)&(self.mask==0)).astype(np.uint8)*255
        self.texture=cv2.inpaint(image,cv2.bitwise_or(fixed,holes),5,cv2.INPAINT_TELEA)
        movable=(self.mask>0)&(fixed==0)
        self.alpha=np.minimum(cv2.distanceTransform(movable.astype(np.uint8),cv2.DIST_L2,3)/5,1)[:,:,None]
        self.color_region=cv2.erode(movable.astype(np.uint8),np.ones((9,9),np.uint8))>0
        self.source_median=np.median(image[self.color_region],axis=0)
        path=np.array(flow,np.float32)*[self.w,self.h]
        def exit_point(point,direction):
            direction=direction/max(np.linalg.norm(direction),1e-6);last=point
            for i in range(self.w+self.h):
                q=np.rint(point+direction*i).astype(int)
                if not (0<=q[0]<self.w and 0<=q[1]<self.h and self.mask[q[1],q[0]]):break
                last=q
            return np.array(last)
        inlet=exit_point(path[0],path[0]-path[1]);outlet=exit_point(path[-1],path[-1]-path[-2])
        size=(max(64,self.w//4),max(36,self.h//4))
        domain=cv2.resize(self.mask,size,interpolation=cv2.INTER_NEAREST)>0
        yy,xx=np.mgrid[:size[1],:size[0]].astype(np.float32)
        phi=xx/max(size[0]-1,1)
        start=np.zeros(size[::-1],np.uint8);end=start.copy()
        for buf,point in [(start,inlet),(end,outlet)]:
            point=np.rint(point*[size[0]/self.w,size[1]/self.h]).astype(int)
            cv2.circle(buf,tuple(point),3,1,-1)
        start=(start>0)&domain;end=(end>0)&domain
        if not start.any() or not end.any() or np.any(start&end):raise ValueError('Flow inlet/outlet invalid')
        kernel=np.array([[0,1,0],[1,0,1],[0,1,0]],np.float32)
        weight=cv2.filter2D(domain.astype(np.float32),-1,kernel,borderType=cv2.BORDER_CONSTANT)
        for _ in range(3600):
            value=cv2.filter2D(phi*domain,-1,kernel,borderType=cv2.BORDER_CONSTANT)/np.maximum(weight,1)
            phi=np.where(domain,value,0).astype(np.float32);phi[start]=0;phi[end]=1
        # Discrete no-flux banks; only neighbours in the water domain contribute.
        def neighbour(dx,dy):
            neighbour_phi=np.roll(phi,(dy,dx),(0,1));valid=np.roll(domain,(dy,dx),(0,1))
            return np.where(valid,neighbour_phi,phi)
        gx=(neighbour(-1,0)-neighbour(1,0))*.5
        gy=(neighbour(0,-1)-neighbour(0,1))*.5
        vx=cv2.resize(gx,(self.w,self.h));vy=cv2.resize(gy,(self.w,self.h))
        norm=np.maximum(np.sqrt(vx*vx+vy*vy),1e-6)
        speed=(self.w/960)*(4+20*np.clip((np.arange(self.h)[:,None]/self.h-.48)/.5,0,1))
        vx=(vx/norm*speed).astype(np.float32);vy=(vy/norm*speed).astype(np.float32)
        vx[self.mask==0]=0;vy[self.mask==0]=0
        self.velocity=np.dstack([vx,vy])
        grid_y,grid_x=np.mgrid[:self.h,:self.w].astype(np.float32)
        self.maps=[(grid_x,grid_y)];mx,my=grid_x.copy(),grid_y.copy();steps=48;dt=period/steps
        for _ in range(steps):
            ux=cv2.remap(vx,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            uy=cv2.remap(vy,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            nx=np.clip(mx-ux*dt,0,self.w-1).astype(np.float32);ny=np.clip(my-uy*dt,0,self.h-1).astype(np.float32)
            valid=cv2.remap(self.mask,nx,ny,cv2.INTER_NEAREST)>0
            mx=np.where(valid,nx,mx).astype(np.float32);my=np.where(valid,ny,my).astype(np.float32)
            self.maps.append((mx.copy(),my.copy()))
        self.layers=[cv2.remap(self.texture,a,b,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).astype(np.float32) for a,b in self.maps]

    def frame(self,seconds):
        phase=(seconds/self.period)%1;other=(phase+.5)%1
        a=self.layers[min(48,round(phase*48))];b=self.layers[min(48,round(other*48))]
        weight=1-abs(2*phase-1)
        water=a*weight+b*(1-weight)
        offset=np.clip(self.source_median-np.median(water[self.color_region],axis=0),-16,16)
        water=np.clip(water+offset,0,255)
        return np.rint(water*self.alpha+self.base*(1-self.alpha)).astype(np.uint8)
