"""Build a synchronized native-resolution river crop for an appearance trial."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def run(before, after, mask_path, output, before_label='修正前：近岸运动较弱', after_label='当前试片：浅水流场驱动，外观近似，待验收'):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'candidate.mp4').exists():
        raise ValueError('Preserve previous comparisons')
    caps=[cv2.VideoCapture(str(p)) for p in (before,after)]
    sizes=[(int(c.get(3)),int(c.get(4)),int(c.get(7)),c.get(5)) for c in caps]
    if sizes[0]!=sizes[1] or sizes[1]!=(1280,720,288,24.0):
        raise ValueError(f'Unexpected or unaligned inputs: {sizes}')
    mask=cv2.imread(str(mask_path),0)>0
    distance=cv2.distanceTransform(mask.astype('uint8'),cv2.DIST_L2,5)
    zones={'near_bank_3_to_20px':(distance>=3)&(distance<=20), 'interior_over_40px':distance>40}
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    bands=[]
    for label in (before_label,after_label):
        band=Image.new('RGB',(1280,40),(245,245,245));ImageDraw.Draw(band).text((18,5),label,font=font,fill=(15,23,42))
        bands.append(np.array(band)[:,:,::-1])
    writer=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','1280x680','-r','24','-i','pipe:0','-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'candidate.mp4')],stdin=subprocess.PIPE)
    samples={}; temporal=[[],[]]
    small_mask=cv2.resize(mask.astype(np.uint8),(320,180),interpolation=cv2.INTER_NEAREST)>0
    try:
        for i in range(288):
            frames=[]
            for c in caps:
                ok,f=c.read()
                if not ok:raise ValueError(f'Truncated input at frame {i}')
                frames.append(f)
            for variant,f in enumerate(frames):
                gray=cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(320,180),interpolation=cv2.INTER_AREA)
                temporal[variant].append(gray[small_mask].astype(np.float32))
            composite=np.concatenate((bands[0],frames[0][420:],bands[1],frames[1][420:]),axis=0)
            writer.stdin.write(composite.tobytes())
            if i in (0,6,12,18,24,30,36,42,144,287):samples[i]=frames
            if i==144:cv2.imwrite(str(output/'comparison-6s.png'),composite)
    finally:
        for c in caps:c.release()
        writer.stdin.close();code=writer.wait()
    if code:raise RuntimeError('Comparison encoder failed')
    observations=[]
    for variant in range(2):
        result={}
        for label,zone in zones.items():
            means=[];motions=[]
            for start in (0,6,12,18,24,30,36):
                a,b=[samples[k][variant] for k in (start,start+6)]
                means.append(float(np.mean(np.abs(a.astype(float)-b.astype(float))[zone])))
                flow=cv2.calcOpticalFlowFarneback(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),None,.5,3,15,3,5,1.2,0)
                motions.append(float(np.median(np.linalg.norm(flow,axis=2)[zone])))
            result[label]={'mean_abs_intensity_change_per_quarter_second':float(np.mean(means)), 'median_apparent_flow_pixels_per_quarter_second':float(np.mean(motions))}
        observations.append(result)
    repetition=[]
    for sequence in temporal:
        series=np.stack(sequence);centered=series-series.mean(axis=0,keepdims=True)
        stats={}
        for lag in (6,12,18,24,30,48):
            a,b=centered[:-lag],centered[lag:]
            corr=float(np.sum(a*b)/max(float(np.sqrt(np.sum(a*a)*np.sum(b*b))),1e-6))
            stats[str(lag/24)]={'temporal_mean_removed_correlation':corr,'mean_absolute_pixel_difference':float(np.abs(series[lag:]-series[:-lag]).mean())}
        repetition.append(stats)
    record={'before':str(before),'after':str(after),'before_label':before_label,'after_label':after_label,'sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in (before,after)],
        'frames':288,'fps':24,'duration':12,'crop':[0,420,1280,720],'rescaled':False,
        'diagnostics_before_after':observations,'lag_seconds_before_after':repetition,'interpretation':'Image motion diagnostics only; neither physical validation nor user acceptance.',
        'user_accepted':False,'training_ready':False}
    (output/'comparison.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('before','after','mask','output'):p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--before-label',default='修正前：近岸运动较弱');p.add_argument('--after-label',default='当前试片：浅水流场驱动，外观近似，待验收')
    a=p.parse_args();run(a.before,a.after,a.mask,a.output,a.before_label,a.after_label)
