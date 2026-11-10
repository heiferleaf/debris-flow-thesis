"""Inspect one local generated clip; no generation, upload, or model training."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--ffmpeg',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--temp',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose a new inspection version')
    args.out.mkdir(parents=True)
    args.temp.mkdir(parents=True,exist_ok=True)
    capture=cv2.VideoCapture(str(args.video))
    fps=capture.get(cv2.CAP_PROP_FPS)
    gray=[]
    frame_hashes=set()
    dimensions=None
    while True:
        ok,frame=capture.read()
        if not ok:
            break
        dimensions=frame.shape[1],frame.shape[0]
        frame_hashes.add(hashlib.sha256(frame.tobytes()).hexdigest())
        gray.append(cv2.cvtColor(cv2.resize(frame,(320,180),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY))
    capture.release()
    if len(gray)<2 or not fps>0:
        raise ValueError('No complete local clip to inspect')
    wanted=sorted(set([0]+[min(round(t*fps),len(gray)-1) for t in (2,4,6,8)]+[len(gray)-1]))
    selected='+'.join('eq(n,%d)'%i for i in wanted)
    command=[str(args.ffmpeg),'-hide_banner','-i',str(args.video),'-vf',"select='%s',showinfo"%selected,
             '-vsync','0','-q:v','2',str(args.temp/'frame-%02d.jpg')]
    decoded=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (args.out/'sample-decode.log').write_bytes(decoded.stderr)
    if decoded.returncode:
        raise RuntimeError('FFmpeg sampling failed; see local decode log')
    timestamps=[float(x) for x in re.findall(r'pts_time:([0-9.e+\-]+)',decoded.stderr.decode('utf-8',errors='replace'))]
    images=sorted(args.temp.glob('frame-*.jpg'))
    if len(images)!=len(wanted) or len(timestamps)!=len(wanted):
        raise ValueError('Native sample time/image count mismatch')
    values=np.stack(gray).astype(np.float32)
    lag=max(1,round(.2*fps))
    changes=np.mean(abs(values[lag:]-values[:-lag]),axis=(1,2))
    to_first=np.mean(abs(values-values[0]),axis=(1,2))
    font_path='C:/Windows/Fonts/msyh.ttc'
    font=ImageFont.truetype(font_path,22)
    title_font=ImageFont.truetype(font_path,26)
    sheet=Image.new('RGB',(1920,850),'#f5f7fa')
    draw=ImageDraw.Draw(sheet)
    draw.text((18,14),'鸡冠山下游生成试片｜实际抽帧对照（未作训练验收）',font=title_font,fill='#172b43')
    for i,(path,t,index) in enumerate(zip(images,timestamps,wanted)):
        x=(i%3)*640;y=65+(i//3)*390
        im=Image.open(path).convert('RGB')
        im.thumbnail((640,360))
        sheet.paste(im,(x,y))
        draw.text((x+12,y+362),'t=%.3f秒 · 源帧%d'%(t,index),font=font,fill='#172b43')
    panel=args.temp/'contact-sheet.jpg'
    sheet.save(panel,quality=93)
    samples=[{'source_index':index,'actual_pts_seconds':t,'path':path.as_posix()}
             for index,t,path in zip(wanted,timestamps,images)]
    report={'video':args.video.as_posix(),'video_sha256':hashlib.sha256(args.video.read_bytes()).hexdigest(),
            'dimensions':dimensions,'decoded_frames':len(gray),'reported_fps':fps,
            'unique_decoded_frame_sha256':len(frame_hashes),'exact_all_frame_repetition':len(frame_hashes)==1,
            'sampled_frames':samples,'contact_sheet':panel.as_posix(),
            'gray_measurement':{'analysis_size':[320,180],'range':[0,255],
                                'lag_frames':lag,'lag_nominal_seconds':lag/fps,
                                'lag_MAD_median':float(np.median(changes)),
                                'lag_MAD_max':float(changes.max()),
                                'MAD_to_first_last':float(to_first[-1]),'MAD_to_first_max':float(to_first.max())},
            'motion_semantics':'Pixel variation is not proof of water flow, camera motion, physical realism or label validity.',
            'model_fits':0,'paid_requests':0,'image_uploads':0,'training_ready':False}
    (args.out/'inspection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['decoded_frames','reported_fps','dimensions','unique_decoded_frame_sha256','gray_measurement','contact_sheet']},ensure_ascii=False))


if __name__=='__main__':
    main()
