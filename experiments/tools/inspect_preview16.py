"""Export 16 traceable inspection frames from a generated preview, not training."""
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.common.video.minimax_preview_workflow import MinimaxPreview,digest
from experiments.common.video.synthetic_wan import atomic_json,utc_now


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job-id',required=True)
    args=parser.parse_args()
    workflow=MinimaxPreview(ROOT)
    folder=workflow.run_path(args.job_id).parent
    source=workflow.video_path(args.job_id)
    capture=cv2.VideoCapture(str(source))
    fps=capture.get(cv2.CAP_PROP_FPS);count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps<=0 or count<16:raise ValueError('Not enough decoded video frames.')
    stop=min(count-1,round(3*fps))
    indices=np.rint(np.linspace(0,stop,16)).astype(int)
    output=folder/'inspection16';output.mkdir(exist_ok=True)
    frames=[];records=[]
    for i,index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,frame=capture.read()
        if not ok:raise ValueError('Frame could not be decoded.')
        path=output/f'frame-{i:02d}.jpg'
        if not cv2.imwrite(str(path),frame,[cv2.IMWRITE_JPEG_QUALITY,92]):raise ValueError('Frame not saved.')
        tile=cv2.resize(frame,(312,174));tile=cv2.copyMakeBorder(tile,22,0,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
        cv2.putText(tile,f'{i:02d} | {index/fps:.3f}s',(8,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
        frames.append(tile);records.append({'index':i,'video_frame_index':int(index),'time_seconds':float(index/fps),'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path)})
    capture.release()
    sheet=folder/'inspection16.jpg'
    cv2.imwrite(str(sheet),np.vstack([np.hstack(frames[k:k+4]) for k in range(0,16,4)]),[cv2.IMWRITE_JPEG_QUALITY,92])
    atomic_json(folder/'inspection16.json',{'created_at':utc_now(),'source_video':source.relative_to(ROOT).as_posix(),'source_video_sha256':digest(source),'fps':fps,'span_seconds':float(stop/fps),'frames':records,'sheet':sheet.relative_to(ROOT).as_posix(),'note':'Uniform time inspection only; no model crop/normalization, no N training release.','training_ready':False})
    print(sheet.resolve())


if __name__=='__main__':main()
