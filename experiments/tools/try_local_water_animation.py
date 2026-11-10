"""One bounded local-water animation and traceable full-scene assembly."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.common.video.minimax_preview_workflow import ACTIVE,MinimaxPreview,read_json,digest
from experiments.common.video import minimax_official as official
from experiments.common.video.synthetic_wan import atomic_json,utc_now
from experiments.tools.explore_minimax_water_motion import execute

PROMPT=("延续首帧里已经存在的普通河水近景。水从画面左侧持续进入，沿中央水道明确地向右侧流出。"
        "清晰可见水面波纹与少量自然白色水花不断从左向右输送、绕过固定石块，水面局部形状随时间变化。"
        "维持首帧水量与已有水域，不出现新出水点。固定镜头，岸边砾石与石块保持原位，"
        "没有镜头移动、没有从无水到有水的变化。")


def prepare(workflow):
    folder=ROOT/'data/process/synthetic-video/wet-first-frame-20261109-v1/camera-c9c07aee7f28'
    source=folder/'wet-first-frame.png'
    if not source.is_file():raise ValueError('Wet first frame is missing.')
    if (workflow.metadata/'wet-local-animation.json').is_file():
        return folder,read_json(folder/'local-motion-plan.json')
    # Rectify the visible foreground stream, not the independent pond.
    quad=np.float32([[0,420],[300,325],[330,385],[0,500]])
    dest=np.float32([[0,0],[767,0],[767,431],[0,431]])
    transform=cv2.getPerspectiveTransform(quad,dest)
    frame=cv2.imread(str(source))
    crop=cv2.warpPerspective(frame,transform,(768,432),flags=cv2.INTER_CUBIC)
    crop_path=folder/'wet-motion-crop.png'
    cv2.imwrite(str(crop_path),crop)
    mask=np.zeros(frame.shape[:2],np.uint8)
    polygon=np.array([[0,435],[100,402],[200,372],[300,341],[390,320],[440,318],[490,317],
                      [530,304],[547,287],[562,289],[553,314],[535,329],[500,340],
                      [450,348],[400,350],[300,382],[200,413],[100,446],[0,487]],np.int32)
    cv2.fillPoly(mask,[polygon],255)
    crop_bounds=np.zeros_like(mask)
    cv2.fillConvexPoly(crop_bounds,quad.astype(np.int32),255)
    mask=cv2.bitwise_and(mask,cv2.erode(crop_bounds,np.ones((3,3),np.uint8)))
    cv2.imwrite(str(folder/'local-motion-water-mask.png'),mask)
    record={'created_at':utc_now(),'source':source.relative_to(ROOT).as_posix(),'source_sha256':digest(source),
            'crop':crop_path.relative_to(ROOT).as_posix(),'crop_sha256':digest(crop_path),
            'source_quad':quad.tolist(),'crop_dimensions':[768,432],'source_to_crop':transform.tolist(),
            'flow_in_source':'left-lower toward center/right-upper','flow_in_crop':'left to right',
            'water_composition_mask':(folder/'local-motion-water-mask.png').relative_to(ROOT).as_posix(),
            'purpose':'Geometric crop for model attention; not a new physical flow observation.', 'training_ready':False}
    atomic_json(folder/'local-motion-plan.json',record)
    return folder,record


def compose(workflow,identity,folder,plan):
    job_folder=workflow.run_path(identity).parent
    output=job_folder/'candidate-composited.mp4'
    if output.exists():raise ValueError('Existing composition is not overwritten.')
    base=cv2.imread(str(folder/'wet-first-frame.png'))
    mask=cv2.imread(str(folder/'local-motion-water-mask.png'),cv2.IMREAD_GRAYSCALE)
    alpha=np.minimum(cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,3)/6,1)[:,:,None]
    inverse=np.linalg.inv(np.array(plan['source_to_crop']))
    cap=cv2.VideoCapture(str(job_folder/'candidate.mp4'));fps=cap.get(cv2.CAP_PROP_FPS)
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    process=subprocess.Popen([ffmpeg,'-nostdin','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','960x540','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-pix_fmt','yuv420p','-crf','18',str(output)],stdin=subprocess.PIPE)
    count=0;outside_max=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            frame=cv2.resize(frame,tuple(plan['crop_dimensions']),interpolation=cv2.INTER_AREA)
            warped=cv2.warpPerspective(frame,inverse,(960,540),flags=cv2.INTER_CUBIC)
            final=np.rint(warped*alpha+base*(1-alpha)).astype(np.uint8)
            outside_max=max(outside_max,int(np.abs(final.astype(int)-base.astype(int))[mask==0].max()))
            process.stdin.write(final.tobytes());count+=1
    finally:
        cap.release();process.stdin.close()
    if process.wait()!=0 or count<16:raise ValueError('Composition encoding failed.')
    assert outside_max==0
    record={'created_at':utc_now(),'source_generated_video':(job_folder/'candidate.mp4').relative_to(ROOT).as_posix(),
            'source_generated_video_sha256':digest(job_folder/'candidate.mp4'),'fixed_background':plan['source'],
            'composition':plan,'frames':count,'fps':fps,'outside_water_changed_before_encoding':outside_max,
            'output':output.relative_to(ROOT).as_posix(),'output_sha256':digest(output),
            'limit':'Video compression can change pixels; motion realism and mask boundary still require review.', 'training_ready':False}
    atomic_json(job_folder/'composition.json',record)
    report=official.motion_check(output,workflow.draft('camera-c9c07aee7f28')['polygon'])
    workflow.update(identity,presentation_video=output.name,presentation_sha256=digest(output),motion_check=report,
                    message='局部水面动画已放回原场景；请检查流向、接缝和自然程度。')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    workflow=MinimaxPreview(ROOT);folder,plan=prepare(workflow)
    print('Prepared local river view:',plan['crop'],flush=True)
    if not args.execute:return
    record=workflow.metadata/'wet-local-animation.json'
    if record.exists():raise ValueError('This G preview was already allocated; no duplicate generation.')
    phase=[j for j in workflow.history() if j.get('batch_id')=='wet-first-frame-20261109-v1']
    if len(phase)!=1 or any(j['status'] in ACTIVE for j in workflow.history()):raise ValueError('Phase limit or unfinished job prevents submission.')
    value={'scene_id':'camera-c9c07aee7f28','provider':'minimax_official','model':official.MODEL,'resolution':'480P','duration':4,
           'estimated_cny':.64,'prompt':PROMPT,'input_mode':'first_frame','reference_kind':'wet_crop','reference_sha256':plan['crop_sha256'],
           'polygon':[[0,0],[1,0],[1,1],[0,1]],'batch_id':'wet-first-frame-20261109-v1','variant_code':'G',
           'variant_name':'局部河道首帧动画后回贴','source_first_frame_sha256':plan['source_sha256'],'training_ready':False}
    body=official.request_body(value,folder/'wet-motion-crop.png')
    identity=execute(workflow,record,value,body)
    if read_json(workflow.run_path(identity)).get('status')=='succeeded':compose(workflow,identity,folder,plan)
    print('Finished G',identity,flush=True)


if __name__=='__main__':main()
