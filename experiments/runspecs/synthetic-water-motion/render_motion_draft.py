"""Render one free image-space motion draft from user-drawn water boundaries."""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.common.video.channel_advection import ChannelAdvection
from experiments.common.video.minimax_preview_workflow import read_json,digest
from experiments.common.video.synthetic_wan import atomic_json,utc_now


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='water-motion-local-20261109-v2');args=parser.parse_args()
    if not args.run_id.startswith('water-motion-local-') or not all(c.isalnum() or c in '-_' for c in args.run_id):raise ValueError('Invalid run ID')
    source_dir=ROOT/'data/process/synthetic-video/user-water-region-20261109-v2/camera-c9c07aee7f28'
    source=source_dir/'wet-first-frame.png'
    identity=args.run_id
    folder=ROOT/'data/process/synthetic-video/jobs'/identity
    if (folder/'state.json').exists():raise ValueError('Existing preview is not overwritten.')
    state={'id':identity,'scene_id':'camera-c9c07aee7f28','provider':'local_algorithm','model':'Local channel advection (not CFD)',
           'resolution':'960x540','duration':10,'estimated_cny':0,'input_mode':'local_motion','reference_kind':'wet_user',
           'created_at':utc_now(),'status':'running','message':'Rendering a free motion-control draft, not physically validated water.',
           'reference_sha256':digest(source),'batch_id':'local-water-motion-20261109-v2','variant_code':'L2',
           'variant_name':'Local motion draft, zero API fee','training_ready':False}
    atomic_json(folder/'state.json',state)
    try:
        region=read_json(source_dir/'user-region.json')['draft']
        image=cv2.imread(str(source))
        animation=ChannelAdvection(image,region['polygon'],region['flow'],allowed_mask=cv2.imread(str(source_dir/'edit-region.png'),0))
        ffmpeg=shutil.which('ffmpeg')
        if not ffmpeg:
            import imageio_ffmpeg
            ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        target=folder/'candidate.mp4'
        process=subprocess.Popen([ffmpeg,'-nostdin','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','960x540','-r','24','-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(target)],stdin=subprocess.PIPE)
        outside_max=0;colors=[]
        try:
            for i in range(240):
                frame=animation.frame(i/24)
                outside_max=max(outside_max,int(np.abs(frame.astype(np.int16)-image.astype(np.int16))[animation.mask==0].max()))
                if i%24==0:colors.append({'seconds':i/24,'median_bgr':np.median(frame[animation.color_region],axis=0).tolist()})
                process.stdin.write(frame.tobytes())
                if i%48==0:print('Rendered',i,'of 240 frames',flush=True)
        finally:process.stdin.close()
        if process.wait()!=0:raise ValueError('Encoding failed')
        assert outside_max==0
        atomic_json(folder/'animation-config.json',{'created_at':utc_now(),'source_image':source.relative_to(ROOT).as_posix(),
                    'source_sha256':digest(source),'user_region_sha256':digest(source_dir/'user-region.json'),
                    'code_sha256':digest(ROOT/'experiments/common/video/channel_advection.py'),'render_entry':Path(__file__).relative_to(ROOT).as_posix(),
                    'fps':24,'frames':240,'near_speed_px_s':28,'far_speed_px_s':4,
                    'outside_water_max_pixel_change_before_encoding':outside_max,'color_observations':colors,
                    'method':'Image-space harmonic flow direction with two short advected texture phases, fixed-object protection and median-color stabilization.',
                    'limitations':'Not a fluid solver. Two-phase blending can create repeated or ghost patterns; obstacles and apparent-flow naturalness require inspection. No physical realism or N label claim.',
                    'external_api_calls':0,'training_ready':False})
        state.update(status='succeeded',provider_status='local_complete',result_sha256=digest(target),
                     message='Free local motion draft rendered; not fluid simulation or approved training data.',updated_at=utc_now())
        atomic_json(folder/'state.json',state)
        print('Finished',identity,flush=True)
    except Exception as exc:
        state.update(status='failed',message='Local rendering failed: '+type(exc).__name__,updated_at=utc_now())
        atomic_json(folder/'state.json',state)
        raise


if __name__=='__main__':main()
