# -*- coding: utf-8 -*-
"""Bounded original-video probes for semantic review; no model or label changes."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'data/process/EXP-022/native-semantic-review-20261107-v1'
REQUESTS = [('v1_95',650),('v1_95',1230),('v1_95',1332),
            ('v1_63',340),('v1_63',500),('v1_63',900),('v1_63',1320),
            ('v1_82',248),('v1_82',600),('v1_82',1200),
            ('v1_122',968),('v1_122',1020),('v1_122',1450)]


def build():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    records=[]
    for uid,start in REQUESTS:
        text=(ROOT/'data/process/EXP-008-units-src256/log'/(uid+'.txt')).read_text(encoding='utf-8',errors='strict')
        path=Path(re.search(r"Input #0, .*?from '(.*?)':",text).group(1))
        if not path.is_file():
            raise FileNotFoundError(path)
        folder=OUT/('{}-{}'.format(uid,start));folder.mkdir()
        command=[ffmpeg,'-hide_banner','-loglevel','info','-ss',str(start),'-i',str(path),
                 '-t','4','-an','-vf','fps=4,scale=960:-2,showinfo','-frames:v','16',
                 '-q:v','2','-vsync','0',str(folder/'%03d.jpg')]
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=50)
        (folder/'ffmpeg.log').write_bytes(result.stderr)
        files=sorted(folder.glob('*.jpg'))
        item=dict(unit_id=uid,requested_seek_seconds=start,requested_duration=4,
                  current_source=path.relative_to(ROOT).as_posix(),source_size=path.stat().st_size,
                  source_mtime_ns=path.stat().st_mtime_ns,command=command,exit_code=result.returncode,
                  frames=len(files),alignment='Seek seconds are requested container-relative coordinates; validate burned clock/image against audit before attributing exact annotation time.',
                  historical_source_hash_claimed=False)
        if not files or result.returncode:
            item['status']='probe_failed_no_semantic_claim'
        else:
            sheet=Image.new('RGB',(1920,650),'white');draw=ImageDraw.Draw(sheet)
            draw.text((8,3),'{} original probe, requested{}s; sampled4fps; time alignment must be checked'.format(uid,start),font=font,fill='#173442')
            for j,k in enumerate(range(0,len(files),2)):
                with Image.open(files[k]) as im:
                    thumb=im.convert('RGB');thumb.thumbnail((475,275),Image.Resampling.LANCZOS)
                x,y=j%4*480+2,j//4*310+31;sheet.paste(thumb,(x,y))
                draw.text((x,y+278),'output+{:.2f}s'.format(k/4),font=font,fill='#173442')
            sheet.save(folder/'context.jpg',quality=94)
            item['status']='sampled_not_yet_visually_reviewed'
        records.append(item)
        print('{} {}: {} frames, exit{}'.format(uid,start,len(files),result.returncode),flush=True)
    record=dict(scope='13 bounded original-video probes,4seconds each; no full-video decode, model or labels.',
                generated_by_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),probes=records)
    (OUT/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    build()
