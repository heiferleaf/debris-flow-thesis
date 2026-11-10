"""Build all150 start/end evidence bundles from existing audit JPEGs only."""
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT=Path(__file__).resolve().parents[2]


def main():
    inv_path=ROOT/'data/metadata/synthetic-video/workflow-preparation-20261108-v1/scene-inventory.json'
    inventory=json.loads(inv_path.read_text(encoding='utf-8-sig'))
    cache={}
    for prefix,folder in [('v1','video1_segment'),('v4','video4_segment'),('v5','video5_segment')]:
        p=ROOT/'data/process/video_frames'/folder/'frames_index.csv'
        with p.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                key=r['video_key']
                path=p.parent/'frames'/r['frame_file']
                if path.is_file():cache.setdefault(key,[]).append((float(r['t_sec']),path))
    meta=ROOT/'data/metadata/annotation-review/boundaries-20261108-v1'
    temp=ROOT/'.codex/docs/.temp/boundaries-20261108-v1'
    meta.mkdir(parents=True,exist_ok=True);temp.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    bundles=[]
    for u in inventory['units']:
        key=u['unit_id'];available=sorted(cache[key]);first=available[0][0];last=available[-1][0]
        targets=[('start',first+x) for x in [0,2,4,10,20,30]]+[('end',last-x) for x in [30,20,10,4,2,0]]
        selected=[]
        sheet=Image.new('RGB',(384*6,256*2+40),'white')
        draw=ImageDraw.Draw(sheet)
        draw.text((8,6),key+' / '+u['source_camera']+' / 缓存首尾抽样（不是连续视频）',font=font,fill='black')
        for i,(side,wanted) in enumerate(targets):
            sec,path=min(available,key=lambda v:abs(v[0]-wanted))
            data=path.read_bytes()
            with Image.open(path) as im:frame=ImageOps.pad(im.convert('RGB'),(384,216),color='black')
            left=(i%6)*384;top=40+(i//6)*256
            draw.text((left+5,top+3),('开头' if side=='start' else '结尾')+' %.2fs'%sec,font=small,fill='black')
            sheet.paste(frame,(left,top+29))
            selected.append({'side':side,'requested_sec':wanted,'t_sec':sec,'path':path.relative_to(ROOT).as_posix(),
                             'sha256':hashlib.sha256(data).hexdigest()})
        output=temp/(key+'.jpg');sheet.save(output,'JPEG',quality=90)
        bundles.append({'unit_id':key,'source_camera':u['source_camera'],'group_id':u['group_id'],
            'contact_sheet':output.relative_to(ROOT).as_posix(),'frames':selected,
            'unique_observed_frames':len({x['path'] for x in selected}),'cached_time_range':[first,last],
            'labels_intentionally_not_displayed':True,'status':'prepared_not_yet_visually_reviewed'})
    document={'schema_version':1,'purpose':'All150 starts and ends, including pure-P starts and pure-N ends; no decoding or relabeling',
        'current_label_pointer':'data/metadata/annotation-revisions/current-review.json',
        'inventory_path':inv_path.relative_to(ROOT).as_posix(),'inventory_sha256':hashlib.sha256(inv_path.read_bytes()).hexdigest(),
        'units':bundles,'unit_count':len(bundles),'training_release':False}
    (meta/'manifest.json').write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for i,letter in enumerate('abc'):
        group={**document,'units':bundles[i::3],'reviewer_partition':letter,'unit_count':len(bundles[i::3])}
        (meta/('partition-'+letter+'.json')).write_text(json.dumps(group,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'units':len(bundles),'partitions':[len(bundles[i::3]) for i in range(3)],'video_decodes':0,'labels_changed':0}))


if __name__=='__main__':main()
