"""Prepare pinned upstream inference files. No API keys or paid services."""
import hashlib
import shutil
from pathlib import Path
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[2]
COMMIT='fa8d61046dfa19d80c5605be4a64e71f499b58f7'
BASE=ROOT/'data/results/synthetic-video/pretrained/slr-sfs-fa8d610'
WEIGHT='baseline2-motion2-fixedMotionFinetune.pth'
HASH='d95a5f59e65bb3ed808283dea73110949778ff36b62cf6165e8ff21a4b91c761'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    BASE.mkdir(parents=True,exist_ok=True)
    source=BASE/'source'/('SLR-SFS-'+COMMIT)
    if not (source/'models/networks/utilities.py').is_file():
        archive=BASE/'upstream.zip'
        urllib.request.urlretrieve('https://codeload.github.com/simon3dv/SLR-SFS/zip/'+COMMIT,archive)
        destination=(BASE/'source').resolve()
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                target=(destination/member.filename).resolve()
                if destination not in target.parents:raise RuntimeError('Invalid archive path')
            z.extractall(destination)
        archive.unlink()
    weights=BASE/WEIGHT
    if not weights.is_file():
        import gdown
        partial=weights.with_suffix('.download')
        print('Downloading the author-released 230 MB model. No account or API key is required.')
        result=gdown.download(id='1sNY4_58EMQKNiunBYsJiZ3jcPWAqGYBA',output=str(partial),quiet=False,resume=True)
        if not result:raise RuntimeError('Download unavailable. See README for the official manual download link.')
        if digest(partial)!=HASH:raise RuntimeError('Model checksum mismatch; the downloaded file was not installed.')
        partial.replace(weights)
    if digest(weights)!=HASH:raise RuntimeError('Existing model checksum mismatch. See README.')
    print('READY: source and model verified. Start the local workbench next.')

if __name__=='__main__':main()
