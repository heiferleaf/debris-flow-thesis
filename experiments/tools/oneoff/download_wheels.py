"""临时脚本：用 Python 直接下载 wheel（绕开 pip 的大索引页）。用完即删。

用法::

    python download_wheels.py <保存目录> <包名==版本> [<包名==版本> ...]
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

WANT_SUFFIXES = ("cp37-abi3-win_amd64.whl", "cp38-cp38-win_amd64.whl", "cp38-none-win_amd64.whl", "py3-none-any.whl", "none-any.whl")
INDEXES = [
    "https://pypi.tuna.tsinghua.edu.cn/pypi/{name}/{version}/json",
    "https://pypi.org/pypi/{name}/{version}/json",
]


def pick_url(payload: dict) -> str:
    urls = [item["url"] for item in payload.get("urls", [])]
    for suffix in WANT_SUFFIXES:
        for url in urls:
            if url.endswith(suffix):
                return url
    raise SystemExit(f"no compatible wheel among: {[u.rsplit('/', 1)[-1] for u in urls]}")


def fetch_json(name: str, version: str) -> dict:
    last: Exception | None = None
    for template in INDEXES:
        url = template.format(name=name, version=version)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"  index failed: {url} -> {exc}")
    raise SystemExit(f"cannot fetch metadata for {name}=={version}: {last}")


def download(url: str, dest: Path) -> None:
    start = time.time()
    with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as fh:
        total = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
    secs = time.time() - start
    mb = total / 1024 / 1024
    print(f"  saved {dest.name}  {mb:.1f} MB in {secs:.1f}s ({mb / max(secs, 0.1):.2f} MB/s)")


def main() -> None:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    for spec in sys.argv[2:]:
        name, _, version = spec.partition("==")
        print(f"[{name}=={version}]")
        payload = fetch_json(name, version)
        url = pick_url(payload)
        print(f"  url: {url.rsplit('/', 1)[-1]}")
        download(url, outdir / url.rsplit("/", 1)[-1])


if __name__ == "__main__":
    main()
