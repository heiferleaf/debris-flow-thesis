"""Inspect an observed public ZIP URL using one HEAD and one bounded Range GET.

This does not extract members or fall back to a full archive download.  The
remote ZIP directory must fit in the saved tail; every received byte is kept.
"""
import argparse
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


class SavedTail(io.RawIOBase):
    def __init__(self, length, tail):
        self.length, self.tail, self.position = length, tail, 0
        self.start = length - len(tail)

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        bases = {0: 0, 1: self.position, 2: self.length}
        if whence not in bases:
            raise ValueError('invalid seek origin')
        position = bases[whence] + offset
        if position < 0:
            raise ValueError('negative seek')
        self.position = position
        return position

    def read(self, size=-1):
        if self.position >= self.length or size == 0:
            return b''
        end = self.length if size < 0 else min(self.length, self.position + size)
        if self.position < self.start:
            raise ValueError('ZIP directory extends before saved tail; no automatic retry')
        data = self.tail[self.position-self.start:end-self.start]
        self.position = end
        return data


def write_json(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--url-key', default='object_url_from_visible_filePath_parameter')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tail-bytes', type=int, default=131072)
    args = parser.parse_args()
    if not 65536 <= args.tail_bytes <= 1048576:
        parser.error('tail-bytes must be between 64 KiB and 1 MiB')
    source_bytes = args.evidence.read_bytes()
    evidence = json.loads(source_bytes.decode('utf-8-sig'))
    url = evidence[args.url_key]
    if urllib.parse.urlparse(url).scheme != 'https':
        parser.error('only an observed HTTPS URL is allowed')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'generator-source.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'url-evidence.json').write_bytes(source_bytes)
    started = time.monotonic()
    result = dict(status='started', url=url,
                  source_evidence_sha256=hashlib.sha256(source_bytes).hexdigest(),
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  python=sys.version, downloaded_body_bytes=0, requests=[])
    try:
        request = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(request, timeout=20) as response:
            headers = dict(response.headers.items())
            result['requests'].append(dict(method='HEAD', status=response.status,
                                           resolved_url=response.url, headers=headers))
            length = int(response.headers['Content-Length'])
            etag = response.headers.get('ETag')
        if length <= 0:
            raise ValueError('missing or invalid archive size')
        count = min(length, args.tail_bytes)
        first = length-count
        headers = {'Range': 'bytes={}-{}'.format(first, length-1)}
        if etag:
            headers['If-Match'] = etag
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=20) as response:
            result['requests'].append(dict(method='GET', status=response.status,
                                           resolved_url=response.url,
                                           request_headers=headers,
                                           headers=dict(response.headers.items())))
            if response.status != 206:
                raise ValueError('server did not honor Range; full body was not read')
            expected_range = 'bytes {}-{}/{}'.format(first, length-1, length)
            if response.headers.get('Content-Range') != expected_range:
                raise ValueError('Content-Range mismatch')
            if int(response.headers['Content-Length']) != count:
                raise ValueError('Range Content-Length mismatch')
            if etag and response.headers.get('ETag') != etag:
                raise ValueError('archive identity changed between HEAD and GET')
            tail = response.read(count+1)
            result['downloaded_body_bytes'] = len(tail)
            (args.out/'zip-tail.bin').write_bytes(tail)
            if len(tail) != count:
                raise ValueError('truncated or oversized Range response')
        with zipfile.ZipFile(SavedTail(length, tail)) as archive:
            entries = [dict(name=item.filename, size=item.file_size,
                            compressed_size=item.compress_size,
                            compression_method=item.compress_type,
                            crc32='{:08x}'.format(item.CRC),
                            flags=item.flag_bits, local_header_offset=item.header_offset,
                            is_directory=item.is_dir()) for item in archive.infolist()]
        write_json(args.out/'members.json', entries)
        result.update(status='directory_read', archive_size=length, etag=etag,
                      tail_first_byte=first, tail_size=len(tail),
                      tail_sha256=hashlib.sha256(tail).hexdigest(),
                      member_count=len(entries), video_bytes_downloaded=0,
                      source_checksum_verified=False)
    except Exception as error:
        result.update(status='stopped', error_type=type(error).__name__, error=str(error))
    result['elapsed_seconds'] = time.monotonic()-started
    write_json(args.out/'acquisition.json', result)
    print(json.dumps({key:result[key] for key in
                      ('status','downloaded_body_bytes','elapsed_seconds')}))
    if result['status'] != 'directory_read':
        print(result.get('error', 'stopped'), file=sys.stderr)
        return 2
    print(json.dumps(entries, ensure_ascii=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
