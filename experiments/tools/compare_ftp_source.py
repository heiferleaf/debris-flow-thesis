# -*- coding: utf-8 -*-
"""Download exactly one known FTP source for byte comparison; never changes raw media.

Credentials are read from FTP_USER/FTP_PASS only. Server certificate must match
the already-trusted FileZilla certificate before credentials are transmitted.
"""
import argparse
import csv
import ftplib
import hashlib
import json
import os
from pathlib import Path
import ssl
import time
import xml.etree.ElementTree as ET


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    destination = (root / args.out).resolve()
    destination.relative_to((root / "data/process/decode-recheck").resolve())
    destination.mkdir(parents=True, exist_ok=False)
    with (root / "data/metadata/annotation-review/recheck-20261108-v2/all-dongchuan-scope.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(r for r in csv.DictReader(stream) if r["unit_id"] == args.unit)
    local_relative = Path(json.loads(row["raw_paths_json"])[0])
    assert local_relative.parts[:3] == ("data", "raw", "video_dataset_1")
    local_path = root / local_relative
    remote_path = "/" + "/".join(local_relative.parts[3:])
    host, port = "ftp.ncdc.ac.cn", 2121
    user, password = os.environ["FTP_USER"], os.environ["FTP_PASS"]
    trust_path = Path(os.environ["APPDATA"]) / "FileZilla/trustedcerts.xml"
    pins = [bytes.fromhex(n.findtext("Data")) for n in ET.parse(str(trust_path)).findall(".//Certificate")
            if n.findtext("Host") == host and n.findtext("Port") == str(port)]
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    ftp = ftplib.FTP_TLS(context=context)
    ftp.encoding = "utf-8"
    started = time.monotonic()
    try:
        ftp.connect(host, port, timeout=45)
        ftp.auth()
        peer = ftp.sock.getpeercert(binary_form=True)
        if peer not in pins:
            raise ValueError("Certificate differs from saved FileZilla trust; credential not sent")
        ftp.login(user, password)
        ftp.prot_p()
        ftp.set_pasv(True)
        ftp.voidcmd("TYPE I")
        size = ftp.size(remote_path)
        modification = ftp.sendcmd("MDTM " + remote_path)
        if size is None or size > 2 * 1024 ** 3:
            raise ValueError("Missing size or single-file limit exceeded")
        print(json.dumps(dict(stage="download_started", unit=args.unit, bytes=size,
                              remote_file=remote_path, certificate_pin_matches=True), ensure_ascii=False), flush=True)
        pending = destination / "source-redownload.mp4.part"
        result_path = destination / "source-redownload.mp4"
        downloaded, last_report = 0, time.monotonic()
        received_hash = hashlib.sha256()
        with pending.open("xb") as handle:
            def receive(block):
                nonlocal downloaded, last_report
                handle.write(block)
                received_hash.update(block)
                downloaded += len(block)
                if time.monotonic() - last_report >= 15:
                    print(json.dumps(dict(stage="downloading", bytes=downloaded, total=size,
                                          percent=round(downloaded * 100 / size, 1))), flush=True)
                    last_report = time.monotonic()
            transfer_reply = ftp.retrbinary("RETR " + remote_path, receive, blocksize=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        assert downloaded == size == pending.stat().st_size
        assert transfer_reply.startswith("226"), transfer_reply[:3]
        pending.rename(result_path)
        local_before = local_path.stat()
        local_hash = digest(local_path)
        local_after = local_path.stat()
        assert (local_before.st_size, local_before.st_mtime_ns) == (local_after.st_size, local_after.st_mtime_ns)
        report = dict(unit_id=args.unit, source_channel="NCDC_FTP", server=host, port=port,
                      remote_path=remote_path, remote_bytes=size, remote_MDTM=modification,
                      server_certificate_sha256=hashlib.sha256(peer).hexdigest(),
                      server_certificate_matches_preexisting_FileZilla_trust=True,
                      original_local_path=local_relative.as_posix(), original_local_bytes=local_before.st_size,
                      original_local_sha256=local_hash,
                      new_download_path=result_path.relative_to(root).as_posix(),
                      new_download_bytes=downloaded, new_download_sha256=received_hash.hexdigest(),
                      byte_identical=(local_before.st_size == size and local_hash == received_hash.hexdigest()),
                      elapsed_seconds=round(time.monotonic() - started, 3),
                      raw_replaced=False, training_inputs_modified=False,
                      credentials_saved_in_report=False)
        (destination / "download-comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(dict(stage="comparison_complete", unit=args.unit,
                              byte_identical=report["byte_identical"], bytes=downloaded,
                              elapsed_seconds=report["elapsed_seconds"])), flush=True)
        ftp.quit()
    except Exception as exc:
        message = str(exc)
        for secret in (password, user):
            if secret:
                message = message.replace(secret, "[redacted]")
        (destination / "failure.json").write_text(json.dumps(dict(error_type=type(exc).__name__, message=message,
                                                                  source_unchanged=True), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(dict(stage="failed", error_type=type(exc).__name__, message=message), ensure_ascii=False), flush=True)
        raise SystemExit(1)
    finally:
        ftp.close()


if __name__ == "__main__":
    main()
