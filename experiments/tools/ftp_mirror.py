"""FTP 全量镜像下载:目录结构照搬、断点续传、跳过已完整的文件。

标准库实现,不装新依赖。凭据不写进本文件,用命令行参数或环境变量
FTP_USER / FTP_PASS 传入。中文文件名服务器编码不确定时,试 --encoding gbk。

用法(仓库根目录,先 dry-run 看清单和总量,再正式下载):
    .venv\\Scripts\\python.exe experiments\\tools\\ftp_mirror.py ^
        --user <u> --pass <p> --remote / --local data/raw/video_dataset_1 --dry-run
    .venv\\Scripts\\python.exe experiments\\tools\\ftp_mirror.py ^
        --user <u> --pass <p> --remote / --local data/raw/video_dataset_1
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import ssl
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ftplib import FTP, FTP_TLS, error_perm, error_temp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FTP 全量镜像下载(续传/跳过已有)")
    parser.add_argument("--host", default="ftp.ncdc.ac.cn")
    parser.add_argument("--port", type=int, default=2121)
    parser.add_argument("--user", default=os.environ.get("FTP_USER", ""), help="账号,缺省读环境变量 FTP_USER")
    parser.add_argument("--pass", dest="password", default=os.environ.get("FTP_PASS", ""), help="密码,缺省读 FTP_PASS")
    parser.add_argument("--remote", default="/", help="远端起始目录")
    parser.add_argument("--local", required=True, help="本地镜像根目录")
    parser.add_argument("--encoding", default="utf-8", help="服务器文件名编码(乱码时试 gbk)")
    parser.add_argument("--no-tls", action="store_true", help="不升级 FTPS(默认显式 TLS,本服务器要求 TLS)")
    parser.add_argument("--active", action="store_true", help="用主动 PORT 模式(默认被动 PASV)")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1,
                        help="并行下载线程数(默认 1;服务器限制单连接带宽时调大可提速)")
    parser.add_argument("--dry-run", action="store_true", help="只列清单和总量,不下载")
    return parser.parse_args()


class FtpMirror:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._tls = threading.local()  # 每个工作线程一条独立 FTP 连接
        self.downloaded = self.skipped = self.failed = 0
        self.bytes_total = 0
        self._lock = threading.Lock()

    # 连接 --------------------------------------------------------------
    def connect(self) -> FTP:
        """返回本线程的 FTP 连接,断了自动重建。"""
        ftp: FTP | None = getattr(self._tls, "ftp", None)
        if ftp is not None:
            try:
                ftp.voidcmd("NOOP")
                return ftp
            except Exception:  # noqa: BLE001 连接已断,重建
                try:
                    ftp.quit()
                except Exception:  # noqa: BLE001
                    pass
                self._tls.ftp = None
        if self.args.no_tls:
            ftp = FTP()
        else:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False  # 源站自签证书,不做校验
            ctx.verify_mode = ssl.CERT_NONE
            ftp = FTP_TLS(context=ctx)
        ftp.encoding = self.args.encoding
        ftp.connect(self.args.host, self.args.port, timeout=60)
        ftp.login(self.args.user, self.args.password)
        if isinstance(ftp, FTP_TLS):
            ftp.prot_p()  # 数据通道也走 TLS(服务器 550: SSL/TLS required)
        ftp.set_pasv(not self.args.active)
        self._tls.ftp = ftp
        print(f"已连接 {self.args.host}:{self.args.port} as {self.args.user}({self.args.encoding})", flush=True)
        return ftp

    # 目录遍历 ----------------------------------------------------------
    def list_dir(self, remote: str) -> list[tuple[str, bool, int]]:
        """返回 [(名字, 是目录, 字节大小)];大小未知为 -1。"""
        ftp = self.connect()
        try:  # 先试 MLSD(机器可读)
            entries = list(ftp.mlsd(remote))
            out = []
            for name, facts in entries:
                if name in (".", ".."):
                    continue
                is_dir = facts.get("type") == "dir"
                size = int(facts["size"]) if facts.get("size", "").isdigit() else -1
                out.append((name, is_dir, size))
            return out
        except error_perm:  # 服务器不支持 MLSD,退回 LIST 文本解析
            lines: list[str] = []
            ftp.retrlines(f"LIST {remote}", lines.append)
            return [parse_list_line(line) for line in lines if line.strip()]

    # 下载 --------------------------------------------------------------
    def remote_size(self, remote_path: str) -> int:
        try:
            return int(self.connect().size(remote_path))
        except Exception:  # noqa: BLE001 SIZE 不支持或路径变了
            return -1

    def fetch_one(self, remote_path: str, local_path: Path, remote_bytes: int) -> None:
        """下载单个文件:已完整则跳过;不完整则续传;续传失败重来。"""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        have = local_path.stat().st_size if local_path.exists() else 0
        if remote_bytes > 0 and have == remote_bytes:
            self.skipped += 1
            return
        last_error: Exception | None = None
        for attempt in range(1, self.args.retries + 1):
            try:
                ftp = self.connect()
                total = remote_bytes if remote_bytes > 0 else self.remote_size(remote_path)
                offset = local_path.stat().st_size if local_path.exists() else 0
                if offset and total > 0 and offset >= total:
                    self.skipped += 1  # 本地反而更大,视为已完整
                    return
                mode = "ab" if 0 < offset < total else "wb"
                got = 0
                with local_path.open(mode) as f:
                    def sink(chunk: bytes) -> None:
                        nonlocal got
                        f.write(chunk)
                        got += len(chunk)

                    if mode == "wb":
                        offset = 0
                    # 大块流式直写:网络字节流经最少次数拷贝直接落盘,不整文件缓存
                    ftp.retrbinary(f"RETR {remote_path}", sink, blocksize=1 << 20, rest=offset)
                actual = local_path.stat().st_size
                if total > 0 and actual != total:
                    raise error_temp(f"大小不符 本地 {actual} ≠ 远端 {total}")
                self.downloaded += 1
                self.bytes_total += actual
                print(f"  ✓ {remote_path}  {actual / 1e6:.1f} MB" + ("(续传)" if offset else ""), flush=True)
                return
            except (error_temp, error_perm, socket.error, OSError) as exc:
                last_error = exc
                print(f"  ✗ 第 {attempt} 次失败 {remote_path}: {exc}", flush=True)
                time.sleep(min(5 * attempt, 20))
        self.failed += 1
        print(f"  ✗✗ 放弃 {remote_path}: {last_error}", flush=True)

    def walk(self, remote: str, local: Path, jobs: list | None = None) -> None:
        """遍历目录树;dry_run 直接打印,否则把文件任务收进 jobs(并行下载用)。"""
        collect = jobs is not None
        try:
            entries = self.list_dir(remote)
        except Exception as exc:  # noqa: BLE001 单目录失败不中断整棵树
            print(f"  ✗ 列目录失败 {remote}: {exc}", flush=True)
            self.failed += 1
            return
        for name, is_dir, size in entries:
            safe = re.sub(r'[<>:"\\|?*]', "_", name)  # Windows 非法字符兜底
            r, l = f"{remote.rstrip('/')}/{name}", local / safe
            if is_dir:
                print(f"[目录] {r}", flush=True)
                self.walk(r, l, jobs)
            else:
                if self.args.dry_run:
                    print(f"  · {r}  {size / 1e6:.1f} MB" if size > 0 else f"  · {r}", flush=True)
                    self.bytes_total += max(size, 0)
                elif collect:
                    jobs.append((r, l, size))
                else:
                    self.fetch_one(r, l, size)

    def run(self) -> int:
        workers = max(1, self.args.workers)
        if self.args.dry_run or workers == 1:
            self.walk(self.args.remote, Path(self.args.local))
        else:
            jobs: list[tuple[str, Path, int]] = []
            self.walk(self.args.remote, Path(self.args.local), jobs)
            print(f"== 共 {len(jobs)} 个待传文件,{workers} 线程并行 ==", flush=True)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(self.fetch_one, r, l, s) for r, l, s in jobs]
                for _ in as_completed(futs):
                    pass  # 结果都打在日志里了
        print(
            f"\n== 完成:下载 {self.downloaded},跳过 {self.skipped},失败 {self.failed},"
            f"新落盘 {self.bytes_total / 1e9:.2f} GB ==",
            flush=True,
        )
        return 1 if self.failed else 0


def parse_list_line(line: str) -> tuple[str, bool, int]:
    """解析 LIST 行(unix 长格式为主),拿名字/是否目录/大小。"""
    parts = line.split(None, 8)
    if len(parts) >= 9 and parts[0].startswith(("d", "-")):  # unix: drwxr-xr-x ...
        is_dir = parts[0].startswith("d")
        try:
            size = int(parts[4])
        except ValueError:
            size = -1
        name = parts[8].split(" -> ")[0]
        return name, is_dir, size
    # windows: 09-10-25  10:23PM  <DIR>  name
    m = re.match(r"^\S+\s+\S+\s+(?:(<DIR>)|(\d+))\s+(.+)$", line)
    if m:
        return m.group(3).strip(), bool(m.group(1)), int(m.group(2) or -1)
    return line.rstrip(), False, -1


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # 后台/管道环境下 GBK 打不出 ✓ 等字符
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    args = parse_args()
    if not args.user:
        print("缺少账号(--user 或环境变量 FTP_USER/FTP_PASS)", file=sys.stderr)
        return 2
    return FtpMirror(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
