# -*- coding: utf-8 -*-
"""Single-page local MiniMax workflow; opening it never calls the provider."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import sys
import threading
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video.local_water_workflow import LocalWater, read as read_json
ASSETS = Path(__file__).with_suffix('')


def handler(workflow, port, local_water=None):
    local_water=local_water or LocalWater(ROOT,cloud_enabled=workflow is not None)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def local(self):
            return self.client_address[0] == '127.0.0.1' and self.headers.get('Host') == '127.0.0.1:%d' % port

        def send(self, status, body, kind='application/json; charset=utf-8'):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode('utf-8')
            if isinstance(body, str):
                body = body.encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; media-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def file(self, path):
            if not path.is_file():
                return self.send(404, {'error': '素材不存在。'})
            kind = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
            if kind in ('text/html', 'text/css', 'text/javascript', 'application/javascript'):
                kind += '; charset=utf-8'
            return self.send(200, path.read_bytes(), kind)

        def do_GET(self):
            if not self.local():
                return self.send(403, {'error': '仅允许本机访问。'})
            route = urlsplit(self.path).path
            try:
                if route=='/local' or (route=='/' and workflow is None):
                    return self.file(ASSETS / 'local.html')
                if route in ('/', '/generate'):
                    return self.file(ASSETS / 'index.html')
                if route in ('/assets/app.js', '/assets/style.css','/assets/local.js','/assets/local.css'):
                    return self.file(ASSETS / route.rsplit('/', 1)[-1])
                if route=='/api/local/state':
                    return self.send(200,local_water.snapshot())
                match=re.fullmatch(r'/api/local/uploads/([a-f0-9]{24})/(image.png|draft)',route)
                if match:
                    if match[2]=='draft':return self.send(200,local_water.draft(match[1]))
                    return self.file(local_water.location('upload',match[1])/'wet-first-frame.png')
                match=re.fullmatch(r'/api/local/jobs/([a-f0-9]{24})/([A-Za-z0-9_.-]+)',route)
                if match:return self.file(local_water.asset(match[1],match[2]))
                if workflow is None:return self.send(404,{'error':'通用版没有配置场景或付费接口。'})
                if route in ('/api/workbench', '/state'):
                    return self.send(200, workflow.snapshot())
                match = re.fullmatch(r'/api/scenes/([A-Za-z0-9_-]+)/draft', route)
                if match:
                    return self.send(200, workflow.draft(match[1]))
                match = re.fullmatch(r'/api/scenes/([A-Za-z0-9_-]+)/(source|dry|wet|wet_crop|wet_user)', route)
                if match:
                    return self.file(workflow.asset(match[1], match[2]))
                match = re.fullmatch(r'/api/prepared/([a-f0-9]{64})/(locator.png|dry.png|reference.png)', route)
                if match:
                    prepared = workflow.metadata / 'prepared' / match[1]
                    if match[2] == 'locator.png':
                        return self.file(prepared / 'locator.png')
                    intent = read_json(prepared / 'intent.json')
                    return self.file(workflow.asset(intent.get('scene_id'), intent.get('reference_kind','dry')))
                match = re.fullmatch(r'/api/jobs/([A-Za-z0-9_-]+)/video', route)
                if match:
                    return self.file(workflow.video_path(match[1]))
                match = re.fullmatch(r'/api/jobs/([A-Za-z0-9_-]+)/raw-video', route)
                if match:
                    return self.file(workflow.run_path(match[1]).parent / 'candidate.mp4')
                match = re.fullmatch(r'/api/jobs/([A-Za-z0-9_-]+)/input.png', route)
                if match:
                    intent = read_json(workflow.run_path(match[1]))
                    return self.file(workflow.asset(intent['scene_id'], intent.get('reference_kind','dry')))
                match = re.fullmatch(r'/api/jobs/([A-Za-z0-9_-]+)/inspection16.jpg', route)
                if match:
                    return self.file(workflow.run_path(match[1]).parent / 'inspection16.jpg')
                return self.send(404, {'error': '页面不存在。'})
            except (ValueError, OSError, KeyError) as exc:
                return self.send(400, {'error': str(exc)[:180]})

        def do_POST(self):
            if (not self.local() or self.headers.get('Origin') != 'http://127.0.0.1:%d' % port or self.headers.get_content_type() != 'application/json'):
                return self.send(403, {'error': '仅允许本机工作台操作。'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                limit=29_000_000 if self.path=='/api/local/upload' else 65536
                if not 0 < length <= limit:
                    raise ValueError('请求大小无效。')
                data = json.loads(self.rfile.read(length).decode('utf-8'))
                if not isinstance(data, dict):
                    raise ValueError('请求格式无效。')
                if self.path=='/api/local/upload':return self.send(200,local_water.upload(data))
                if self.path=='/api/local/scene-base' and workflow is not None:return self.send(200,local_water.scene_base(workflow,data.get('scene_id')))
                if self.path=='/api/local/confirm-scene':return self.send(200,local_water.confirm_scene(data))
                if self.path=='/api/local/draft':return self.send(200,local_water.draft(data.get('upload_id'),data))
                if self.path=='/api/local/start':return self.send(200,local_water.start(data))
                if self.path=='/api/local/cancel':return self.send(200,local_water.cancel(data.get('id')))
                if workflow is None:return self.send(404,{'error':'通用版不调用付费接口。'})
                if self.path == '/api/draft':
                    return self.send(200, workflow.save_draft(data))
                if self.path == '/api/credentials':
                    return self.send(200, workflow.configure_key(data))
                if self.path == '/api/prepare':
                    return self.send(200, workflow.prepare(data))
                if self.path == '/api/submit':
                    return self.send(200, workflow.submit(data))
                if self.path == '/api/review':
                    return self.send(200, workflow.review(data))
                if self.path == '/api/poll':
                    identity = data.get('job_id')
                    value = read_json(workflow.run_path(identity))
                    if value.get('provider') != 'minimax_official':
                        raise ValueError('百炼历史作业仅用于查看，不通过官方密钥查询。')
                    if not value.get('task_id'):
                        raise ValueError('没有任务号，请先在百炼核对该次提交。')
                    if value.get('status') == 'succeeded':
                        return self.send(200, {'message': '视频已保存。'})
                    threading.Thread(target=workflow.poll, args=(identity,), daemon=True).start()
                    return self.send(200, {'message': '正在查询已有任务，没有新建生成。'})
                return self.send(404, {'error': '操作不存在。'})
            except (ValueError, OSError, KeyError, TypeError) as exc:
                return self.send(400, {'error': str(exc)[:180]})
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8774)
    parser.add_argument('--local-only',action='store_true',help='Portable local page without private scenes or cloud credentials.')
    parser.add_argument('--open-browser',action='store_true')
    args = parser.parse_args()
    if args.local_only:
        workflow=None
    else:
        from experiments.common.video.minimax_preview_workflow import MinimaxPreview
        workflow=MinimaxPreview(ROOT)
    local_water=LocalWater(ROOT,cloud_enabled=not args.local_only)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(workflow, args.port,local_water))
    print('Water video workbench: http://127.0.0.1:%d/' % args.port, flush=True)
    if args.open_browser:
        import webbrowser
        threading.Thread(target=webbrowser.open,args=('http://127.0.0.1:%d/local'%args.port,),daemon=True).start()
    try:server.serve_forever()
    finally:
        if local_water.active:local_water.cancel(local_water.active)
        server.server_close()


if __name__ == '__main__':
    main()
