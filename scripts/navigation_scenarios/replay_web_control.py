"""Small browser control page for an isolated rosbag2 playback domain."""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rosbag2_interfaces.srv import IsPaused, Pause, Resume


PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>仿真回放控制</title>
<style>body{font:18px system-ui;max-width:650px;margin:60px auto;padding:24px;background:#151a24;color:#eee}
button,a{font:inherit;padding:14px 28px;margin:8px 12px 8px 0;border-radius:8px}button{cursor:pointer}a{display:inline-block;color:#8bcaff;padding-left:0}#status{margin:24px 0}</style>
<h1>仿真回放控制</h1><p>接近垃圾桶 · bin_20261015_02</p>
<div id="status">正在读取状态…</div>
<button onclick="control('pause')">暂停</button><button onclick="control('resume')">继续播放</button>
<p><a id="viewer" target="_blank" rel="noopener">打开 Foxglove 画面 ↗</a></p>
<p>暂停会停止 bag 数据推进；继续播放从暂停位置接着走。播放结束后自动循环。</p>
<script>
document.querySelector('#viewer').href='https://app.foxglove.dev/~/view?ds=foxglove-websocket&ds.url='+encodeURIComponent('ws://'+location.hostname+':8768');
async function status(){try{let r=await fetch('/status');let d=await r.json();if(!r.ok)throw Error(d.error);document.querySelector('#status').textContent=d.paused?'已暂停':'正在播放';}catch(e){document.querySelector('#status').textContent='连接失败：'+e.message;}}
async function control(action){document.querySelectorAll('button').forEach(b=>b.disabled=true);try{let r=await fetch('/'+action,{method:'POST',headers:{'X-Replay-Control':'1'}});let d=await r.json();if(!r.ok)throw Error(d.error);await status();}catch(e){document.querySelector('#status').textContent='操作失败：'+e.message;}finally{document.querySelectorAll('button').forEach(b=>b.disabled=false);}}
status();setInterval(status,1500);
</script></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node("junbin_replay_web_control")
    if (
        node.get_namespace() != "/"
        or __import__("os").environ.get("ROS_DOMAIN_ID") != "151"
    ):
        raise RuntimeError("This controller requires isolated ROS_DOMAIN_ID=151")
    clients = {
        name: (node.create_client(kind, "/rosbag2_player/" + name), kind)
        for name, kind in [
            ("pause", Pause),
            ("resume", Resume),
            ("is_paused", IsPaused),
        ]
    }
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    def call(name):
        client, kind = clients[name]
        if not client.wait_for_service(timeout_sec=2):
            raise RuntimeError("回放服务不可用")
        done = threading.Event()
        future = client.call_async(kind.Request())
        future.add_done_callback(lambda _: done.set())
        if not done.wait(4):
            raise RuntimeError("回放服务响应超时")
        return future.result()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, data, content_type="application/json"):
            payload = (
                data.encode() if isinstance(data, str) else json.dumps(data).encode()
            )
            self.send_response(code)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/":
                return self.reply(200, PAGE, "text/html")
            if self.path != "/status":
                return self.reply(404, {"error": "Not found"})
            try:
                self.reply(200, {"paused": call("is_paused").paused})
            except Exception as error:
                self.reply(503, {"error": str(error)})

        def do_POST(self):
            # Custom header prevents cross-origin form submissions; no CORS enabled.
            if self.headers.get("X-Replay-Control") != "1":
                return self.reply(403, {"error": "Invalid control request"})
            if self.path not in ("/pause", "/resume"):
                return self.reply(404, {"error": "Not found"})
            try:
                call(self.path[1:])
                self.reply(200, {"ok": True})
            except Exception as error:
                self.reply(503, {"error": str(error)})

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
