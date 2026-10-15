#!/usr/bin/env python3
"""Serve only exported MCAPs with Range/CORS support for Foxglove remote files."""
import argparse
import html
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    root = args.directory.resolve(strict=True)

    class Handler(BaseHTTPRequestHandler):
        def end_headers(self):
            origin = self.headers.get("Origin")
            if origin in ("https://app.foxglove.dev", "https://embed.foxglove.dev"):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Range, If-Range")
            self.send_header(
                "Access-Control-Expose-Headers",
                "Accept-Ranges, Content-Length, Content-Range, ETag",
            )
            self.send_header("Accept-Ranges", "bytes")
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_HEAD(self):
            self.serve(False)

        def do_GET(self):
            self.serve(True)

        def serve(self, body):
            name = unquote(urlsplit(self.path).path).lstrip("/")
            if not name:
                links = []
                for path in sorted(root.glob("*.mcap")):
                    if path.is_symlink():
                        continue
                    url = f"http://{args.bind}:{args.port}/{quote(path.name)}"
                    viewer = (
                        "https://app.foxglove.dev/~/view?ds=remote-file&ds.url="
                        + quote(url, safe="")
                    )
                    links.append(
                        f'<li><a href="{html.escape(viewer)}">{html.escape(path.name)}</a></li>'
                    )
                data = (
                    '<!doctype html><meta charset="utf-8"><title>仿真结果</title><h1>Foxglove 远程回放</h1><p>选择结果，在 Foxglove 中使用时间轴暂停和拖动。</p><ul>'
                    + "".join(links)
                    + "</ul>"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if body:
                    self.wfile.write(data)
                return
            path = root / name
            if (
                "/" in name
                or "\\" in name
                or path.suffix != ".mcap"
                or path.is_symlink()
                or not path.is_file()
            ):
                self.send_error(404)
                return
            size = path.stat().st_size
            etag = f'"{size:x}-{path.stat().st_mtime_ns:x}"'
            start, end, partial = 0, size - 1, False
            requested = self.headers.get("Range")
            if requested and self.headers.get("If-Range", etag) == etag:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                try:
                    if not match or not any(match.groups()):
                        raise ValueError()
                    first, last = match.groups()
                    if first:
                        start = int(first)
                        end = min(int(last), size - 1) if last else size - 1
                    else:
                        length = int(last)
                        if length <= 0:
                            raise ValueError()
                        start = max(0, size - length)
                    if start > end or start >= size:
                        raise ValueError()
                    partial = True
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("ETag", etag)
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if body:
                try:
                    with path.open("rb") as stream:
                        stream.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            chunk = stream.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

    ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
