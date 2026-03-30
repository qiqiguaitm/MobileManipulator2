#!/usr/bin/env python3
"""
网络传输基准服务端。接收二进制 blob，原样返回 depth。
通过 X-Server-Ms header 报告服务端处理耗时，客户端据此算纯传输时间。

用法:
  python3 _cc_bench_server.py --port 8087
"""

import argparse
import struct
import time

from flask import Flask, request, jsonify, make_response

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

FIELDS = ('rgb', 'dpt', 'ir_left', 'ir_right')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/predict_bin', methods=['POST'])
def predict_bin():
    data = request.data
    t0 = time.time()
    offset = 0
    dpt_bytes = None

    for name in FIELDS:
        if offset + 4 > len(data):
            return jsonify({'error': f'Truncated at {name}'}), 400
        length = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        if offset + length > len(data):
            return jsonify({'error': f'Truncated at {name}'}), 400
        if name == 'dpt':
            dpt_bytes = data[offset:offset + length]
        offset += length

    server_ms = (time.time() - t0) * 1000

    resp = make_response(dpt_bytes)
    resp.headers['Content-Type'] = 'image/png'
    resp.headers['X-Server-Ms'] = f'{server_ms:.2f}'
    return resp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8087)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)
