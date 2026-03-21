#!/usr/bin/env python3
"""
combined_server.py — 合并 SAM3 + FoundationStereo 推理服务

单进程加载两个模型到同一 GPU, 通过 ThreadPoolExecutor 并行推理。
每个 GPU 运行一个实例, 由 start_combined.sh 启动。

API:
  POST /api/perceive   → binary framing [4B json_len LE][json_bytes][depth_zlib_bytes]
  GET  /api/health      → {"status":"ok", "gpu":0, "models":["sam3","fs"]}

Usage:
  python combined_server.py --port 8090 --gpu 0
"""

import argparse
import asyncio
import hashlib
import json
import os
import struct
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import Response, JSONResponse
import uvicorn

# TurboJPEG for faster color decode (~1.5x vs cv2)
_TURBOJPEG_LIB = '/data1/miniconda3/pkgs/libjpeg-turbo-3.0.0-hd590300_1/lib/libturbojpeg.so'
_tj = None
try:
    from turbojpeg import TurboJPEG, TJPF_BGR
    _tj = TurboJPEG(_TURBOJPEG_LIB)
    print(f'[INIT] TurboJPEG loaded from {_TURBOJPEG_LIB}')
except Exception as e:
    print(f'[INIT] TurboJPEG not available ({e}), using cv2 fallback')

# ─── Globals (filled at startup) ───
_sam3_tiled = None        # SAM3TiledEngine instance
_fs_processor = None      # FoundationStereoDepthProcessor instance
_gpu_id = 0

_pool = ThreadPoolExecutor(max_workers=2)

# Intrinsics cache: md5_bytes → calib_dict  (same scheme as stereo_server.py)
_intrinsics_cache: dict = {}

app = FastAPI(title='Combined Perception Server')


# ─────────────────── helpers ───────────────────

def decode_jpeg_color(data: bytes):
    """Decode color JPEG — turbojpeg if available, else cv2."""
    if _tj is not None:
        return _tj.decode(data, pixel_format=TJPF_BGR)
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def decode_jpeg_gray(data: bytes):
    """Decode grayscale JPEG — always cv2 (turbojpeg has no gray advantage)."""
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def encode_depth_zlib(depth_uint16: np.ndarray) -> bytes:
    """Encode depth as zlib-compressed half-res uint16 raw bytes.

    Format: [2B H_half LE][2B W_half LE][zlib(raw_uint16_bytes)]
    ~2x faster encode than PNG, similar compressed size.
    """
    H, W = depth_uint16.shape[:2]
    hh, hw = H // 2, W // 2
    half = cv2.resize(depth_uint16, (hw, hh), interpolation=cv2.INTER_NEAREST)
    header = struct.pack('<HH', hh, hw)
    raw = np.ascontiguousarray(half).tobytes()
    return header + zlib.compress(raw, level=1)


def _parse_intrinsics_dict(data: dict) -> dict:
    """Parse intrinsics YAML dict → calib dict (same as stereo_server.py)."""
    prof = data['profiles'][0]
    rgb_intr = {
        'fx': prof['fx'], 'fy': prof['fy'],
        'cx': prof['cx'], 'cy': prof['cy'],
        'width': prof['image_width'], 'height': prof['image_height'],
    }
    ir = data['ir_profiles'][0]
    ir_intr = {
        'fx': ir['fx'], 'fy': ir['fy'],
        'cx': ir['cx'], 'cy': ir['cy'],
        'width': ir['image_width'], 'height': ir['image_height'],
    }
    baseline = data.get('baseline', 0.0499)
    ext = data.get('ir_to_color_extrinsics', {})
    R = np.array(ext.get('rotation', np.eye(3).flatten())).reshape(3, 3)
    t_vec = np.array(ext.get('translation', np.zeros(3)))
    return {
        'rgb_intr': rgb_intr, 'ir_intr': ir_intr,
        'baseline': baseline,
        'ir_to_color_R': R, 'ir_to_color_t': t_vec,
    }


def parse_intrinsics_cached(raw_bytes: bytes) -> tuple:
    """Parse intrinsics YAML bytes with MD5 cache. Returns (calib_dict, hex_hash)."""
    key = hashlib.md5(raw_bytes).digest()
    hex_hash = key.hex()
    cached = _intrinsics_cache.get(key)
    if cached is not None:
        return cached, hex_hash
    data = yaml.safe_load(raw_bytes)
    calib = _parse_intrinsics_dict(data)
    _intrinsics_cache[key] = calib
    return calib, hex_hash


def resolve_intrinsics(file_bytes, hash_header):
    """Resolve intrinsics from upload or cached hash."""
    if file_bytes and len(file_bytes) > 0:
        return parse_intrinsics_cached(file_bytes)
    if hash_header:
        key = bytes.fromhex(hash_header)
        calib = _intrinsics_cache.get(key)
        if calib:
            return calib, hash_header
    raise ValueError('No intrinsics provided and no cached hash match')


# ─────────────────── SAM3 wrapper ───────────────────

def _detection_to_dict(det, return_mask, img_h, img_w):
    """Convert trtsam3.DetectionBox → JSON-serializable dict."""
    box = det.box
    result = {
        'class_name': det.class_name,
        'score': round(float(det.score), 4),
        'bbox': [round(float(box.left), 1), round(float(box.top), 1),
                 round(float(box.right), 1), round(float(box.bottom), 1)],
    }
    if return_mask and det.segmentation is not None:
        try:
            from pycocotools import mask as mask_util
            seg_mask = np.array(det.segmentation.mask)
            if seg_mask.size > 0:
                full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                x1, y1 = max(0, int(box.left)), max(0, int(box.top))
                x2, y2 = min(img_w, int(box.right)), min(img_h, int(box.bottom))
                crop_h, crop_w = y2 - y1, x2 - x1
                if seg_mask.shape[0] == crop_h and seg_mask.shape[1] == crop_w:
                    full_mask[y1:y2, x1:x2] = seg_mask
                rle = mask_util.encode(np.asfortranarray(full_mask))
                rle['counts'] = rle['counts'].decode('ascii')
                result['mask'] = {'size': list(rle['size']), 'counts': rle['counts']}
        except Exception:
            pass
    return result


def run_sam3(rgb_bgr, text_prompt, confidence, return_mask):
    """Run SAM3 tiled detection. Returns dict with 'objects' list."""
    t0 = time.time()
    h, w = rgb_bgr.shape[:2]

    # Parse categories for tiled engine
    categories = [c.strip() for c in text_prompt.replace('.', ',').split(',') if c.strip()]
    te = _get_tiled_engine(categories, confidence)
    dets = te.detect(rgb_bgr, return_mask=return_mask)

    objects = [_detection_to_dict(d, return_mask, h, w) for d in dets]
    elapsed = (time.time() - t0) * 1000
    return {'objects': objects, 'sam3_ms': round(elapsed, 1)}


def run_fs(ir_left, ir_right, calib_dict):
    """Run FoundationStereo → uint16 depth (mm)."""
    t0 = time.time()
    result = _fs_processor.process(
        left_ir=ir_left,
        right_ir=ir_right,
        calib_dict=calib_dict,
        fill_holes=True,
    )
    depth_m = result['depth_rgb_aligned']
    depth_mm = (depth_m * 1000).clip(0, 65535).astype(np.uint16)
    elapsed = (time.time() - t0) * 1000
    return depth_mm, round(elapsed, 1)


# ─────────────────── endpoints ───────────────────

@app.get('/api/health')
def health():
    models = []
    if _sam3_tiled is not None:
        models.append('sam3')
    if _fs_processor is not None:
        models.append('fs')
    return {'status': 'ok', 'gpu': _gpu_id, 'models': models}


@app.post('/api/perceive')
async def perceive(
    request: Request,
    rgb: UploadFile = File(...),
    ir_left: UploadFile = File(...),
    ir_right: UploadFile = File(...),
    intrinsics: UploadFile = File(None),
    text_prompt: str = Form('object'),
    confidence: str = Form('0.30'),
    return_mask: str = Form('false'),
):
    t_start = time.time()

    # Opt C: parallel upload reads via asyncio.gather
    if intrinsics:
        rgb_bytes, ir_l_bytes, ir_r_bytes, intr_bytes = await asyncio.gather(
            rgb.read(), ir_left.read(), ir_right.read(), intrinsics.read())
    else:
        rgb_bytes, ir_l_bytes, ir_r_bytes = await asyncio.gather(
            rgb.read(), ir_left.read(), ir_right.read())
        intr_bytes = None
    t_read = time.time()

    # Resolve intrinsics
    hash_header = request.headers.get('X-Intrinsics-Hash')
    try:
        calib_dict, intr_hash = resolve_intrinsics(intr_bytes, hash_header)
    except ValueError as e:
        return JSONResponse({'error': str(e)}, status_code=400)

    conf = float(confidence)
    mask = return_mask.lower() == 'true'

    # ─── Parallel decode + inference ───
    # Decode RGB on main thread, IR pair on pool (overlap)
    fut_ir_l = _pool.submit(decode_jpeg_gray, ir_l_bytes)
    fut_ir_r = _pool.submit(decode_jpeg_gray, ir_r_bytes)
    rgb_bgr = decode_jpeg_color(rgb_bytes)  # ~5ms, overlaps with IR decode
    if rgb_bgr is None:
        return JSONResponse({'error': 'Failed to decode RGB'}, status_code=400)
    t_rgb_dec = time.time()

    ir_l = fut_ir_l.result(timeout=5)
    ir_r = fut_ir_r.result(timeout=5)
    if ir_l is None or ir_r is None:
        return JSONResponse({'error': 'Failed to decode IR'}, status_code=400)

    t_decode = time.time()

    # Launch both simultaneously — SAM3 is lightweight (~31ms),
    # finishes before FS TRT peak, minimal contention
    fut_fs = _pool.submit(run_fs, ir_l, ir_r, calib_dict)
    fut_sam3 = _pool.submit(run_sam3, rgb_bgr, text_prompt, conf, mask)

    det_result = fut_sam3.result(timeout=30)
    t_sam3_done = time.time()
    depth_mm, fs_ms = fut_fs.result(timeout=60)

    t_infer = time.time()

    # ─── Binary framing response ───
    det_json_bytes = json.dumps(det_result, ensure_ascii=False).encode('utf-8')
    t_json = time.time()

    depth_bytes = encode_depth_zlib(depth_mm) if depth_mm is not None else b''
    t_png = time.time()

    json_len = len(det_json_bytes)
    body = struct.pack('<I', json_len) + det_json_bytes + depth_bytes

    t_encode = time.time()

    timing = {
        'read_ms': round((t_read - t_start) * 1000, 1),
        'rgb_dec_ms': round((t_rgb_dec - t_read) * 1000, 1),
        'ir_dec_ms': round((t_decode - t_rgb_dec) * 1000, 1),
        'sam3_ms': det_result.get('sam3_ms', 0),
        'sam3_wait_ms': round((t_sam3_done - t_decode) * 1000, 1),
        'fs_ms': fs_ms,
        'fs_wait_ms': round((t_infer - t_decode) * 1000, 1),
        'json_ms': round((t_json - t_infer) * 1000, 1),
        'png_ms': round((t_png - t_json) * 1000, 1),
        'frame_ms': round((t_encode - t_png) * 1000, 1),
        'total_ms': round((t_encode - t_start) * 1000, 1),
        'rgb_kb': round(len(rgb_bytes) / 1024, 1),
        'ir_l_kb': round(len(ir_l_bytes) / 1024, 1),
        'ir_r_kb': round(len(ir_r_bytes) / 1024, 1),
        'depth_kb': round(len(depth_bytes) / 1024, 1),
    }
    print(f'[PERF] {timing}')

    # Client-compatible timing (backward compat keys)
    client_timing = {
        'decode_ms': timing['read_ms'] + timing['rgb_dec_ms'] + timing['ir_dec_ms'],
        'sam3_ms': timing['sam3_ms'],
        'fs_ms': timing['fs_ms'],
        'infer_wall_ms': timing['fs_wait_ms'],
        'encode_ms': timing['json_ms'] + timing['png_ms'] + timing['frame_ms'],
        'total_ms': timing['total_ms'],
    }

    return Response(
        content=body,
        media_type='application/octet-stream',
        headers={
            'X-Intrinsics-Hash': intr_hash,
            'X-Timing': json.dumps(client_timing),
        },
    )


# ─────────────────── model loading ───────────────────

_tiled_engine_cache = {}  # frozenset(prompts) → SAM3TiledEngine


def _get_tiled_engine(categories, confidence):
    """Get or build SAM3TiledEngine for given categories (cached)."""
    from sam3_tiled import SAM3TiledEngine

    key = frozenset(categories)
    if key in _tiled_engine_cache:
        return _tiled_engine_cache[key]

    cats = {c: {'prompt': c, 'threshold': confidence} for c in categories}
    te = SAM3TiledEngine(
        _sam3_engine, _tokenizer, cats,
        tile_size=2160, tile_overlap=0.05, nms_iou=0.5,
        infer_lock=_infer_lock,
    )
    _tiled_engine_cache[key] = te
    print(f'[CACHE] Built tiled engine for: {categories}')
    return te


# Raw engine + tokenizer (needed to build tiled engines)
_sam3_engine = None
_tokenizer = None
_infer_lock = threading.Lock()


def load_models(args):
    """Load FoundationStereo + SAM3 into current GPU.

    IMPORTANT: FS must load FIRST to initialize the Python/PyTorch CUDA context.
    SAM3's C++ TRT engine adapts to an existing context, but not vice-versa.
    Reversing this order causes CUDA error 400 on multi-GPU systems.
    """
    global _sam3_engine, _tokenizer, _sam3_tiled, _fs_processor, _gpu_id
    _gpu_id = args.gpu

    # ── FoundationStereo (load FIRST — sets up CUDA context) ──
    fs_dir = os.path.join(os.path.dirname(__file__), '..', 'foundation_stereo')
    sys.path.insert(0, fs_dir)

    from core.stereo_depth_processor import FoundationStereoDepthProcessor

    engine_path = args.fs_engine_path
    print(f'[INIT] Loading FoundationStereo from {engine_path} (GPU {args.gpu}) ...')
    _fs_processor = FoundationStereoDepthProcessor(
        engine_path=engine_path,
        backend='trt',
        gpu_id=args.gpu,
    )
    print('[INIT] FoundationStereo ready.')

    # ── SAM3 (load SECOND — adapts to existing CUDA context) ──
    sam3_dir = os.path.join(os.path.dirname(__file__), '..', 'sam3_trt')
    sys.path.insert(0, sam3_dir)
    sys.path.insert(0, os.path.join(sam3_dir, 'script'))

    import trtsam3
    from tokenizers import Tokenizer

    engine_dir = args.sam3_engine_dir
    print(f'[INIT] Loading SAM3 engines from {engine_dir} (GPU {args.gpu}) ...')
    _sam3_engine = trtsam3.Sam3Infer.create_instance(
        vision_path=os.path.join(engine_dir, 'vision-encoder.engine'),
        text_path=os.path.join(engine_dir, 'text-encoder.engine'),
        geometry_path=os.path.join(engine_dir, 'geometry-encoder.engine'),
        decoder_path=os.path.join(engine_dir, 'decoder.engine'),
        gpu_id=args.gpu,
    )
    if _sam3_engine is None:
        raise RuntimeError('Failed to create Sam3Infer instance')

    _tokenizer = Tokenizer.from_file(os.path.join(engine_dir, 'tokenizer.json'))
    _tokenizer.enable_padding(length=32, pad_id=49407)
    _tokenizer.enable_truncation(max_length=32)

    # Pre-build default tiled engine
    default_cats = ['pen', 'box', 'phone', 'bottle', 'toy']
    _get_tiled_engine(default_cats, 0.30)
    _sam3_tiled = True  # sentinel for health check
    print('[INIT] SAM3 ready.')


# ─────────────────── main ───────────────────

_MODELS_DIR = os.environ.get(
    'GRASPFORGE_MODELS_DIR',
    os.path.expanduser('~/workspace/GraspForge/models'))


def main():
    parser = argparse.ArgumentParser(description='Combined SAM3 + FS Server')
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--sam3-engine-dir', type=str,
                        default=os.path.join(_MODELS_DIR, 'SAM3Models', 'engine-models'))
    parser.add_argument('--fs-engine-path', type=str,
                        default=os.path.join(_MODELS_DIR, 'FoundationStereoModels',
                                             'trt_engines', 'fs_vits_576x960_fp16.engine'))
    args = parser.parse_args()

    # LD_LIBRARY_PATH for TensorRT (same as sam3_server.py)
    print(f'[combined_server] GPU={args.gpu}, port={args.port}')

    load_models(args)

    uvicorn.run(app, host=args.host, port=args.port,
                log_level='info', access_log=False)


if __name__ == '__main__':
    main()
