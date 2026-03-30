#!/usr/bin/env python3
"""
SAM3 双相机并发 benchmark — 对齐 percept-full-sam3-stereo-hand-nocdm 生产管线

精确测量：
  1. SAM3 HTTP 检测 (preprocess + encode + http)
  2. Mask 解码 (coco_mask.decode RLE)
  3. 3D 测量 (batch_camera_3d_percept: erode + depth filter + backproject + centroid)
  4. 坐标变换 (matrix multiplication)
  5. 匈牙利融合 (DualCameraMatcher)
  6. ByteTracker3D 跟踪

模式:
  --serial    串行 SAM3 (单路) 性能
  --dual      双相机并发 (2x SAM3) + 后处理 + 融合
  --all       全部测试 (默认)
"""

import sys
import os
import time
import statistics
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import cv2
import numpy as np

# 确保绕过代理
os.environ['no_proxy'] = os.environ.get('no_proxy', '') + ',192.168.112.14'
os.environ['NO_PROXY'] = os.environ.get('NO_PROXY', '') + ',192.168.112.14'

# ROS2 colcon install 路径
_install_py = os.path.join(
    os.path.dirname(__file__), '..', 'install', 'perception',
    'local', 'lib', 'python3.10', 'dist-packages')
if os.path.isdir(_install_py):
    sys.path.insert(0, _install_py)
# 也添加源码路径作为 fallback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'perception', 'src'))

from perception.percept import SAM3Online
from perception.scene_perception_core import SimpleConfig

# ========== 配置 ==========
SAM3_URL = 'http://192.168.112.14:8081'
SAMPLES_DIR = '/data/workspace/MobileManipulator2/src/perception/samples'
DATASETS = ['001', '002', '666', '770']
TEXT_PROMPT = 'pen,box,phone,bottle,toy'

# 与生产管线一致的参数
SAM3_RESIZE = (1280, 720)
SAM3_MIN_SCORE = 0.30
MIN_MASK_AREA = 300
MASK_ERODE_KERNEL = 5
DEPTH_MIN = 0.3
DEPTH_MAX = 10.0
IQR_FACTOR = 1.5
MIN_DEPTH_POINTS = 10

# 典型 RealSense D435i 内参 (720p)
INTRINSICS = {'fx': 640.0, 'fy': 640.0, 'cx': 640.0, 'cy': 360.0}

# 模拟外参矩阵 (identity for benchmark — 不影响 timing)
EXTRINSIC_MATRIX = np.eye(4)


def load_data():
    """加载样本数据"""
    all_data = []
    for ds in DATASETS:
        rgb_path = os.path.join(SAMPLES_DIR, f'{ds}-rgb.jpg')
        depth_path = os.path.join(SAMPLES_DIR, f'{ds}-dpt.png')
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            print(f'  跳过 {ds}: 文件不存在')
            continue
        rgb = cv2.imread(rgb_path)
        depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb is None or depth_mm is None:
            print(f'  跳过 {ds}: 读取失败')
            continue
        depth_m = depth_mm.astype(np.float32) / 1000.0
        all_data.append({'name': ds, 'rgb': rgb, 'depth_mm': depth_mm, 'depth_m': depth_m})
        print(f'  加载 {ds}: RGB {rgb.shape}, Depth {depth_mm.shape}')
    return all_data


def sam3_forward_timed(sam3_service, text, rgb):
    """与 _timed_detection 完全一致的 SAM3 检测 + mask 解码"""
    import pycocotools.mask as coco_mask

    timing = {}

    # Phase 1: SAM3 HTTP
    _timing_inner = {}
    t0 = time.perf_counter()
    raw_result = sam3_service.forward(
        text=text, rgb=rgb,
        min_score=SAM3_MIN_SCORE,
        _timing=_timing_inner,
    )
    t1 = time.perf_counter()
    timing['api_http_ms'] = (t1 - t0) * 1000
    timing['sam3_preprocess_ms'] = _timing_inner.get('sam3_preprocess', 0)
    timing['sam3_encode_ms'] = _timing_inner.get('sam3_encode', 0)
    timing['sam3_http_ms'] = _timing_inner.get('sam3_http', 0)

    # Phase 2: Mask decode (与 _timed_detection 完全一致)
    objects = raw_result.result.get('objects', [])
    if not objects:
        timing['mask_decode_ms'] = 0
        timing['n_objects'] = 0
        return [], timing

    rle_masks = []
    bbox_masks = []
    obj_indices = []
    for i, obj in enumerate(objects):
        mask_rle = obj.get('mask')
        if mask_rle:
            rle_masks.append(mask_rle)
            obj_indices.append(i)
        else:
            bbox_masks.append((i, obj['bbox']))

    decoded_masks = {}
    if rle_masks:
        masks_array = coco_mask.decode(rle_masks)
        if masks_array.ndim == 2:
            masks_array = masks_array[:, :, np.newaxis]
        for idx, obj_idx in enumerate(obj_indices):
            decoded_masks[obj_idx] = masks_array[:, :, idx].astype(np.uint8)

    for obj_idx, bbox in bbox_masks:
        binary_mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        x1, y1, x2, y2 = map(int, bbox)
        binary_mask[y1:y2, x1:x2] = 1
        decoded_masks[obj_idx] = binary_mask

    detections = []
    category_counts = {}
    for i, obj in enumerate(objects):
        binary_mask = decoded_masks.get(i)
        if binary_mask is None or binary_mask.sum() < MIN_MASK_AREA:
            continue
        category = obj.get('category', text)
        score = obj.get('score', 0)
        if category not in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1
        detections.append({
            'object_id': f"{category}_{category_counts[category]}",
            'category': category,
            'score': score,
            'bbox': obj.get('bbox', [0, 0, 0, 0]),
            'mask': binary_mask,
        })

    t2 = time.perf_counter()
    timing['mask_decode_ms'] = (t2 - t1) * 1000
    timing['n_objects'] = len(detections)
    return detections, timing


def batch_3d_percept_timed(depth_m, detections):
    """与 batch_camera_3d_percept 完全一致的 3D 测量（不依赖 ROS）"""
    t0 = time.perf_counter()

    H, W = depth_m.shape[:2]
    fx, fy = INTRINSICS['fx'], INTRINSICS['fy']
    cx, cy = INTRINSICS['cx'], INTRINSICS['cy']
    kernel = np.ones((MASK_ERODE_KERNEL, MASK_ERODE_KERNEL), np.uint8)

    results = []
    for det in detections:
        mask = det['mask']
        bbox = det.get('bbox', [0, 0, 0, 0])

        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(W, int(bbox[2]))
        y2 = min(H, int(bbox[3]))

        if x2 <= x1 or y2 <= y1:
            results.append(None)
            continue

        mask_roi = mask[y1:y2, x1:x2]
        depth_roi = depth_m[y1:y2, x1:x2]

        eroded_roi = cv2.erode(mask_roi, kernel, iterations=1)
        if eroded_roi.sum() < MIN_DEPTH_POINTS:
            results.append(None)
            continue

        ys_local, xs_local = np.where(eroded_roi > 0)
        ds = depth_roi[ys_local, xs_local]

        valid = (ds > DEPTH_MIN) & (ds < DEPTH_MAX)
        if valid.sum() < MIN_DEPTH_POINTS:
            results.append(None)
            continue

        depth_values = ds[valid]
        q1, q3 = np.percentile(depth_values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - IQR_FACTOR * iqr
        upper_bound = q3 + IQR_FACTOR * iqr

        valid2 = valid & (ds >= lower_bound) & (ds <= upper_bound)
        if valid2.sum() < MIN_DEPTH_POINTS:
            results.append(None)
            continue

        xs_img = xs_local[valid2].astype(np.float64) + x1
        ys_img = ys_local[valid2].astype(np.float64) + y1
        ds_valid = ds[valid2]

        X = (xs_img - cx) * ds_valid / fx
        Y = (ys_img - cy) * ds_valid / fy
        Z = ds_valid
        points = np.column_stack([X, Y, Z])
        centroid_optical = np.median(points, axis=0)

        results.append({
            'centroid_optical': centroid_optical,
            'num_points': int(valid2.sum()),
        })

    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000
    valid_count = sum(1 for r in results if r is not None)
    return results, elapsed_ms, valid_count


def coord_transform_timed(measurements, extrinsic_matrix):
    """与 _run_detection_task 完全一致的批量坐标变换"""
    t0 = time.perf_counter()

    optical_centroids = [m['centroid_optical'] for m in measurements if m is not None]
    if not optical_centroids:
        return [], (time.perf_counter() - t0) * 1000

    optical_points = np.array(optical_centroids)
    points_h = np.hstack([optical_points, np.ones((len(optical_points), 1))])
    target_centroids = (extrinsic_matrix @ points_h.T).T[:, :3]

    t1 = time.perf_counter()
    return target_centroids, (t1 - t0) * 1000


def hungarian_fusion_timed(detections_cam1, measurements_cam1,
                            detections_cam2, measurements_cam2):
    """模拟匈牙利匹配融合（与 _fuse_results 对齐）"""
    from scipy.optimize import linear_sum_assignment

    t0 = time.perf_counter()

    # 构建有效目标列表
    objs1 = []
    for det, meas in zip(detections_cam1, measurements_cam1):
        if meas is not None:
            objs1.append({'det': det, 'pos': meas['centroid_optical']})

    objs2 = []
    for det, meas in zip(detections_cam2, measurements_cam2):
        if meas is not None:
            objs2.append({'det': det, 'pos': meas['centroid_optical']})

    if not objs1 or not objs2:
        t1 = time.perf_counter()
        n_fused = len(objs1) + len(objs2)
        return n_fused, 0, (t1 - t0) * 1000

    # 计算距离成本矩阵
    n1, n2 = len(objs1), len(objs2)
    cost_matrix = np.zeros((n1, n2))
    for i, o1 in enumerate(objs1):
        for j, o2 in enumerate(objs2):
            dist = np.linalg.norm(o1['pos'] - o2['pos'])
            cat_cost = 0.0 if o1['det']['category'] == o2['det']['category'] else 0.3
            cost_matrix[i, j] = dist + cat_cost

    # 匈牙利匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    max_cost = 0.7
    matched = 0
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < max_cost:
            matched += 1

    n_fused = len(objs1) + len(objs2) - matched

    t1 = time.perf_counter()
    return n_fused, matched, (t1 - t0) * 1000


def bytetracker_update_timed(n_objects):
    """模拟 ByteTracker3D.update 的计时（构建输入 + 距离计算）"""
    t0 = time.perf_counter()

    # 模拟 tracker 输入构建和距离矩阵计算
    positions = np.random.randn(n_objects, 3).astype(np.float64)
    if n_objects > 1:
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(positions, positions)

    t1 = time.perf_counter()
    return (t1 - t0) * 1000


def print_stats(label, times_ms):
    """打印统计"""
    if not times_ms:
        print(f'  {label}: 无数据')
        return
    avg = statistics.mean(times_ms)
    std = statistics.stdev(times_ms) if len(times_ms) > 1 else 0
    mn = min(times_ms)
    mx = max(times_ms)
    p50 = sorted(times_ms)[len(times_ms) // 2]
    p95 = sorted(times_ms)[int(len(times_ms) * 0.95)]
    print(f'  {label:<28}: avg={avg:7.1f}ms  p50={p50:7.1f}ms  p95={p95:7.1f}ms  '
          f'min={mn:7.1f}ms  max={mx:7.1f}ms  std={std:5.1f}ms')


def run_serial_benchmark(sam3_service, all_data, num_runs, warmup):
    """串行 SAM3 benchmark — 测量单路检测全链路"""
    n = len(all_data)

    print(f"\n{'='*80}")
    print(f'[串行] SAM3 单路检测全链路 ({num_runs} 次, 预热 {warmup} 次, {n} 组数据)')
    print(f"{'='*80}")

    # 收集各阶段 timing
    t_api_http = []
    t_preprocess = []
    t_encode = []
    t_http = []
    t_mask_decode = []
    t_3d_measure = []
    t_coord_tf = []
    t_wall = []
    obj_counts = []

    # 预热
    for i in range(warmup):
        d = all_data[i % n]
        sam3_forward_timed(sam3_service, TEXT_PROMPT, d['rgb'])
        print(f'  [预热 {i+1}/{warmup}] {d["name"]}')

    # 正式测试
    for r in range(num_runs):
        for d in all_data:
            tw0 = time.perf_counter()

            # Step 1: SAM3 检测 + mask 解码
            detections, det_timing = sam3_forward_timed(sam3_service, TEXT_PROMPT, d['rgb'])

            # Step 2: 3D 测量
            measurements, ms_3d, valid_count = batch_3d_percept_timed(d['depth_m'], detections)

            # Step 3: 坐标变换
            _, ms_tf = coord_transform_timed(measurements, EXTRINSIC_MATRIX)

            tw1 = time.perf_counter()
            wall_ms = (tw1 - tw0) * 1000

            t_api_http.append(det_timing['api_http_ms'])
            t_preprocess.append(det_timing['sam3_preprocess_ms'])
            t_encode.append(det_timing['sam3_encode_ms'])
            t_http.append(det_timing['sam3_http_ms'])
            t_mask_decode.append(det_timing['mask_decode_ms'])
            t_3d_measure.append(ms_3d)
            t_coord_tf.append(ms_tf)
            t_wall.append(wall_ms)
            obj_counts.append(det_timing['n_objects'])

            idx = len(t_wall)
            print(f'  #{idx:3d} [{d["name"]}] wall={wall_ms:6.1f}ms | '
                  f'pre={det_timing["sam3_preprocess_ms"]:.0f} '
                  f'enc={det_timing["sam3_encode_ms"]:.0f} '
                  f'http={det_timing["sam3_http_ms"]:.0f} '
                  f'mask={det_timing["mask_decode_ms"]:.1f} '
                  f'3d={ms_3d:.1f} tf={ms_tf:.2f} | '
                  f'{det_timing["n_objects"]} objs')

    # 统计
    print(f"\n{'─'*80}")
    print(f'串行单路统计 ({len(t_wall)} 次)')
    print(f"{'─'*80}")
    print_stats('SAM3 预处理 (resize)', t_preprocess)
    print_stats('SAM3 JPEG编码', t_encode)
    print_stats('SAM3 HTTP请求', t_http)
    print_stats('SAM3 API总耗时', t_api_http)
    print_stats('Mask RLE解码', t_mask_decode)
    print_stats('3D 测量 (batch_percept)', t_3d_measure)
    print_stats('坐标变换 (matrix)', t_coord_tf)
    print(f"  {'─'*76}")
    print_stats('单路端到端 wall', t_wall)

    avg_wall = statistics.mean(t_wall)
    avg_objs = statistics.mean(obj_counts)
    print(f'\n  单路检测频率: {1000/avg_wall:.1f} Hz (avg {avg_objs:.1f} objs/frame)')

    return {
        'api_http': t_api_http, 'preprocess': t_preprocess, 'encode': t_encode,
        'http': t_http, 'mask_decode': t_mask_decode, '3d_measure': t_3d_measure,
        'coord_tf': t_coord_tf, 'wall': t_wall, 'obj_counts': obj_counts,
    }


def run_dual_benchmark(sam3_service, all_data, num_runs, warmup):
    """双相机并发 benchmark — 完整模拟 percept-full-sam3-stereo-hand-nocdm"""
    n = len(all_data)

    print(f"\n{'='*80}")
    print(f'[双相机] 2x SAM3 并发 + 后处理 + 融合 + 跟踪')
    print(f'  ({num_runs} 次, 预热 {warmup} 次, {n} 组数据)')
    print(f"{'='*80}")

    # 阶段计时
    t_parallel_sam3 = []     # 2x SAM3 并发 wall time
    t_sam3_cam1 = []         # cam1 SAM3 总耗时
    t_sam3_cam2 = []         # cam2 SAM3 总耗时
    t_3d_cam1 = []
    t_3d_cam2 = []
    t_tf_cam1 = []
    t_tf_cam2 = []
    t_post_total = []        # 双路后处理总计
    t_fusion = []            # 匈牙利融合
    t_tracking = []          # ByteTracker
    t_wall_total = []        # 全链路 wall
    t_idle = []              # 调度开销
    obj_counts_fused = []

    executor = ThreadPoolExecutor(max_workers=4)

    def run_one_camera(data):
        """单相机检测全链路: SAM3 + mask + 3D + TF"""
        t0 = time.perf_counter()
        detections, det_timing = sam3_forward_timed(sam3_service, TEXT_PROMPT, data['rgb'])
        measurements, ms_3d, valid_count = batch_3d_percept_timed(data['depth_m'], detections)
        _, ms_tf = coord_transform_timed(measurements, EXTRINSIC_MATRIX)
        t1 = time.perf_counter()
        return {
            'detections': detections,
            'measurements': measurements,
            'det_timing': det_timing,
            'ms_3d': ms_3d,
            'ms_tf': ms_tf,
            'wall_ms': (t1 - t0) * 1000,
        }

    # 预热
    for i in range(warmup):
        d1 = all_data[i % n]
        d2 = all_data[(i + 1) % n]
        f1 = executor.submit(run_one_camera, d1)
        f2 = executor.submit(run_one_camera, d2)
        f1.result(timeout=30)
        f2.result(timeout=30)
        print(f'  [预热 {i+1}/{warmup}] {d1["name"]}+{d2["name"]}')

    # 正式测试
    for r in range(num_runs):
        for di in range(n):
            d1 = all_data[di]
            d2 = all_data[(di + 1) % n]

            tw0 = time.perf_counter()

            # Phase 1: 双相机并发 SAM3 + mask + 3D + TF
            tp0 = time.perf_counter()
            f1 = executor.submit(run_one_camera, d1)
            f2 = executor.submit(run_one_camera, d2)
            r1 = f1.result(timeout=30)
            r2 = f2.result(timeout=30)
            tp1 = time.perf_counter()
            parallel_ms = (tp1 - tp0) * 1000

            # Phase 2: 融合
            n_fused, n_matched, ms_fusion = hungarian_fusion_timed(
                r1['detections'], r1['measurements'],
                r2['detections'], r2['measurements'])

            # Phase 3: 跟踪
            ms_track = bytetracker_update_timed(n_fused)

            tw1 = time.perf_counter()
            wall_total = (tw1 - tw0) * 1000

            # 收集数据
            cam1_total = r1['det_timing']['api_http_ms'] + r1['det_timing']['mask_decode_ms'] + r1['ms_3d'] + r1['ms_tf']
            cam2_total = r2['det_timing']['api_http_ms'] + r2['det_timing']['mask_decode_ms'] + r2['ms_3d'] + r2['ms_tf']
            idle = parallel_ms - max(r1['wall_ms'], r2['wall_ms'])

            t_parallel_sam3.append(parallel_ms)
            t_sam3_cam1.append(r1['wall_ms'])
            t_sam3_cam2.append(r2['wall_ms'])
            t_3d_cam1.append(r1['ms_3d'])
            t_3d_cam2.append(r2['ms_3d'])
            t_tf_cam1.append(r1['ms_tf'])
            t_tf_cam2.append(r2['ms_tf'])
            t_post_total.append(r1['ms_3d'] + r1['ms_tf'] + r2['ms_3d'] + r2['ms_tf'])
            t_fusion.append(ms_fusion)
            t_tracking.append(ms_track)
            t_wall_total.append(wall_total)
            t_idle.append(idle)
            obj_counts_fused.append(n_fused)

            idx = len(t_wall_total)
            n1 = r1['det_timing']['n_objects']
            n2 = r2['det_timing']['n_objects']
            print(f'  #{idx:3d} [{d1["name"]}+{d2["name"]}] '
                  f'wall={wall_total:6.1f}ms | '
                  f'parallel={parallel_ms:.0f}ms(idle={idle:.0f}) '
                  f'cam1={r1["wall_ms"]:.0f}ms({n1}obj) '
                  f'cam2={r2["wall_ms"]:.0f}ms({n2}obj) '
                  f'fuse={ms_fusion:.1f}ms track={ms_track:.1f}ms '
                  f'→ {n_fused}obj({n_matched}matched)')

    executor.shutdown(wait=False)

    # 统计
    print(f"\n{'─'*80}")
    print(f'双相机并发统计 ({len(t_wall_total)} 次)')
    print(f"{'─'*80}")

    print(f'\n  [Phase 1] 并发检测 (2x SAM3 + mask + 3D + TF):')
    print_stats('cam1 全链路', t_sam3_cam1)
    print_stats('cam2 全链路', t_sam3_cam2)
    print_stats('并发 wall (max+调度)', t_parallel_sam3)
    print_stats('调度空闲开销', t_idle)

    print(f'\n  [Phase 2] 后处理:')
    print_stats('3D测量 cam1', t_3d_cam1)
    print_stats('3D测量 cam2', t_3d_cam2)
    print_stats('坐标变换 cam1', t_tf_cam1)
    print_stats('坐标变换 cam2', t_tf_cam2)

    print(f'\n  [Phase 3] 融合 + 跟踪:')
    print_stats('匈牙利融合', t_fusion)
    print_stats('ByteTracker 跟踪', t_tracking)

    print(f'\n  [总计]:')
    print_stats('端到端 wall', t_wall_total)

    avg_wall = statistics.mean(t_wall_total)
    avg_parallel = statistics.mean(t_parallel_sam3)
    avg_fusion = statistics.mean(t_fusion)
    avg_track = statistics.mean(t_tracking)
    avg_objs = statistics.mean(obj_counts_fused)

    hz = 1000 / avg_wall
    print(f'\n  双相机融合检测频率: {hz:.2f} Hz')
    print(f'  每帧平均目标数: {avg_objs:.1f}')

    # 时间分布饼图
    total = avg_wall
    print(f'\n  时间分布:')
    print(f'    并发检测:  {avg_parallel:6.1f}ms ({avg_parallel/total*100:4.1f}%)')
    print(f'    匈牙利融合: {avg_fusion:6.1f}ms ({avg_fusion/total*100:4.1f}%)')
    print(f'    跟踪:      {avg_track:6.1f}ms ({avg_track/total*100:4.1f}%)')

    # 进一步分解并发检测内部
    avg_cam1 = statistics.mean(t_sam3_cam1)
    avg_cam1_http = statistics.mean([t_sam3_cam1[i] - t_3d_cam1[i] - t_tf_cam1[i]
                                      for i in range(len(t_sam3_cam1))])
    avg_3d = statistics.mean(t_3d_cam1)
    avg_tf = statistics.mean(t_tf_cam1)

    print(f'\n  单相机内部分解 (cam1 平均):')
    print(f'    SAM3 API+mask: {avg_cam1_http:6.1f}ms ({avg_cam1_http/avg_cam1*100:4.1f}%)')
    print(f'    3D测量:        {avg_3d:6.1f}ms ({avg_3d/avg_cam1*100:4.1f}%)')
    print(f'    坐标变换:      {avg_tf:6.1f}ms ({avg_tf/avg_cam1*100:4.1f}%)')

    # 可达性判断
    for target_hz in [5.0, 8.0, 10.0, 15.0]:
        target_ms = 1000.0 / target_hz
        ok = avg_wall < target_ms
        p95 = sorted(t_wall_total)[int(len(t_wall_total) * 0.95)]
        ok_p95 = p95 < target_ms
        status = 'OK' if ok else 'NO'
        status_p95 = 'OK' if ok_p95 else 'NO'
        print(f'\n  {target_hz:5.1f} Hz 可达性: [{status}] avg={avg_wall:.1f}ms < {target_ms:.0f}ms '
              f'| [{status_p95}] p95={p95:.1f}ms < {target_ms:.0f}ms')

    return {
        'wall_total': t_wall_total,
        'parallel': t_parallel_sam3,
        'fusion': t_fusion,
        'tracking': t_tracking,
    }


def main():
    parser = argparse.ArgumentParser(description='SAM3 双相机并发 Benchmark')
    parser.add_argument('--serial', action='store_true', help='串行 SAM3 单路')
    parser.add_argument('--dual', action='store_true', help='双相机并发 + 融合')
    parser.add_argument('--all', action='store_true', help='全部测试 (默认)')
    parser.add_argument('--num-runs', type=int, default=5, help='每组数据测试轮次 (default: 5)')
    parser.add_argument('--warmup', type=int, default=3, help='预热次数 (default: 3)')
    args = parser.parse_args()

    if not args.serial and not args.dual:
        args.all = True

    print(f'SAM3 双相机并发 Benchmark')
    print(f'  SAM3 URL: {SAM3_URL}')
    print(f'  Prompt: {TEXT_PROMPT}')
    print(f'  Resize: {SAM3_RESIZE}')
    print(f'  Runs: {args.num_runs}, Warmup: {args.warmup}')

    print(f'\n加载样本数据...')
    all_data = load_data()
    if not all_data:
        print('没有可用的测试数据')
        sys.exit(1)
    print(f'共 {len(all_data)} 组数据\n')

    # 初始化 SAM3
    print('初始化 SAM3 服务...')
    cfg = SimpleConfig(
        url=SAM3_URL,
        min_score=SAM3_MIN_SCORE,
        resize=SAM3_RESIZE,
        return_mask=True,
        warmup=0,
    )
    sam3_service = SAM3Online(cfg)
    print(f'SAM3 初始化完成\n')

    if args.serial or args.all:
        run_serial_benchmark(sam3_service, all_data, args.num_runs, args.warmup)

    if args.dual or args.all:
        run_dual_benchmark(sam3_service, all_data, args.num_runs, args.warmup)


if __name__ == '__main__':
    main()
