#!/usr/bin/env python3
"""FS 深度后处理诊断 — 离线跑 Combined Server，OpenCV 交互式对比 raw / FS 深度 + 3D 定位。

用法:
  export PATH="/usr/bin:$PATH"
  DISPLAY=:1 python3 scripts/_cc_fs_depth_diag.py /home/didi/capture/top_capture
  DISPLAY=:1 python3 scripts/_cc_fs_depth_diag.py /home/didi/capture/top_capture --prompt "bottle"
  DISPLAY=:1 python3 scripts/_cc_fs_depth_diag.py /home/didi/capture/top_capture --camera chassis

交互:
  - 主窗口显示 RGB + 检测框，点击物体查看深度详情
  - 左右箭头 / n/p 键切换物体
  - d 键切换全帧深度对比
  - q / ESC 退出
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'perception', 'src'))
from percept import CombinedPerceptionClient  # noqa: E402
from coordinate_transformer import CoordinateTransformer  # noqa: E402

# ── 算法常量 (与 scene_perception_core.py 一致) ──
DEPTH_MIN = 0.3
DEPTH_MAX = 10.0
MASK_ERODE_KERNEL = 5
IQR_FACTOR = 1.5
MIN_DEPTH_POINTS = 10
NEAR_RANGE_ZERO_THRESH = 0.3
NEAR_RANGE_DEPTH_MIN = 0.05
NEAR_RANGE_DEPTH_MAX = 0.5
DEPTH_SPREAD_THRESH = 0.3
DEPTH_CLUSTER_BAND = 0.3
DEPTH_REF_PERCENTILE = 10

COLORS = [
    (0, 255, 0), (255, 100, 0), (0, 200, 255), (255, 0, 255),
    (0, 255, 255), (128, 0, 255), (255, 128, 0), (0, 128, 255),
    (200, 200, 0), (128, 255, 0), (255, 0, 128), (0, 255, 128),
]


def load_capture(capture_dir, frame_idx=0):
    idx = f'{frame_idx:03d}'
    rgb = cv2.imread(os.path.join(capture_dir, 'rgb', f'{idx}.png'))
    ir_left = cv2.imread(os.path.join(capture_dir, 'ir_left', f'{idx}.png'), cv2.IMREAD_GRAYSCALE)
    ir_right = cv2.imread(os.path.join(capture_dir, 'ir_right', f'{idx}.png'), cv2.IMREAD_GRAYSCALE)
    raw_depth = cv2.imread(os.path.join(capture_dir, 'depth', f'{idx}.png'), cv2.IMREAD_UNCHANGED)

    with open(os.path.join(capture_dir, 'intrinsics.yaml')) as f:
        intr = yaml.safe_load(f)

    for name, img in [('RGB', rgb), ('IR left', ir_left), ('IR right', ir_right), ('Raw depth', raw_depth)]:
        assert img is not None, f'{name} not found in {capture_dir}'

    color_profile = intr['profiles'][0]
    color_intr = {
        'fx': color_profile['fx'], 'fy': color_profile['fy'],
        'cx': color_profile['cx'], 'cy': color_profile['cy'],
        'width': color_profile['image_width'],
        'height': color_profile['image_height'],
    }
    return rgb, ir_left, ir_right, raw_depth, intr, color_intr


def depth_colormap(depth_m, vmin=0.1, vmax=5.0):
    d = np.clip(depth_m, vmin, vmax)
    d = ((d - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    d[depth_m < 0.01] = 0
    return cv2.applyColorMap(d, cv2.COLORMAP_TURBO)


def run_pipeline_on_depth(depth_m, mask, bbox):
    """走完 batch_camera_3d_percept 深度后处理，返回各步骤详情。"""
    H, W = depth_m.shape[:2]
    x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
    x2 = min(W, int(bbox[2])); y2 = min(H, int(bbox[3]))
    steps = []

    mask_roi = mask[y1:y2, x1:x2]
    depth_roi = depth_m[y1:y2, x1:x2]

    # 0. 原始 mask
    ys0, xs0 = np.where(mask_roi > 0)
    ds0 = depth_roi[ys0, xs0]
    steps.append(('0.Mask原始', len(ds0), int((ds0 > 0.01).sum()),
                  f'nonzero={int((ds0>0.01).sum())}/{len(ds0)}'))

    # 1. 腐蚀
    kernel = np.ones((MASK_ERODE_KERNEL, MASK_ERODE_KERNEL), np.uint8)
    eroded = cv2.erode(mask_roi, kernel, iterations=1)
    if eroded.sum() < MIN_DEPTH_POINTS:
        steps.append(('1.腐蚀', 0, 0, 'FAIL: 腐蚀后不足'))
        return steps, None, None, None

    ys, xs = np.where(eroded > 0)
    ds = depth_roi[ys, xs]
    steps.append(('1.腐蚀5x5', int(mask_roi.sum()), int(eroded.sum()),
                  f'{int(mask_roi.sum())}→{int(eroded.sum())} (-{int(mask_roi.sum())-int(eroded.sum())})'))

    # 2. 非零
    nonzero = ds[ds > 0.01]
    if len(nonzero) == 0:
        steps.append(('2.非零', 0, 0, 'FAIL: 全零'))
        return steps, None, None, None
    steps.append(('2.非零过滤', len(ds), len(nonzero),
                  f'zero={len(ds)-len(nonzero)} ({(len(ds)-len(nonzero))/max(len(ds),1)*100:.1f}%)'))

    # 3. 近距离
    p25 = np.percentile(nonzero, 25)
    is_near = p25 < DEPTH_MIN
    if is_near:
        zero_ratio = 1.0 - len(nonzero) / len(ds)
        if zero_ratio > NEAR_RANGE_ZERO_THRESH:
            steps.append(('3.近距离', 0, 0, f'FAIL: 零值{zero_ratio:.0%}'))
            return steps, None, None, None
        d_min, d_max = NEAR_RANGE_DEPTH_MIN, NEAR_RANGE_DEPTH_MAX
    else:
        d_min, d_max = DEPTH_MIN, DEPTH_MAX
    steps.append(('3.近距离判定', 0, 0,
                  f'P25={p25:.3f}m {"YES" if is_near else "NO"} → [{d_min},{d_max}]m'))

    # 4. 范围过滤
    valid = (ds > d_min) & (ds < d_max)
    steps.append(('4.范围过滤', len(ds), int(valid.sum()),
                  f'{len(ds)}→{int(valid.sum())} (-{len(ds)-int(valid.sum())})'))
    if valid.sum() < MIN_DEPTH_POINTS:
        steps.append(('4.范围', 0, 0, 'FAIL: 不足'))
        return steps, None, None, None
    depth_values = ds[valid]

    # 5. 背景穿透（仅近距离）
    if is_near:
        spread = depth_values.max() - depth_values.min()
        if spread > DEPTH_SPREAD_THRESH:
            ref = np.percentile(depth_values, DEPTH_REF_PERCENTILE)
            before = len(depth_values)
            depth_values = depth_values[depth_values <= ref + DEPTH_CLUSTER_BAND]
            steps.append(('5.背景穿透', before, len(depth_values),
                          f'spread={spread:.3f}m ref={ref:.3f}m → {before}→{len(depth_values)}'))
        else:
            steps.append(('5.背景穿透', 0, 0, f'skip: spread={spread:.3f}m<{DEPTH_SPREAD_THRESH}m'))
    else:
        steps.append(('5.背景穿透', 0, 0, 'skip: 非近距离'))

    if len(depth_values) < MIN_DEPTH_POINTS:
        return steps, None, None, None

    # 6. IQR
    q1, q3 = np.percentile(depth_values, [25, 75])
    iqr = q3 - q1
    lower = q1 - IQR_FACTOR * iqr
    upper = q3 + IQR_FACTOR * iqr
    before = len(depth_values)
    depth_values = depth_values[(depth_values >= lower) & (depth_values <= upper)]
    steps.append(('6.IQR剔除', before, len(depth_values),
                  f'[{lower:.3f},{upper:.3f}]m → {before}→{len(depth_values)} (-{before-len(depth_values)})'))

    if len(depth_values) < MIN_DEPTH_POINTS:
        return steps, None, None, None

    # 最终 valid2
    valid2 = valid & (ds >= lower) & (ds <= upper)
    med = np.median(depth_values)
    std = np.std(depth_values)
    steps.append(('7.最终', int(valid2.sum()), int(valid2.sum()),
                  f'{int(valid2.sum())}pts median={med:.4f}m std={std:.4f}m'))

    # 构建过滤后 mask (全图尺寸) 用于可视化
    filtered_mask = np.zeros((H, W), dtype=np.uint8)
    xs_valid = xs[valid2] + x1
    ys_valid = ys[valid2] + y1
    filtered_mask[ys_valid, xs_valid] = 255

    return steps, (ys, xs, valid2, y1, x1), depth_values, filtered_mask


def compute_3d(ys_local, xs_local, valid2, y1, x1, depth_roi, intr):
    ds = depth_roi[ys_local, xs_local]
    xs_img = xs_local[valid2].astype(np.float64) + x1
    ys_img = ys_local[valid2].astype(np.float64) + y1
    ds_v = ds[valid2]
    X = (xs_img - intr['cx']) * ds_v / intr['fx']
    Y = (ys_img - intr['cy']) * ds_v / intr['fy']
    Z = ds_v
    centroid = np.array([np.median(X), np.median(Y), np.median(Z)])
    return centroid, len(ds_v)


def project_px(centroid, intr):
    u = centroid[0] * intr['fx'] / centroid[2] + intr['cx']
    v = centroid[1] * intr['fy'] / centroid[2] + intr['cy']
    return int(round(u)), int(round(v))


def put_text(img, text, pos, scale=0.45, color=(255, 255, 255), thickness=1, bg=True):
    if bg:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(img, (pos[0]-1, pos[1]-th-3), (pos[0]+tw+1, pos[1]+3), (0, 0, 0), -1)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# ───────────────── 交互式主流程 ─────────────────

class DiagUI:
    def __init__(self, rgb, raw_depth_m, fs_depth_m, objects, color_intr,
                 optical_to_base=None):
        self.rgb = rgb
        self.raw_depth_m = raw_depth_m
        self.fs_depth_m = fs_depth_m
        self.objects = objects
        self.intr = color_intr
        self.optical_to_base = optical_to_base  # 4x4 matrix or None
        self.H, self.W = rgb.shape[:2]
        self.selected = -1  # -1 = overview
        self.show_depth_full = False

        # 预解码所有 mask
        import pycocotools.mask as coco_mask
        self.masks = []
        self.bboxes = []
        self.categories = []
        for obj in objects:
            bbox = obj.get('bbox', [0, 0, 0, 0])
            self.bboxes.append([int(v) for v in bbox])
            self.categories.append(obj.get('class_name', obj.get('category', 'object')))
            mask_rle = obj.get('mask')
            if mask_rle:
                rle = {'counts': mask_rle['counts'].encode() if isinstance(mask_rle['counts'], str)
                       else mask_rle['counts'], 'size': mask_rle['size']}
                self.masks.append(coco_mask.decode(rle))
            else:
                m = np.zeros((self.H, self.W), dtype=np.uint8)
                x1, y1, x2, y2 = self.bboxes[-1]
                m[y1:y2, x1:x2] = 1
                self.masks.append(m)

        # 预计算所有物体的 pipeline 结果
        self.results = []  # list of (raw_steps, raw_centroid, fs_steps, fs_centroid, ...)
        for i in range(len(objects)):
            r = self._compute_object(i)
            self.results.append(r)

    def _compute_object(self, idx):
        mask = self.masks[idx]
        bbox = self.bboxes[idx]
        H, W = self.H, self.W
        x1, y1, x2, y2 = max(0,bbox[0]), max(0,bbox[1]), min(W,bbox[2]), min(H,bbox[3])

        res = {}
        for name, depth in [('raw', self.raw_depth_m), ('fs', self.fs_depth_m)]:
            steps, indices, dv, fmask = run_pipeline_on_depth(depth, mask, bbox)
            centroid = None
            centroid_base = None
            npts = 0
            if indices is not None:
                ys, xs, valid2, y1r, x1r = indices
                depth_roi = depth[y1:y2, x1:x2]
                centroid, npts = compute_3d(ys, xs, valid2, y1r, x1r, depth_roi, self.intr)
                if centroid is not None and self.optical_to_base is not None:
                    p_h = np.append(centroid, 1.0)
                    centroid_base = (self.optical_to_base @ p_h)[:3]
            res[name] = {
                'steps': steps, 'centroid': centroid, 'centroid_base': centroid_base,
                'npts': npts, 'filtered_mask': fmask,
            }
        return res

    def _draw_overview(self):
        """主视图：RGB + 所有检测框 + 3D 投影点"""
        img = self.rgb.copy()
        for i, (bbox, cat) in enumerate(zip(self.bboxes, self.categories)):
            color = COLORS[i % len(COLORS)]
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            score = self.objects[i].get('score', 0)
            put_text(img, f'{i}:{cat} {score:.2f}', (x1, y1 - 5), color=color)

            r = self.results[i]
            # Raw centroid: 红色十字
            raw_c = r['raw']['centroid']
            if raw_c is not None:
                u, v = project_px(raw_c, self.intr)
                cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
                rb = r['raw']['centroid_base']
                rlbl = f'R:{np.linalg.norm(rb):.2f}m' if rb is not None else f'R:{raw_c[2]:.2f}m'
                put_text(img, rlbl, (u+10, v-5), scale=0.35, color=(0, 0, 255))

            # FS centroid: 蓝色十字
            fs_c = r['fs']['centroid']
            if fs_c is not None:
                u, v = project_px(fs_c, self.intr)
                cv2.drawMarker(img, (u, v), (255, 100, 0), cv2.MARKER_CROSS, 14, 2)
                fb = r['fs']['centroid_base']
                flbl = f'F:{np.linalg.norm(fb):.2f}m' if fb is not None else f'F:{fs_c[2]:.2f}m'
                put_text(img, flbl, (u+10, v+12), scale=0.35, color=(255, 100, 0))

        # 帮助文本
        frame_tag = 'base_link' if self.optical_to_base is not None else 'optical(no extrinsics)'
        help_lines = [
            'Click object / n,p: select | d: depth view | q: quit',
            f'Objects: {len(self.objects)}  Red=Raw  Blue=FS  frame={frame_tag}',
        ]
        for hi, hl in enumerate(help_lines):
            put_text(img, hl, (10, self.H - 10 - (len(help_lines)-1-hi)*20), scale=0.4, color=(200, 200, 200))
        return img

    def _draw_depth_full(self):
        """全帧深度对比：Raw | FS | Diff"""
        valid_all = np.concatenate([self.raw_depth_m[self.raw_depth_m > 0.1],
                                     self.fs_depth_m[self.fs_depth_m > 0.1]])
        vmin, vmax = np.percentile(valid_all, [2, 98])

        raw_cm = depth_colormap(self.raw_depth_m, vmin, vmax)
        fs_cm = depth_colormap(self.fs_depth_m, vmin, vmax)
        diff = np.abs(self.fs_depth_m - self.raw_depth_m)
        diff[self.raw_depth_m < 0.1] = 0
        diff_cm = depth_colormap(diff, 0, 0.5)

        # bbox overlay
        for i, bbox in enumerate(self.bboxes):
            x1, y1, x2, y2 = bbox
            for cm in [raw_cm, fs_cm]:
                cv2.rectangle(cm, (x1, y1), (x2, y2), (255, 255, 255), 1)
                put_text(cm, str(i), (x1+2, y1+14), scale=0.4)

        pad = np.zeros((self.H, 4, 3), dtype=np.uint8)
        row = np.hstack([raw_cm, pad, fs_cm, pad, diff_cm])

        # 缩放到合理大小
        scale = min(1.0, 1800 / row.shape[1])
        if scale < 1.0:
            row = cv2.resize(row, None, fx=scale, fy=scale)

        # 标签
        header = np.zeros((30, row.shape[1], 3), dtype=np.uint8)
        sw = int(self.W * scale)
        put_text(header, 'Raw Depth', (10, 22), scale=0.5)
        put_text(header, 'FS Depth', (sw + 10, 22), scale=0.5)
        put_text(header, '|FS - Raw|', (2*sw + 10, 22), scale=0.5)
        return np.vstack([header, row])

    def _draw_object_detail(self, idx):
        """选中物体的详细对比面板"""
        bbox = self.bboxes[idx]
        cat = self.categories[idx]
        mask = self.masks[idx]
        r = self.results[idx]
        x1, y1, x2, y2 = bbox

        margin = 30
        vx1 = max(0, x1 - margin); vy1 = max(0, y1 - margin)
        vx2 = min(self.W, x2 + margin); vy2 = min(self.H, y2 + margin)

        # ── 左侧: ROI 对比 ──
        rgb_roi = self.rgb[vy1:vy2, vx1:vx2].copy()
        raw_roi = self.raw_depth_m[vy1:vy2, vx1:vx2]
        fs_roi = self.fs_depth_m[vy1:vy2, vx1:vx2]
        mask_roi = mask[vy1:vy2, vx1:vx2]

        # 统一色标
        both = np.concatenate([raw_roi[raw_roi > 0.1], fs_roi[fs_roi > 0.1]])
        if len(both) > 0:
            vm, vx = np.percentile(both, [2, 98])
        else:
            vm, vx = 0.1, 5.0
        raw_cm = depth_colormap(raw_roi, vm, vx)
        fs_cm = depth_colormap(fs_roi, vm, vx)

        # Diff
        diff = np.abs(fs_roi - raw_roi)
        diff[(raw_roi < 0.1) | (fs_roi < 0.1)] = 0
        diff_cm = depth_colormap(diff, 0, 0.3)

        # Filtered depth 可视化
        raw_filt_cm = np.zeros_like(raw_cm)
        fs_filt_cm = np.zeros_like(fs_cm)
        for name, fcm in [('raw', raw_filt_cm), ('fs', fs_filt_cm)]:
            fmask = r[name]['filtered_mask']
            if fmask is not None:
                fmask_roi = fmask[vy1:vy2, vx1:vx2]
                dm = self.raw_depth_m if name == 'raw' else self.fs_depth_m
                d_roi = dm[vy1:vy2, vx1:vx2].copy()
                d_roi[fmask_roi == 0] = 0
                fcm[:] = depth_colormap(d_roi, vm, vx)

        # Mask 轮廓叠加
        contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cm in [rgb_roi, raw_cm, fs_cm, diff_cm, raw_filt_cm, fs_filt_cm]:
            cv2.drawContours(cm, contours, -1, (255, 255, 255), 1)

        # 投影质心
        for name, color in [('raw', (0, 0, 255)), ('fs', (255, 100, 0))]:
            c = r[name]['centroid']
            if c is not None:
                u, v = project_px(c, self.intr)
                u -= vx1; v -= vy1
                for cm in [rgb_roi, raw_cm if name == 'raw' else fs_cm]:
                    cv2.drawMarker(cm, (u, v), color, cv2.MARKER_CROSS, 10, 2)

        # 拼成 2 行 3 列
        pad_v = np.zeros((rgb_roi.shape[0], 3, 3), dtype=np.uint8)
        row1 = np.hstack([rgb_roi, pad_v, raw_cm, pad_v, fs_cm])
        row2 = np.hstack([diff_cm, pad_v, raw_filt_cm, pad_v, fs_filt_cm])
        pad_h = np.zeros((3, row1.shape[1], 3), dtype=np.uint8)
        grid = np.vstack([row1, pad_h, row2])

        # 标签
        rw = rgb_roi.shape[1] + 3
        header = np.zeros((22, grid.shape[1], 3), dtype=np.uint8)
        put_text(header, 'RGB', (5, 16), scale=0.4)
        put_text(header, 'Raw Depth', (rw + 5, 16), scale=0.4)
        put_text(header, 'FS Depth', (2*rw + 5, 16), scale=0.4)
        mid_header = np.zeros((22, grid.shape[1], 3), dtype=np.uint8)
        put_text(mid_header, '|FS-Raw|', (5, 16), scale=0.4)
        put_text(mid_header, 'Raw Filtered', (rw + 5, 16), scale=0.4)
        put_text(mid_header, 'FS Filtered', (2*rw + 5, 16), scale=0.4)

        vis_grid = np.vstack([header, row1, pad_h, mid_header, row2])

        # 放大到合理显示尺寸
        min_w = 700
        if vis_grid.shape[1] < min_w:
            scale_f = min_w / vis_grid.shape[1]
            vis_grid = cv2.resize(vis_grid, None, fx=scale_f, fy=scale_f, interpolation=cv2.INTER_NEAREST)

        # ── 右侧: 文字统计面板 ──
        line_h = 18
        lines = []
        lines.append(f'Object {idx}: {cat}  score={self.objects[idx].get("score",0):.2f}')
        lines.append(f'bbox: {bbox}  mask: {int(mask.sum())}px')
        lines.append('')

        for name, label, clr in [('raw', 'RAW', (0,0,255)), ('fs', 'FS', (255,100,0))]:
            lines.append(f'=== {label} ===')
            for step_name, _, _, desc in r[name]['steps']:
                lines.append(f'  {step_name}: {desc}')
            c = r[name]['centroid']
            cb = r[name]['centroid_base']
            if c is not None:
                lines.append(f'  optical: ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})')
                if cb is not None:
                    lines.append(f'  base_lk: ({cb[0]:.3f}, {cb[1]:.3f}, {cb[2]:.3f})')
                    lines.append(f'  dist={np.linalg.norm(cb):.3f}m  pts={r[name]["npts"]}')
                else:
                    lines.append(f'  dist={np.linalg.norm(c):.3f}m(optical)  pts={r[name]["npts"]}')
            else:
                lines.append(f'  3D: FAILED')
            lines.append('')

        # 对比
        rc = r['raw']['centroid_base'] if r['raw']['centroid_base'] is not None else r['raw']['centroid']
        fc = r['fs']['centroid_base'] if r['fs']['centroid_base'] is not None else r['fs']['centroid']
        if rc is not None and fc is not None:
            dz = fc[2] - rc[2]
            d3d = np.linalg.norm(fc - rc)
            lines.append(f'=== DIFF (base_link) ===')
            lines.append(f'  dZ: {dz*1000:.1f}mm')
            lines.append(f'  d3D: {d3d*1000:.1f}mm')
            up, vp = project_px(r['raw']['centroid'], self.intr) if r['raw']['centroid'] is not None else (0, 0)
            uf, vf = project_px(r['fs']['centroid'], self.intr) if r['fs']['centroid'] is not None else (0, 0)
            lines.append(f'  pixel: ({abs(uf-up)}, {abs(vf-vp)})px')

        text_w = 350
        text_h = max(vis_grid.shape[0], (len(lines) + 1) * line_h + 10)
        text_panel = np.zeros((text_h, text_w, 3), dtype=np.uint8)
        for li, line in enumerate(lines):
            color = (200, 200, 200)
            if line.startswith('=== RAW'): color = (100, 100, 255)
            elif line.startswith('=== FS'): color = (255, 150, 100)
            elif line.startswith('=== DIFF'): color = (0, 255, 255)
            put_text(text_panel, line, (5, 16 + li * line_h), scale=0.38, color=color, bg=False)

        # 拼接
        if vis_grid.shape[0] < text_h:
            vis_grid = np.vstack([vis_grid,
                                   np.zeros((text_h - vis_grid.shape[0], vis_grid.shape[1], 3), dtype=np.uint8)])
        elif text_panel.shape[0] < vis_grid.shape[0]:
            text_panel = np.vstack([text_panel,
                                     np.zeros((vis_grid.shape[0] - text_panel.shape[0], text_w, 3), dtype=np.uint8)])

        return np.hstack([vis_grid, text_panel])

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # 检查点击了哪个物体
        for i, (bbox, mask) in enumerate(zip(self.bboxes, self.masks)):
            if y < self.H and x < self.W and mask[y, x] > 0:
                self.selected = i
                return
        self.selected = -1

    def run(self):
        win_main = 'FS Depth Diag'
        win_detail = 'Object Detail'
        cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_main, self._on_mouse)

        prev_selected = None
        while True:
            # 主窗口
            if self.show_depth_full:
                main_img = self._draw_depth_full()
            else:
                main_img = self._draw_overview()
            cv2.imshow(win_main, main_img)

            # 详情窗口
            if self.selected >= 0 and self.selected != prev_selected:
                detail = self._draw_object_detail(self.selected)
                cv2.namedWindow(win_detail, cv2.WINDOW_NORMAL)
                cv2.imshow(win_detail, detail)
                prev_selected = self.selected
            elif self.selected < 0 and prev_selected is not None:
                cv2.destroyWindow(win_detail)
                prev_selected = None

            key = cv2.waitKey(50) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                break
            elif key == ord('d'):
                self.show_depth_full = not self.show_depth_full
            elif key in (ord('n'), 83, 3):  # n or right arrow
                if len(self.objects) > 0:
                    self.selected = (self.selected + 1) % len(self.objects)
            elif key in (ord('p'), 81, 2):  # p or left arrow
                if len(self.objects) > 0:
                    self.selected = (self.selected - 1) % len(self.objects)

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='FS 深度后处理诊断 (交互式)')
    parser.add_argument('capture_dir', help='capture 数据目录')
    parser.add_argument('--frame', type=int, default=0, help='帧索引 (default: 0)')
    parser.add_argument('--prompt', default='object', help='检测 prompt (default: object)')
    parser.add_argument('--server', default='http://192.168.112.14:8090', help='Combined server URL')
    parser.add_argument('--camera', default='top', help='相机名称 (default: top)')
    args = parser.parse_args()

    # 加载
    print('加载数据...')
    rgb, ir_left, ir_right, raw_depth_u16, intr_yaml, color_intr = \
        load_capture(args.capture_dir, args.frame)
    raw_depth_m = raw_depth_u16.astype(np.float32) / 1000.0
    H, W = raw_depth_m.shape[:2]
    print(f'  RGB: {rgb.shape}  Depth: {raw_depth_m.shape}')

    # 自动检测相机名: 从 intrinsics.yaml 的 camera_id 推断 (如 "chassis_d435" → "chassis")
    if args.camera == 'top':  # 仅在未显式指定时自动检测
        camera_id = intr_yaml.get('camera_id', '')
        for prefix in ('chassis', 'hand', 'top'):
            if camera_id.startswith(prefix):
                if prefix != args.camera:
                    print(f'  [AUTO] 从 camera_id="{camera_id}" 检测到相机: {prefix}')
                    args.camera = prefix
                break

    # 调用 Combined Server
    print('调用 Combined Server...')

    class _Cfg:
        def __init__(self, url):
            self.url = url
            self.confidence = 0.30
            self.return_mask = True

    client = CombinedPerceptionClient(_Cfg(args.server))
    _timing = {}
    resp = client.forward(rgb=rgb, ir_left=ir_left, ir_right=ir_right,
                          intrinsics=intr_yaml, text_prompt=args.prompt, _timing=_timing)
    if not resp.get('success'):
        print(f'错误: {resp.get("error")}')
        sys.exit(1)

    print(f'  HTTP: {_timing.get("combined_http",0):.0f}ms  '
          f'SAM3: {_timing.get("server_sam3",0):.0f}ms  FS: {_timing.get("server_fs",0):.0f}ms')

    objects = resp.get('detections', {}).get('objects', [])
    fs_depth_u16 = resp.get('depth')
    if fs_depth_u16 is None:
        print('FS 未返回深度!'); sys.exit(1)
    fs_depth_m = fs_depth_u16.astype(np.float32) / 1000.0
    if fs_depth_m.shape != raw_depth_m.shape:
        fs_depth_m = cv2.resize(fs_depth_m, (W, H), interpolation=cv2.INTER_NEAREST)

    print(f'  检测: {len(objects)} 个物体')

    # 加载外参 (optical → base_link)
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'perception', 'config')
    optical_to_base = None
    try:
        transformer = CoordinateTransformer(config_dir)
        transformer.load_all_extrinsics(camera_name=args.camera)
        optical_to_base = transformer.get_transform('optical_to_base')
        print(f'  外参: optical→base_link loaded ({args.camera})')
    except Exception as e:
        print(f'  [WARN] 无法加载外参, 保持 optical 坐标系: {e}')

    # 打印物体 base_link 坐标汇总
    if objects and optical_to_base is not None:
        print(f'\n  === 物体 base_link 坐标 ({args.camera} camera) ===')
        print(f'  {"#":>2} {"类别":<15} {"X(m)":>8} {"Y(m)":>8} {"Z(m)":>8} {"距离(m)":>8}  (base_link)')
        print(f'  {"-"*65}')

        # 临时计算每个物体的 3D 位置
        import pycocotools.mask as _coco_mask
        for i, obj in enumerate(objects):
            cat = obj.get('class_name', obj.get('category', 'object'))
            bbox = [int(v) for v in obj.get('bbox', [0, 0, 0, 0])]
            mask_rle = obj.get('mask')
            if mask_rle:
                rle = {'counts': mask_rle['counts'].encode() if isinstance(mask_rle['counts'], str)
                       else mask_rle['counts'], 'size': mask_rle['size']}
                mask = _coco_mask.decode(rle)
            else:
                mask = np.zeros((H, W), dtype=np.uint8)
                mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

            # 用 FS depth 计算 3D
            steps, indices, dv, fmask = run_pipeline_on_depth(fs_depth_m, mask, bbox)
            if indices is not None:
                ys, xs, valid2, y1r, x1r = indices
                x1, y1, x2, y2 = max(0,bbox[0]), max(0,bbox[1]), min(W,bbox[2]), min(H,bbox[3])
                depth_roi = fs_depth_m[y1:y2, x1:x2]
                centroid, npts = compute_3d(ys, xs, valid2, y1r, x1r, depth_roi, color_intr)
                if centroid is not None:
                    p_h = np.append(centroid, 1.0)
                    cb = (optical_to_base @ p_h)[:3]
                    dist = np.linalg.norm(cb)
                    print(f'  {i:>2} {cat:<15} {cb[0]:>8.3f} {cb[1]:>8.3f} {cb[2]:>8.3f} {dist:>8.3f}')
                else:
                    print(f'  {i:>2} {cat:<15} {"--":>8} {"--":>8} {"--":>8} {"FAIL":>8}')
            else:
                print(f'  {i:>2} {cat:<15} {"--":>8} {"--":>8} {"--":>8} {"FAIL":>8}')
        print()

    print('启动交互界面... (q退出, n/p切换物体, d深度全图, 鼠标点击选物体)')

    ui = DiagUI(rgb, raw_depth_m, fs_depth_m, objects, color_intr,
                optical_to_base=optical_to_base)
    ui.run()


if __name__ == '__main__':
    main()
