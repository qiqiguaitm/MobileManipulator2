#!/usr/bin/env python3
"""
CDM 深度诊断工具 - chassis 相机
=====================================
用于排查 CDM (Camera Depth Mapper) 深度优化前后的差异问题。
对 chassis 相机的 RGB + 深度图，依次调用 SAM3 检测和 CDM 深度优化，
并通过四格可视化界面直观对比原始深度与 CDM 优化深度的差异。

【前置条件】
  1. chassis 相机已启动（make cam-dual 或 make cam-chassis）
  2. SAM3 服务可访问：http://192.168.112.14:8081
  3. CDM  服务可访问：http://192.168.112.14:8082

【启动方式】
  # 默认参数（prompt="bottle.cup.box"，置信度 0.30）
  export PATH="/usr/bin:$PATH" && python3 scripts/_cc_cdm_depth_diag.py

  # 自定义检测目标和置信度
  python3 scripts/_cc_cdm_depth_diag.py --prompt "bottle.cup" --conf 0.25

  # 自定义面板宽度（默认 640px，两列共 1280px）
  python3 scripts/_cc_cdm_depth_diag.py --panel-w 800

  # 自定义服务地址
  python3 scripts/_cc_cdm_depth_diag.py --sam3-url http://x.x.x.x:8081 --cdm-url http://x.x.x.x:8082

【四格可视化布局】
  ┌─────────────────────┬─────────────────────┐
  │  左上：RGB + SAM3   │  右上：深度差异热图  │
  │  分割掩码 + 质心    │  (CDM - Raw)        │
  ├─────────────────────┼─────────────────────┤
  │  左下：原始深度     │  右下：CDM 优化深度  │
  │  伪彩色 (JET)       │  伪彩色 (JET)       │
  └─────────────────────┴─────────────────────┘
  底部状态栏：显示鼠标点击像素的 3D 坐标

【差异热图颜色约定】
  蓝色  → CDM 深度更大（比原始深）
  红色  → CDM 深度更小（比原始浅）
  深灰  → 差异 ≤ 5mm（基本一致）
  黑色  → 无效像素（深度为 0）

【左上格标注说明】
  - 彩色矩形框：SAM3 检测到的目标 BBox
  - 半透明彩色区域：SAM3 分割掩码
  - 十字标记：质心像素位置
  - CDM Z: x.xxxm  → 使用 CDM 深度计算的质心 Z 轴距离
  - Raw Z: x.xxxm  → 使用原始深度计算的质心 Z 轴距离
  - raw:xxxmm cdm:xxxmm → 质心像素处的单点深度值（直接读取，未平均）

【质心 3D 计算方法（与正式 pipeline 一致）】
  1. 对分割掩码做 5×5 腐蚀（去除边缘噪声）
  2. IQR 异常值过滤（去除深度离群点）
  3. 对所有有效点反投影到相机光学坐标系
  4. 取 X/Y/Z 各自中值作为最终质心（鲁棒中值法）

【鼠标点击 3D 坐标】
  在任意面板单击左键，状态栏显示：
    Pixel(px,py)  Raw:深度mm -> (X,Y,Z)m  CDM:深度mm -> (X,Y,Z)m  Diff:差值mm
  注：点击为单点直读，未做滤波，噪声像素结果可能不准。

【键盘操作】
  d      - 触发 SAM3 检测 + CDM 优化（同时进行，结果刷新四格）
  s      - 仅运行 CDM 优化（不检测，适合纯深度对比）
  q/ESC  - 退出
"""

import argparse
import base64
import threading
import time

import cv2
import numpy as np
import requests
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
import message_filters

# ─── Defaults ───────────────────────────────────────────────────────────────
SAM3_URL     = "http://192.168.112.14:8081"
CDM_URL      = "http://192.168.112.14:8082"
PROMPT       = "bottle.cup.box"
CONF         = 0.30
PANEL_W      = 640   # width of each panel (total canvas = PANEL_W*2)
STATUS_H     = 72
DEPTH_VIS_MM = 3000  # colormap range (mm)
DIFF_CLIP_MM = 300   # diff heatmap clip range (mm)
MASK_ERODE_K = 5

COLORS = [
    (0, 255, 0), (0, 128, 255), (255, 128, 0),
    (255, 0, 128), (0, 255, 128), (128, 0, 255),
    (255, 255, 0), (128, 255, 0),
]


# ─── CDM letterbox (matches percept.py exactly) ────────────────────────────
def letterbox(img, target_long=798, target_short=448, fill=0):
    h, w = img.shape[:2]
    if w >= h:
        tw, th = target_long, target_short
    else:
        tw, th = target_short, target_long
    scale  = min(tw / w, th / h)
    new_w  = int(w * scale)
    new_h  = int(h * scale)
    interp = cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    pt = (th - new_h) // 2
    pl = (tw - new_w) // 2
    padded = cv2.copyMakeBorder(
        resized, pt, th - new_h - pt, pl, tw - new_w - pl,
        cv2.BORDER_CONSTANT, value=fill)
    return padded, scale, pt, pl


def unletterbox(img, orig_h, orig_w, scale, pad_top, pad_left):
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    cropped = img[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    interp = cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=interp)


# ─── RLE mask decoder ──────────────────────────────────────────────────────
def rle_to_mask(rle, h, w):
    try:
        from pycocotools import mask as coco_mask
        return coco_mask.decode(rle).astype(bool)
    except Exception:
        pass
    counts = rle.get("counts", [])
    if isinstance(counts, str):
        # compressed RLE - need pycocotools; fall back to empty
        return np.zeros((h, w), dtype=bool)
    m = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for cnt in counts:
        m[pos:pos + cnt] = val
        pos += cnt
        val = 1 - val
    return m.reshape(h, w, order="F").astype(bool)


# ─── SAM3 HTTP call ────────────────────────────────────────────────────────
def call_sam3(bgr, url, prompt, conf):
    """Returns list of {bbox, score, class_name, mask(bool ndarray or None)}"""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return []
    files = {"images": ("img.jpg", buf.tobytes(), "image/jpeg")}
    data  = {"text_prompt": prompt, "confidence": conf,
             "return_mask": "true", "tiled": "false"}
    try:
        resp = requests.post(f"{url}/api/predict", files=files,
                             data=data, timeout=15,
                             proxies={"http": None, "https": None})
        resp.raise_for_status()
        j = resp.json()
    except Exception as e:
        print(f"[SAM3] request failed: {e}")
        return []
    h, w   = bgr.shape[:2]
    result = []
    for obj in j.get("results", [{}])[0].get("objects", []):
        mask = None
        rle  = obj.get("mask")
        if rle:
            sz  = rle.get("size", [h, w])
            mask = rle_to_mask(rle, sz[0], sz[1])
            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST).astype(bool)
        result.append({
            "bbox":       obj.get("bbox", [0, 0, 0, 0]),
            "score":      obj.get("score", 0.0),
            "class_name": obj.get("class_name", "?"),
            "mask":       mask,
        })
    return result


# ─── CDM HTTP call ─────────────────────────────────────────────────────────
def call_cdm(bgr, depth_mm, url):
    """Returns optimized depth (uint16 ndarray, mm) or None on failure."""
    orig_h, orig_w = bgr.shape[:2]
    rgb_lb, scale, pt, pl = letterbox(bgr,   798, 448, fill=0)
    dpt_lb, _,     _,  _  = letterbox(depth_mm, 798, 448, fill=0)

    _, buf_rgb = cv2.imencode(".jpg", rgb_lb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    _, buf_dpt = cv2.imencode(".png", dpt_lb)
    files = {
        "rgb": ("rgb.jpg", buf_rgb.tobytes(), "image/jpeg"),
        "dpt": ("dpt.png", buf_dpt.tobytes(), "image/png"),
    }
    data = {"chosen_policy": "dn"}
    try:
        resp = requests.post(f"{url}/api/predict", files=files,
                             data=data, timeout=20,
                             proxies={"http": None, "https": None})
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if "image/png" in ct:
            arr = np.frombuffer(resp.content, np.uint8)
            depth_out = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        else:
            j     = resp.json()
            b64   = j.get("depth") or j.get("result", {}).get("depth", "")
            arr   = np.frombuffer(base64.b64decode(b64), np.uint8)
            depth_out = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"[CDM] request failed: {e}")
        return None
    if depth_out is None:
        return None
    result = unletterbox(depth_out, orig_h, orig_w, scale, pt, pl)
    return result.astype(np.uint16)


# ─── 3D geometry ───────────────────────────────────────────────────────────
def pixel_to_3d(px, py, depth_mm_val, intr):
    """Pixel + scalar depth (mm) -> (X,Y,Z) in camera optical frame (m)."""
    if depth_mm_val <= 0 or depth_mm_val > 60000:
        return None
    d = depth_mm_val / 1000.0
    X = (px - intr["cx"]) * d / intr["fx"]
    Y = (py - intr["cy"]) * d / intr["fy"]
    return (X, Y, d)


def mask_centroid_3d(mask, depth_mm, intr):
    """
    Returns (centroid_xyz_m, centroid_pixel) for a boolean mask.
    Uses eroded mask + IQR filtering + median, matching pipeline logic.
    """
    k      = np.ones((MASK_ERODE_K, MASK_ERODE_K), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), k)
    ys, xs = np.where(eroded > 0)
    if len(ys) == 0:
        return None, None
    ds = depth_mm[ys, xs].astype(np.float32)
    valid = (ds > 50) & (ds < 10000)
    if valid.sum() < 5:
        return None, None
    ds, xs, ys = ds[valid], xs[valid], ys[valid]
    q1, q3 = np.percentile(ds, 25), np.percentile(ds, 75)
    iqr    = q3 - q1
    ok     = (ds >= q1 - 1.5 * iqr) & (ds <= q3 + 1.5 * iqr)
    if ok.sum() < 3:
        return None, None
    ds, xs, ys = ds[ok], xs[ok], ys[ok]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    X = (xs - cx) * ds / fx / 1000.0
    Y = (ys - cy) * ds / fy / 1000.0
    Z = ds / 1000.0
    centroid = np.median(np.column_stack([X, Y, Z]), axis=0)
    cpx = (int(np.median(xs)), int(np.median(ys)))
    return centroid, cpx


# ─── Visualization helpers ─────────────────────────────────────────────────
def depth_colormap(depth_mm, max_mm=DEPTH_VIS_MM):
    d      = np.clip(depth_mm, 0, max_mm).astype(np.float32)
    norm   = (d / max_mm * 255).astype(np.uint8)
    color  = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    # black out invalid pixels
    color[depth_mm <= 0] = 0
    return color


def diff_colormap(raw_mm, cdm_mm, clip_mm=DIFF_CLIP_MM):
    """
    Blue  = CDM deeper than raw  (diff > 0)
    Red   = CDM shallower (diff < 0)
    Gray  = near-zero difference
    Black = invalid (either depth missing)
    """
    valid = (raw_mm > 0) & (cdm_mm > 0) & (raw_mm < 60000) & (cdm_mm < 60000)
    diff  = cdm_mm.astype(np.int32) - raw_mm.astype(np.int32)
    vis   = np.zeros((*raw_mm.shape, 3), dtype=np.uint8)
    neg   = valid & (diff < -5)
    pos   = valid & (diff > 5)
    zero  = valid & (np.abs(diff) <= 5)
    vis[neg, 2] = np.clip(-diff[neg] / clip_mm * 255, 0, 255).astype(np.uint8)
    vis[pos, 0] = np.clip( diff[pos] / clip_mm * 255, 0, 255).astype(np.uint8)
    vis[zero]   = (60, 60, 60)
    return vis


def put_text(img, text, pos, scale=0.45, color=(220, 220, 220), thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


# ─── ROS2 Node ─────────────────────────────────────────────────────────────
class ChassisListener(Node):
    def __init__(self):
        super().__init__("cdm_diag")
        self.bridge = CvBridge()
        self._lock  = threading.Lock()
        self._rgb   = None
        self._depth = None
        self._intr  = None

        color_sub = message_filters.Subscriber(
            self, Image, "/camera/chassis/color/image_raw")
        depth_sub = message_filters.Subscriber(
            self, Image, "/camera/chassis/aligned_depth_to_color/image_raw")
        info_sub  = message_filters.Subscriber(
            self, CameraInfo, "/camera/chassis/color/camera_info")
        sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub], queue_size=5, slop=0.05)
        sync.registerCallback(self._cb)
        self.get_logger().info("Waiting for chassis camera data...")

    def _cb(self, color_msg, depth_msg, info_msg):
        bgr   = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        intr  = {
            "fx": info_msg.k[0], "fy": info_msg.k[4],
            "cx": info_msg.k[2], "cy": info_msg.k[5],
        }
        with self._lock:
            self._rgb   = bgr.copy()
            self._depth = depth.copy()
            self._intr  = intr

    def get_latest(self):
        with self._lock:
            if self._rgb is None:
                return None, None, None
            return self._rgb.copy(), self._depth.copy(), dict(self._intr)


# ─── Main Visualizer ───────────────────────────────────────────────────────
class Visualizer:
    WIN = "CDM Depth Diagnostic  [d=detect  s=CDM-only  q=quit  LClick=3D]"

    def __init__(self, panel_w):
        self.pw          = panel_w
        self.result      = None        # last full detection+CDM result
        self.status_line = "Click on any panel to show 3D coordinates."
        self._panel_h    = panel_w * 9 // 16   # initial estimate

        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN, panel_w * 2, panel_w * 9 // 8 + STATUS_H)
        cv2.setMouseCallback(self.WIN, self._on_mouse)

    def update(self, result):
        self.result = result

    # ── mouse ──────────────────────────────────────────────────────────────
    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.result is None:
            return
        r   = self.result
        pw  = self.pw
        ph  = self._panel_h
        col = 0 if x < pw else 1
        row = 0 if y < ph else 1

        # map click -> original image pixel
        px_panel = x - col * pw
        py_panel = y - row * ph
        img_h, img_w = r["rgb"].shape[:2]
        px = int(np.clip(px_panel * img_w / pw,   0, img_w - 1))
        py = int(np.clip(py_panel * img_h / ph,   0, img_h - 1))

        raw_d = int(r["raw_depth"][py, px])
        cdm_d = int(r["cdm_depth"][py, px])
        intr  = r["intrinsics"]

        pt_raw = pixel_to_3d(px, py, raw_d, intr)
        pt_cdm = pixel_to_3d(px, py, cdm_d, intr)

        def fmt(pt):
            return f"({pt[0]:+.3f},{pt[1]:+.3f},{pt[2]:.3f})m" if pt else "invalid"

        diff_mm = cdm_d - raw_d if raw_d > 0 and cdm_d > 0 else 0
        self.status_line = (
            f"Pixel({px},{py})  "
            f"Raw:{raw_d}mm -> {fmt(pt_raw)}  "
            f"CDM:{cdm_d}mm -> {fmt(pt_cdm)}  "
            f"Diff: {diff_mm:+d}mm"
        )
        print(f"[CLICK] {self.status_line}")

    # ── panel builders ──────────────────────────────────────────────────────
    def _panel_rgb(self, r, pw, ph):
        img_orig = r["rgb"].copy()
        orig_h, orig_w = img_orig.shape[:2]
        sx = pw / orig_w   # scale factors for coordinate mapping
        sy = ph / orig_h

        # Collect annotation info and apply mask overlay (in original space)
        annotations = []
        for i, det in enumerate(r["detections"]):
            c  = COLORS[i % len(COLORS)]
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]

            if det["mask"] is not None:
                overlay = img_orig.copy()
                overlay[det["mask"]] = (
                    np.array(c, dtype=np.float32) * 0.45
                    + overlay[det["mask"]].astype(np.float32) * 0.55
                ).astype(np.uint8)
                img_orig = overlay

                cen_cdm, cpx = mask_centroid_3d(det["mask"], r["cdm_depth"], r["intrinsics"])
                cen_raw, _   = mask_centroid_3d(det["mask"], r["raw_depth"], r["intrinsics"])
            else:
                cen_cdm, cen_raw, cpx = None, None, None

            annotations.append((c, x1, y1, x2, y2, det["class_name"], det["score"],
                                 cen_cdm, cen_raw, cpx))

        # Resize FIRST, then draw all text/markers in panel space
        img = cv2.resize(img_orig, (pw, ph))

        for c, x1, y1, x2, y2, cls, score, cen_cdm, cen_raw, cpx in annotations:
            # bbox in panel coords
            px1 = int(x1 * sx); py1 = int(y1 * sy)
            px2 = int(x2 * sx); py2 = int(y2 * sy)
            cv2.rectangle(img, (px1, py1), (px2, py2), c, 2)
            put_text(img, f"{cls} {score:.2f}", (px1, max(py1 - 6, 14)), 0.50, c, 1)

            if cpx is not None:
                # centroid in panel coords
                cx_p = int(cpx[0] * sx)
                cy_p = int(cpx[1] * sy)
                cv2.drawMarker(img, (cx_p, cy_p), c, cv2.MARKER_CROSS, 16, 2)
                raw_d_at_cen = int(r["raw_depth"][cpx[1], cpx[0]])
                cdm_d_at_cen = int(r["cdm_depth"][cpx[1], cpx[0]])
                tx = min(cx_p + 8, pw - 150)
                if cen_cdm is not None:
                    put_text(img, f"CDM Z:{cen_cdm[2]:.3f}m",
                             (tx, max(cy_p - 6, 14)), 0.50, c)
                if cen_raw is not None:
                    put_text(img, f"Raw Z:{cen_raw[2]:.3f}m",
                             (tx, cy_p + 16), 0.50, (180, 180, 180))
                put_text(img, f"raw:{raw_d_at_cen}mm cdm:{cdm_d_at_cen}mm",
                         (tx, cy_p + 34), 0.42, (200, 200, 100))

        put_text(img, f"RGB + SAM3 Masks ({len(r['detections'])} objects)",
                 (6, 18), 0.48, (255, 255, 255))
        return img

    def _panel_diff(self, r, pw, ph):
        vis = diff_colormap(r["raw_depth"], r["cdm_depth"], DIFF_CLIP_MM)
        vis = cv2.resize(vis, (pw, ph))
        put_text(vis, f"Depth Diff = CDM - Raw  (Blue>0 Red<0  clip={DIFF_CLIP_MM}mm)",
                 (6, 18), 0.42, (255, 255, 255))
        # legend
        items = [("CDM deeper", (200, 0, 0)), ("same", (60, 60, 60)), ("CDM shallower", (0, 0, 200))]
        for i, (lbl, col) in enumerate(items):
            lx = 8 + i * (pw // 3)
            cv2.rectangle(vis, (lx, ph - 22), (lx + 14, ph - 8), col, -1)
            put_text(vis, lbl, (lx + 18, ph - 9), 0.38, (200, 200, 200))
        return vis

    def _panel_depth(self, depth_mm, label, pw, ph):
        vis = depth_colormap(depth_mm, DEPTH_VIS_MM)
        vis = cv2.resize(vis, (pw, ph))
        put_text(vis, f"{label}  (0-{DEPTH_VIS_MM}mm, JET colormap)",
                 (6, 18), 0.46, (255, 255, 255))
        return vis

    def _status_bar(self, total_w):
        bar = np.full((STATUS_H, total_w, 3), 25, dtype=np.uint8)
        # split long line at double-space boundaries
        parts = self.status_line.split("  ")
        y = 22
        line = ""
        for p in parts:
            candidate = line + ("  " if line else "") + p
            if len(candidate) > 100:
                put_text(bar, line, (8, y), 0.44, (190, 220, 190))
                y += 22
                line = p
            else:
                line = candidate
        if line:
            put_text(bar, line, (8, y), 0.44, (190, 220, 190))
        return bar

    def _placeholder(self, pw, ph, text):
        p = np.zeros((ph, pw, 3), dtype=np.uint8)
        put_text(p, text, (10, ph // 2), 0.50, (100, 100, 100))
        return p

    # ── main render ─────────────────────────────────────────────────────────
    def render(self, live_bgr, live_depth=None):
        pw = self.pw
        r  = self.result

        # Always compute panel height from current image
        img_h, img_w = live_bgr.shape[:2]
        ph = pw * img_h // img_w
        self._panel_h = ph

        if r is None:
            # 4-panel with live data where available, placeholders otherwise
            p0 = cv2.resize(live_bgr, (pw, ph))
            put_text(p0, "Live RGB  |  press 'd' to detect, 's' for CDM only",
                     (6, 18), 0.45, (0, 255, 255))

            p1 = self._placeholder(pw, ph, "Depth Diff (CDM-Raw)  -- run CDM first ('s' or 'd')")

            if live_depth is not None:
                p2 = self._panel_depth(live_depth, "Raw Depth (live)", pw, ph)
            else:
                p2 = self._placeholder(pw, ph, "Raw Depth -- waiting for camera...")

            p3 = self._placeholder(pw, ph, "CDM Depth  -- run CDM first ('s' or 'd')")
        else:
            p0 = self._panel_rgb(r, pw, ph)
            p1 = self._panel_diff(r, pw, ph)
            p2 = self._panel_depth(r["raw_depth"], "Raw Depth",  pw, ph)
            p3 = self._panel_depth(r["cdm_depth"], "CDM Depth",  pw, ph)

        row0   = np.hstack([p0, p1])
        row1   = np.hstack([p2, p3])
        status = self._status_bar(pw * 2)
        return np.vstack([row0, row1, status])

    def show(self, live_bgr, live_depth=None):
        cv2.imshow(self.WIN, self.render(live_bgr, live_depth))

    def destroy(self):
        cv2.destroyAllWindows()


# ─── Background task runner ────────────────────────────────────────────────
class TaskRunner:
    def __init__(self, node, vis, args):
        self.node    = node
        self.vis     = vis
        self.args    = args
        self._thread = None

    def _busy(self):
        return self._thread is not None and self._thread.is_alive()

    def run(self, mode):
        if self._busy():
            print("[WARN] Previous task still running, please wait.")
            return

        def _task():
            rgb, raw_d, intr = self.node.get_latest()
            if rgb is None:
                print("[WARN] No camera data yet.")
                return

            detections = []
            if mode == "detect":
                print(f"[SAM3] Calling detector, prompt='{self.args.prompt}'...")
                t0 = time.time()
                detections = call_sam3(rgb, self.args.sam3_url,
                                       self.args.prompt, self.args.conf)
                print(f"[SAM3] {len(detections)} objects  ({time.time()-t0:.2f}s)")

            print("[CDM]  Calling depth optimizer...")
            t0    = time.time()
            cdm_d = call_cdm(rgb, raw_d, self.args.cdm_url)
            if cdm_d is None:
                print("[CDM]  Failed, using raw depth as fallback.")
                cdm_d = raw_d.copy()
            else:
                print(f"[CDM]  Done  ({time.time()-t0:.2f}s)")

            self.vis.update({
                "rgb":        rgb,
                "raw_depth":  raw_d,
                "cdm_depth":  cdm_d,
                "detections": detections,
                "intrinsics": intr,
            })

        self._thread = threading.Thread(target=_task, daemon=True)
        self._thread.start()


# ─── Entry point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CDM Depth Diagnostic Tool")
    parser.add_argument("--sam3-url", default=SAM3_URL)
    parser.add_argument("--cdm-url",  default=CDM_URL)
    parser.add_argument("--prompt",   default=PROMPT)
    parser.add_argument("--conf",     type=float, default=CONF)
    parser.add_argument("--panel-w",  type=int,   default=PANEL_W)
    args = parser.parse_args()

    rclpy.init()
    node   = ChassisListener()
    vis    = Visualizer(args.panel_w)
    runner = TaskRunner(node, vis, args)

    spin_t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_t.start()

    print("=== CDM Depth Diagnostic ===")
    print(f"  SAM3 : {args.sam3_url}")
    print(f"  CDM  : {args.cdm_url}")
    print(f"  Prompt: '{args.prompt}'  conf={args.conf}")
    print("  Keys: d=detect  s=CDM-only  q/ESC=quit  LClick=3D coords")

    try:
        while rclpy.ok():
            rgb, depth, _ = node.get_latest()
            if rgb is not None:
                vis.show(rgb, depth)
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("d"):
                runner.run("detect")
            elif key == ord("s"):
                runner.run("cdm")
    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
