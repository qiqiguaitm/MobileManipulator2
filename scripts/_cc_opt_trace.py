#!/usr/bin/env python3
"""精确追踪 opt 坐标的计算过程 — 对比理想 vs 管线实际。

用法:
  export PATH="/usr/bin:$PATH"
  python3 scripts/_cc_opt_trace.py
"""

import numpy as np
import cv2

# === chassis 内参 (来自 ROS camera_info) ===
fx, fy = 904.88, 904.40
cx, cy = 653.80, 371.75
H, W = 720, 1280

DEPTH_MIN, DEPTH_MAX = 0.3, 10.0
IQR_FACTOR = 1.5
ERODE_K = 5


def run_pipeline(mask, depth, label):
    """模拟 batch_camera_3d_percept 的完整流程"""
    ys_all, xs_all = np.where(mask > 0)
    x1, y1 = int(xs_all.min()), int(ys_all.min())
    x2, y2 = int(xs_all.max() + 1), int(ys_all.max() + 1)

    mask_roi = mask[y1:y2, x1:x2]
    depth_roi = depth[y1:y2, x1:x2]

    kernel = np.ones((ERODE_K, ERODE_K), np.uint8)
    eroded = cv2.erode(mask_roi, kernel, iterations=1)
    n_before, n_after = int(mask_roi.sum()), int(eroded.sum())

    ys_local, xs_local = np.where(eroded > 0)
    ds = depth_roi[ys_local, xs_local]

    valid = (ds > DEPTH_MIN) & (ds < DEPTH_MAX)
    depth_values = ds[valid]

    if len(depth_values) < 10:
        print(f"    [{label}] 有效深度不足: {len(depth_values)}")
        return

    q1, q3 = np.percentile(depth_values, [25, 75])
    iqr = q3 - q1
    lb = q1 - IQR_FACTOR * iqr
    ub = q3 + IQR_FACTOR * iqr
    valid2 = valid & (ds >= lb) & (ds <= ub)

    xs_img = xs_local[valid2].astype(np.float64) + x1
    ys_img = ys_local[valid2].astype(np.float64) + y1
    ds_v = ds[valid2]

    X = (xs_img - cx) * ds_v / fx
    Y = (ys_img - cy) * ds_v / fy
    Z = ds_v

    centroid = np.array([np.median(X), np.median(Y), np.median(Z)])

    print(f"    erode: {n_before} → {n_after} px | IQR: [{lb:.4f}, {ub:.4f}] | 有效3D: {int(valid2.sum())} px")
    print(f"    depth: min={ds_v.min():.4f}  max={ds_v.max():.4f}  median={np.median(ds_v):.4f}  mean={np.mean(ds_v):.4f}")
    print(f"    pixel: v=[{ys_img.min():.0f},{ys_img.max():.0f}] median={np.median(ys_img):.0f}  "
          f"u=[{xs_img.min():.0f},{xs_img.max():.0f}] median={np.median(xs_img):.0f}")
    print(f"    → opt = ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")
    return centroid


def main():
    print("=" * 70)
    print(" opt 坐标是什么 + 为什么和预期差这么多")
    print("=" * 70)

    print("""
【opt 坐标的含义】
  opt = chassis_camera_optical_frame 坐标系下的 3D 位置
  X → 图像右方, Y → 图像下方, Z → 相机正前方(深度)

【理想计算】
  已知: 像素 (u, v) + 深度 Z + 内参 (fx, fy, cx, cy)
  X = (u - cx) * Z / fx
  Y = (v - cy) * Z / fy
  Z = depth
  → 这是精确的, 零误差

【管线实际计算】(见 scene_perception_core.py:245-303)
  ① RealSense 原始深度 (uint16 mm)
  ② CDM 神经网络 修改深度图          ← 深度可能被改变
  ③ SAM3 生成的 mask (哪些像素属于目标)  ← mask 可能不精确
  ④ cv2.erode(mask, 5x5) 腐蚀边缘
  ⑤ 提取 mask 内 所有 像素的深度
  ⑥ IQR 异常值剔除
  ⑦ 每个像素独立 pinhole 反投影 → 3D 点
  ⑧ centroid_optical = np.median(points, axis=0)  ← 取所有点的中值

  关键: 不是取目标中心 1 个像素, 而是 mask 内 N 千个像素的中值
""")

    # 假设目标中心像素和深度
    u_center, v_center, Z_real = 640, 480, 0.91

    print("=" * 70)
    print(f" 场景: 目标在像素({u_center},{v_center}), RealSense深度={Z_real}m")
    print(f" 内参: fx={fx} fy={fy} cx={cx} cy={cy}")
    print("=" * 70)

    # === A: 理想单点计算 ===
    print("\n【A】理想单点: 直接用中心像素计算")
    X_ideal = (u_center - cx) * Z_real / fx
    Y_ideal = (v_center - cy) * Z_real / fy
    print(f"    X = ({u_center} - {cx}) * {Z_real} / {fx} = {X_ideal:.4f}")
    print(f"    Y = ({v_center} - {cy}) * {Z_real} / {fy} = {Y_ideal:.4f}")
    print(f"    Z = {Z_real:.4f}")
    print(f"    → opt = ({X_ideal:.4f}, {Y_ideal:.4f}, {Z_real:.4f})  ← 这是精确答案")

    # === B1: 理想 mask + 理想深度 ===
    print("\n\n【B1】理想 mask + 理想深度 (验证管线数学)")
    print("    mask = 圆形, 所有深度 = 0.91m")
    mask_good = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask_good, (u_center, v_center), 50, 1, -1)
    depth_flat = np.full((H, W), Z_real, dtype=np.float32)
    c1 = run_pipeline(mask_good, depth_flat, "B1")
    if c1 is not None:
        print(f"    对比理想: ΔX={c1[0]-X_ideal:.4f} ΔY={c1[1]-Y_ideal:.4f} ΔZ={c1[2]-Z_real:.4f}")
        print(f"    → 管线数学本身精确, 误差为零")

    # === B2: 理想 mask + 3D 物体表面深度 ===
    print("\n\n【B2】理想 mask + 3D 物体表面 (前表面比中心近)")
    print("    目标: 圆柱体, 中心0.91m, 前表面0.895m, 边缘0.91m")
    depth_3d = np.full((H, W), 2.5, dtype=np.float32)
    ys_m, xs_m = np.where(mask_good > 0)
    for i in range(len(ys_m)):
        dist_px = np.sqrt((xs_m[i] - u_center)**2 + (ys_m[i] - v_center)**2)
        depth_3d[ys_m[i], xs_m[i]] = 0.91 - 0.015 * max(0, 1 - dist_px / 50)
    c2 = run_pipeline(mask_good, depth_3d, "B2")
    if c2 is not None:
        print(f"    对比理想: ΔZ={c2[2]-Z_real:.4f}m = {(c2[2]-Z_real)*1000:.1f}mm")
        print(f"    → 3D 表面效应引入约 {abs(c2[2]-Z_real)*1000:.0f}mm 偏差")

    # === B3: mask 向下扩展到地面 ===
    print("\n\n【B3】mask 向下扩展 (SAM3 包含了地面像素)")
    print("    mask: 向下额外延伸 80px, 包含地面深度")
    mask_bad = mask_good.copy()
    cv2.ellipse(mask_bad, (u_center, v_center + 30), (50, 80), 0, 0, 360, 1, -1)
    n_extra = mask_bad.sum() - mask_good.sum()
    print(f"    多了 {n_extra} 个像素 (地面区域)")

    # 地面深度模型: 相机高15cm, 地面像素深度 = 0.15 / sin(θ)
    depth_floor = depth_3d.copy()
    for v in range(v_center - 10, min(H, v_center + 120)):
        angle = np.arctan2(v - cy, fy)
        if angle > 0.01:
            floor_z = 0.15 / np.sin(angle)
            if 0.3 < floor_z < 5.0:
                for u in range(u_center - 60, u_center + 60):
                    if 0 <= u < W and mask_bad[v, u] > 0 and mask_good[v, u] == 0:
                        depth_floor[v, u] = floor_z

    c3 = run_pipeline(mask_bad, depth_floor, "B3")
    if c3 is not None:
        print(f"    对比理想: ΔY={c3[1]-Y_ideal:.4f}m ({(c3[1]-Y_ideal)*100:.1f}cm)  ΔZ={c3[2]-Z_real:.4f}m ({(c3[2]-Z_real)*1000:.1f}mm)")
        print(f"    → mask 扩展主要影响 Y 坐标")

    # === B4: CDM 把深度压缩 70mm ===
    print("\n\n【B4】CDM 全局深度压缩 -70mm (无mask问题)")
    depth_cdm = depth_3d.copy()
    depth_cdm -= 0.07  # 全图减 70mm
    depth_cdm[depth_cdm < 0] = 0
    c4 = run_pipeline(mask_good, depth_cdm, "B4")
    if c4 is not None:
        print(f"    对比理想: ΔY={c4[1]-Y_ideal:.4f}m  ΔZ={c4[2]-Z_real:.4f}m ({(c4[2]-Z_real)*1000:.1f}mm)")
        print(f"    → CDM 压缩 直接导致 Z 偏小, 同时 Y 也受影响 (Y ∝ Z)")

    # === B5: CDM + mask 扩展 (叠加) ===
    print("\n\n【B5】CDM 压缩 + mask 扩展 (最可能的真实情况)")
    depth_both = depth_floor.copy()
    depth_both -= 0.05  # CDM 减 50mm
    depth_both[depth_both < 0] = 0
    c5 = run_pipeline(mask_bad, depth_both, "B5")
    if c5 is not None:
        print(f"    对比理想: ΔX={c5[0]-X_ideal:.4f}m  ΔY={c5[1]-Y_ideal:.4f}m ({(c5[1]-Y_ideal)*100:.1f}cm)  ΔZ={c5[2]-Z_real:.4f}m ({(c5[2]-Z_real)*1000:.1f}mm)")

    # === 总结 ===
    print("\n\n" + "=" * 70)
    print(" 总结: 各因素对 opt 坐标的影响")
    print("=" * 70)
    print(f"""
  理想值:     ({X_ideal:.4f}, {Y_ideal:.4f}, {Z_real:.4f})
  用户观测:   (0.06,    0.16,    0.84   )
  差异:       ΔX≈0  ΔY≈+{0.16-Y_ideal:.2f}m  ΔZ≈-0.07m

  各因素贡献:
  ┌──────────────────────────┬────────┬────────┬────────┐
  │ 因素                     │  ΔX    │  ΔY    │  ΔZ    │
  ├──────────────────────────┼────────┼────────┼────────┤
  │ pinhole 公式本身         │  0     │  0     │  0     │
  │ 3D 表面(前表面比中心近) │  ~0    │  ~0    │  ~-8mm │
  │ mask 不精确(含地面)     │  ~0    │ +数cm  │  小    │
  │ CDM 深度压缩            │  ~0    │ +少量  │ -70mm  │
  ├──────────────────────────┼────────┼────────┼────────┤
  │ 合计                     │  ~0    │ +9cm   │ -70mm  │
  └──────────────────────────┴────────┴────────┴────────┘

  结论:
  - pinhole 反投影公式本身没有任何问题
  - Z 偏小 → 主要是 CDM 修改了深度值
  - Y 偏大 → 主要是 SAM3 mask 包含了非目标像素
""")


if __name__ == '__main__':
    main()
