#!/usr/bin/env python3
"""模拟 CDM letterbox/unletterbox 往返，量化 resize 引入的深度误差。

不需要 CDM 服务在线 — 只测试 letterbox → unletterbox 的几何变换。
用法:
  export PATH="/usr/bin:$PATH"
  python3 scripts/_cc_cdm_roundtrip.py
"""

import numpy as np
import cv2
import os

def letterbox(img, target_long=798, target_short=448, fill_value=0):
    h, w = img.shape[:2]
    if w >= h:
        target_w, target_h = target_long, target_short
    else:
        target_w, target_h = target_short, target_long

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    if len(img.shape) == 3:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    if len(img.shape) == 3:
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=(fill_value,) * 3)
    else:
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=fill_value)

    return padded, scale, pad_top, pad_left, target_h, target_w


def unletterbox(img, orig_h, orig_w, scale, pad_top, pad_left):
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    cropped = img[pad_top:pad_top + new_h, pad_left:pad_left + new_w]

    if len(img.shape) == 3:
        restored = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    else:
        restored = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return restored


def main():
    depth_path = '/tmp/diag_depth_mm.npy'
    if not os.path.exists(depth_path):
        print(f'错误: 找不到 {depth_path}')
        print('请先运行: python3 scripts/_cc_depth_diag.py --save')
        return

    depth_mm = np.load(depth_path)
    print(f'=== CDM letterbox/unletterbox 往返误差分析 ===')
    print(f'原始深度: {depth_mm.shape} dtype={depth_mm.dtype}')
    print(f'目标: 798x448 (CDM default)')
    print()

    orig_h, orig_w = depth_mm.shape[:2]

    # letterbox
    padded, scale, pad_top, pad_left, target_h, target_w = letterbox(depth_mm)
    print(f'缩放: scale={scale:.6f}')
    print(f'缩放后: {int(orig_w * scale)}x{int(orig_h * scale)}')
    print(f'Padding: top={pad_top} left={pad_left}')
    print(f'Padded: {padded.shape}')

    # 模拟: 如果 CDM 不改变任何像素值，只做 letterbox→unletterbox
    restored = unletterbox(padded, orig_h, orig_w, scale, pad_top, pad_left)
    print(f'还原: {restored.shape}')
    print()

    # 比较
    valid = depth_mm > 100  # > 10cm
    diff = restored.astype(np.int32) - depth_mm.astype(np.int32)
    diff_valid = diff[valid]

    print('=== 往返误差 (不含 CDM 网络) ===')
    print(f'有效像素: {valid.sum()}')
    print(f'完全一致: {(diff_valid == 0).sum()} ({(diff_valid == 0).sum() / len(diff_valid) * 100:.1f}%)')
    print(f'误差范围: [{diff_valid.min()}, {diff_valid.max()}] mm')
    print(f'|误差|>0 的像素: {(np.abs(diff_valid) > 0).sum()} ({(np.abs(diff_valid) > 0).sum() / len(diff_valid) * 100:.1f}%)')
    print(f'|误差|>5mm 的像素: {(np.abs(diff_valid) > 5).sum()}')
    print(f'|误差|>10mm 的像素: {(np.abs(diff_valid) > 10).sum()}')
    print(f'均值误差: {diff_valid.mean():.3f} mm')
    print(f'标准差: {diff_valid.std():.3f} mm')
    print()

    # 在不同深度区域分析误差
    print('=== 分深度段误差 ===')
    for d_min, d_max in [(500, 1000), (800, 1000), (900, 920), (1000, 2000), (2000, 3000)]:
        mask = (depth_mm >= d_min) & (depth_mm < d_max)
        if mask.sum() > 0:
            d = diff[mask]
            print(f'  {d_min}-{d_max}mm: n={mask.sum():6d}  '
                  f'mean_err={d.mean():.3f}mm  std={d.std():.3f}mm  '
                  f'max_abs={np.abs(d).max()}mm')
    print()

    # 模拟更真实的场景: CDM 可能引入的 PNG 编解码
    print('=== 加上 PNG 编解码 (CDM 传输格式) ===')
    _, encoded = cv2.imencode('.png', padded)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    restored2 = unletterbox(decoded, orig_h, orig_w, scale, pad_top, pad_left)

    diff2 = restored2.astype(np.int32) - depth_mm.astype(np.int32)
    diff2_valid = diff2[valid]
    print(f'PNG 往返 完全一致: {(diff2_valid == 0).sum()} ({(diff2_valid == 0).sum() / len(diff2_valid) * 100:.1f}%)')
    print(f'PNG 往返 均值误差: {diff2_valid.mean():.3f} mm')
    print()

    # 重点分析: 在图像下部 (目标可能出现的位置) 的误差
    print('=== 图像下部 (v=500-700, 目标常出现区域) ===')
    roi = depth_mm[500:700, :]
    roi_restored = restored[500:700, :]
    roi_valid = roi > 100
    roi_diff = (roi_restored.astype(np.int32) - roi.astype(np.int32))[roi_valid]
    if len(roi_diff) > 0:
        print(f'  有效像素: {roi_valid.sum()}')
        print(f'  完全一致: {(roi_diff == 0).sum()} ({(roi_diff == 0).sum() / len(roi_diff) * 100:.1f}%)')
        print(f'  均值误差: {roi_diff.mean():.3f} mm')
        print(f'  最大|误差|: {np.abs(roi_diff).max()} mm')
    print()

    # 关键测试: 在特定深度值(910mm ≈ 91cm)的像素上测试
    print('=== 深度值 ~910mm 的像素往返精度 ===')
    target_mask = (depth_mm >= 900) & (depth_mm <= 920)
    if target_mask.sum() > 0:
        orig_vals = depth_mm[target_mask]
        rest_vals = restored[target_mask]
        d = rest_vals.astype(np.int32) - orig_vals.astype(np.int32)
        print(f'  像素数: {target_mask.sum()}')
        print(f'  原始: median={np.median(orig_vals):.0f} mean={np.mean(orig_vals):.1f}mm')
        print(f'  还原: median={np.median(rest_vals):.0f} mean={np.mean(rest_vals):.1f}mm')
        print(f'  误差: mean={d.mean():.3f}mm std={d.std():.3f}mm max_abs={np.abs(d).max()}mm')
    else:
        print(f'  (当前帧没有 900-920mm 范围的像素)')

    print()
    print('完成。')


if __name__ == '__main__':
    main()
