#!/usr/bin/env python3
"""单独测试 perception 模块的 SAM3Online 服务 + 相机出图
不依赖 camera.py，直接用 pyrealsense2 抓帧，避免 matplotlib/numpy 冲突。
"""

import sys
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import requests

# 添加 perception 模块路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'src', 'perception', 'src'))

from percept import SAM3Online


# 从 perception_grasp.yaml 读取的默认 SAM3 地址
SAM3_URL = 'http://192.168.112.14:8080'
CAMERA_SERIAL = '337122071190'
IMG_W, IMG_H, FPS = 1280, 720, 6


class SimpleConfig:
    """轻量配置，模拟 perception 节点中的用法"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


def test_service_health():
    """检查 SAM3 服务是否可达"""
    print(f"[1] 检查 SAM3 服务: {SAM3_URL}")
    try:
        r = requests.get(f'{SAM3_URL}/api/health', timeout=3)
        print(f"  health: {r.status_code} {r.text[:200]}")
        return True
    except Exception as e:
        print(f"  ✗ 连接失败: {e}")
        return False


def grab_frame():
    """直接用 pyrealsense2 抓一帧 RGB + Depth"""
    print(f"\n[2] 连接相机 (serial={CAMERA_SERIAL})...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(CAMERA_SERIAL)
    config.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.rgb8, FPS)
    config.enable_stream(rs.stream.depth, IMG_W, IMG_H, rs.format.z16, FPS)
    profile = pipeline.start(config)

    # 对齐深度到彩色
    align = rs.align(rs.stream.color)

    # 跳几帧让自动曝光稳定
    for _ in range(10):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    frames = align.process(frames)

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    rgb = np.asanyarray(color_frame.get_data())          # RGB, uint8
    depth = np.asanyarray(depth_frame.get_data()) * 0.001  # mm -> m, float

    pipeline.stop()

    print(f"  RGB: shape={rgb.shape}, dtype={rgb.dtype}, min={rgb.min()}, max={rgb.max()}")
    print(f"  Depth: shape={depth.shape}, dtype={depth.dtype}, min={depth.min():.3f}, max={depth.max():.3f}")

    save_path = '/tmp/_cc_cam_test.jpg'
    cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"  已保存 RGB 到 {save_path}")

    return rgb, depth


def test_sam3(rgb):
    """测试 SAM3 检测 — 使用 perception 模块的 SAM3Online"""
    print("\n[3] 测试 perception SAM3Online 检测...")

    # 和 perception_grasp_node.py 一样初始化
    cfg = SimpleConfig(
        url=SAM3_URL,
        min_score=0.25,
        return_mask=True,
    )
    ref = SAM3Online(cfg)

    # 先用 demo.py 同款 prompt
    text = 'toy cubic.remote.bottle'
    print(f"  text_prompt: '{text}'")
    print(f"  image: {rgb.shape}, dtype={rgb.dtype}")

    t0 = time.time()
    task = ref.forward(rgb=rgb, text=text)
    dt = time.time() - t0
    result = task.result
    objects = result.get('objects', [])
    error = result.get('error', None)
    print(f"  耗时: {dt:.2f}s")
    if error:
        print(f"  ✗ 服务返回错误: {error}")
    print(f"  objects 数量: {len(objects)}")
    for i, obj in enumerate(objects):
        print(f"    [{i}] category={obj.get('category')}, score={obj.get('score'):.3f}, bbox={obj.get('bbox')}")

    if not objects:
        print("\n  ⚠ SAM3 返回空结果，逐个 prompt 尝试...")
        for alt_text in ['bottle', 'object', 'pen', 'remote', 'cup', 'toy']:
            task2 = ref.forward(rgb=rgb, text=alt_text)
            objs2 = task2.result.get('objects', [])
            err2 = task2.result.get('error', None)
            status = f"error={err2}" if err2 else f"{len(objs2)} objects"
            print(f"    prompt='{alt_text}' → {status}")
            for o in objs2:
                print(f"      category={o.get('category')}, score={o.get('score'):.3f}, bbox={o.get('bbox')}")

    # 对比 RGB vs BGR 输入
    print("\n  对比测试: BGR 格式输入...")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    task3 = ref.forward(rgb=bgr, text=text)
    objs3 = task3.result.get('objects', [])
    print(f"    BGR输入 → {len(objs3)} objects")
    for o in objs3:
        print(f"      category={o.get('category')}, score={o.get('score'):.3f}, bbox={o.get('bbox')}")

    return ref


if __name__ == '__main__':
    service_ok = test_service_health()
    rgb, depth = grab_frame()
    if service_ok:
        test_sam3(rgb)
    else:
        print("\n✗ SAM3 服务不可达，跳过检测测试")
    print("\n完成。")
