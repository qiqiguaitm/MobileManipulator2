#!/usr/bin/env python3
"""
RGB + 对齐深度图同步采集 (top + chassis)
- 按 's' 保存当前帧
- 按 'q' 退出
保存目录: captures/rgbd/<timestamp>/
"""

import os
import sys
import threading
from datetime import datetime

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
import message_filters

CAMERAS = ["top", "chassis"]
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "captures", "rgbd")


class RGBDCapture(Node):
    def __init__(self):
        super().__init__("rgbd_capture")
        self.bridge = CvBridge()
        self.latest: dict[str, tuple] = {}  # cam -> (bgr, depth_uint16)
        self.lock = threading.Lock()
        self.frame_count = 0

        for cam in CAMERAS:
            color_sub = message_filters.Subscriber(
                self, Image, f"/camera/{cam}/color/image_raw"
            )
            depth_sub = message_filters.Subscriber(
                self, Image, f"/camera/{cam}/aligned_depth_to_color/image_raw"
            )
            sync = message_filters.ApproximateTimeSynchronizer(
                [color_sub, depth_sub], queue_size=10, slop=0.05
            )
            sync.registerCallback(self._make_cb(cam))
            self.get_logger().info(f"[{cam}] subscribed")

    def _make_cb(self, cam: str):
        def cb(color_msg: Image, depth_msg: Image):
            bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            with self.lock:
                self.latest[cam] = (bgr.copy(), depth.copy())
        return cb

    def get_snapshot(self):
        with self.lock:
            return {k: (b.copy(), d.copy()) for k, (b, d) in self.latest.items()}

    def save(self):
        snapshot = self.get_snapshot()
        missing = [c for c in CAMERAS if c not in snapshot]
        if missing:
            print(f"[WARN] 尚未收到数据: {missing}")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        save_dir = os.path.join(SAVE_DIR, ts)
        os.makedirs(save_dir, exist_ok=True)

        for cam, (bgr, depth) in snapshot.items():
            cv2.imwrite(os.path.join(save_dir, f"{cam}_color.png"), bgr)
            cv2.imwrite(os.path.join(save_dir, f"{cam}_depth.png"), depth)  # 16-bit PNG

        self.frame_count += 1
        print(f"[SAVE #{self.frame_count}] {save_dir}")

    def build_preview(self, width=640) -> np.ndarray:
        """拼接两个相机的 RGB 和深度伪彩色预览"""
        snapshot = self.get_snapshot()
        panels = []
        for cam in CAMERAS:
            if cam not in snapshot:
                h = width * 9 // 16
                blank = np.zeros((h, width, 3), dtype=np.uint8)
                cv2.putText(blank, f"{cam}: waiting...", (20, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                panels.append(blank)
                continue

            bgr, depth = snapshot[cam]
            # resize color
            h_orig, w_orig = bgr.shape[:2]
            h_new = width * h_orig // w_orig
            color_resized = cv2.resize(bgr, (width, h_new))

            # depth -> 伪彩色 (clip to 5m)
            depth_vis = np.clip(depth, 0, 5000).astype(np.float32)
            depth_norm = (depth_vis / 5000 * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            depth_color = cv2.resize(depth_color, (width, h_new))

            label_color = np.zeros((30, width, 3), dtype=np.uint8)
            cv2.putText(label_color, f"{cam} RGB", (5, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            label_depth = np.zeros((30, width, 3), dtype=np.uint8)
            cv2.putText(label_depth, f"{cam} Depth (0-5m)", (5, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            panel = np.vstack([label_color, color_resized, label_depth, depth_color])
            panels.append(panel)

        # 两列并排
        max_h = max(p.shape[0] for p in panels)
        padded = []
        for p in panels:
            dh = max_h - p.shape[0]
            padded.append(np.vstack([p, np.zeros((dh, p.shape[1], 3), dtype=np.uint8)]))
        return np.hstack(padded)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    rclpy.init()
    node = RGBDCapture()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("=== RGB+Depth 采集工具 ===")
    print("  's' - 保存当前帧")
    print("  'q' - 退出")
    print(f"  保存目录: {os.path.abspath(SAVE_DIR)}")

    cv2.namedWindow("RGBD Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGBD Preview", 1280, 900)

    try:
        while rclpy.ok():
            preview = node.build_preview(width=600)
            cv2.imshow("RGBD Preview", preview)
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                node.save()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
