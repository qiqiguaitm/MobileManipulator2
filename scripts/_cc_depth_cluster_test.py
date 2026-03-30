#!/usr/bin/env python3
"""测试 chassis camera 深度聚类管线

订阅 chassis 深度图 + 内参，跑完整 pipeline:
  depth → pointcloud → ROI → outlier → RANSAC → DBSCAN
打印每阶段点数 + 每个聚类的 (cx, cy, cz) 坐标

用法:
  export PATH="/usr/bin:$PATH"
  python3 scripts/_cc_depth_cluster_test.py
"""
import time
import sys
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# ---------- 参数 (与 approach_navigator.yaml 一致) ----------
DEPTH_TOPIC = "/camera/chassis/aligned_depth_to_color/image_raw"
INFO_TOPIC  = "/camera/chassis/aligned_depth_to_color/camera_info"

DOWNSAMPLE     = 2       # 降采样2x
MIN_VALID      = 0.10
MAX_VALID      = 2.0
DETECT_WIDTH   = 0.8     # |x| < 此值
OBS_MIN_H      = -0.50   # y 下限 (上方)
OBS_MAX_H      = 0.24    # y 上限 (下方, 含地面)
GROUND_MAX_H   = 0.158   # 地面 y
GROUND_MARGIN  = 0.05
RANSAC_DIST    = 0.02
RANSAC_ITER    = 80
RANSAC_MIN_R   = 0.3
GND_INLIER_MIN = -0.01
GND_INLIER_MAX = 0.03
OUTLIER_R      = 0.05
OUTLIER_MIN_NB = 5
CLUSTER_EPS    = 0.05
CLUSTER_MIN    = 50
CLUSTER_MAX    = 10000


class DepthClusterTest(Node):
    def __init__(self):
        super().__init__('depth_cluster_test')
        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = 0.0
        self.has_intrinsics = False
        self.depth_image = None
        self.depth_ts = 0.0
        self.frame_count = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, DEPTH_TOPIC, self._depth_cb, qos)
        self.create_subscription(CameraInfo, INFO_TOPIC, self._info_cb, qos)
        self.get_logger().info(f"订阅: {DEPTH_TOPIC}")
        self.get_logger().info(f"downsample={DOWNSAMPLE}, detect_width={DETECT_WIDTH}")

    def _depth_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32)
        if msg.encoding == '16UC1':
            img /= 1000.0
        self.depth_image = img
        self.depth_ts = time.time()

    def _info_cb(self, msg):
        if not self.has_intrinsics:
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]
            self.has_intrinsics = True
            self.get_logger().info(
                f"内参: fx={self.fx:.1f} fy={self.fy:.1f} "
                f"cx={self.cx:.1f} cy={self.cy:.1f}")

    def process_once(self):
        if self.depth_image is None or not self.has_intrinsics:
            return
        if time.time() - self.depth_ts > 0.5:
            return

        t0 = time.time()
        depth = self.depth_image.copy()
        h, w = depth.shape[:2]

        # 1. depth → pointcloud
        ds = DOWNSAMPLE
        v_arr = np.arange(0, h, ds)
        u_arr = np.arange(0, w, ds)
        u_grid, v_grid = np.meshgrid(u_arr, v_arr)
        u_flat = u_grid.ravel()
        v_flat = v_grid.ravel()
        d_flat = depth[::ds, ::ds].ravel()

        valid = (d_flat > MIN_VALID) & (d_flat < MAX_VALID)
        d = d_flat[valid]
        u = u_flat[valid]
        v = v_flat[valid]
        x = (u - self.cx) * d / self.fx
        y = (v - self.cy) * d / self.fy
        z = d
        points = np.column_stack([x, y, z])
        n_total = len(points)
        t1 = time.time()

        # 2. ROI
        xc, yc, zc = points[:, 0], points[:, 1], points[:, 2]
        roi = (
            (zc > MIN_VALID) & (zc < MAX_VALID) &
            (np.abs(xc) < DETECT_WIDTH) &
            (yc > OBS_MIN_H) & (yc < OBS_MAX_H)
        )
        front = points[roi]
        n_front = len(front)
        t2 = time.time()

        # 3. outlier removal
        from scipy.spatial import cKDTree
        if len(front) > OUTLIER_MIN_NB:
            counts = cKDTree(front).query_ball_point(front, OUTLIER_R, return_length=True)
            front = front[counts > OUTLIER_MIN_NB]
        n_denoised = len(front)
        t3 = time.time()

        # 4. RANSAC ground removal
        obstacle, ground = self._remove_ground(front)
        n_obs = len(obstacle)
        n_gnd = len(ground)
        t4 = time.time()

        # 5. DBSCAN clustering
        clusters = self._cluster(obstacle)
        t5 = time.time()

        total_ms = (t5 - t0) * 1000
        pc_ms = (t1 - t0) * 1000
        roi_ms = (t2 - t1) * 1000
        out_ms = (t3 - t2) * 1000
        gnd_ms = (t4 - t3) * 1000
        cls_ms = (t5 - t4) * 1000

        self.frame_count += 1
        print(f"\n===== Frame {self.frame_count}  total={total_ms:.0f}ms =====")
        print(f"  pointcloud: {n_total} pts  ({pc_ms:.0f}ms)")
        print(f"  ROI:        {n_front} pts  ({roi_ms:.0f}ms)")
        print(f"  denoise:    {n_denoised} pts  ({out_ms:.0f}ms)")
        print(f"  ground:     obs={n_obs} gnd={n_gnd}  ({gnd_ms:.0f}ms)")
        print(f"  clusters:   {len(clusters)}  ({cls_ms:.0f}ms)")

        # 打印每个聚类
        for i, cls_pts in enumerate(clusters):
            centroid = np.mean(cls_pts, axis=0)
            front_z = float(np.min(cls_pts[:, 2]))
            print(f"    [{i}] n={len(cls_pts):4d}  "
                  f"cx={centroid[0]:+.3f} cy={centroid[1]:+.3f} cz={centroid[2]:.3f}  "
                  f"front_z={front_z:.3f}m")

        # 目标区域诊断: 在 z=0.4~0.8 范围内有多少点
        target_mask = (obstacle[:, 2] > 0.3) & (obstacle[:, 2] < 0.9)
        n_target_zone = int(np.sum(target_mask))
        if n_target_zone > 0:
            tz_pts = obstacle[target_mask]
            print(f"  [目标区 z=0.3~0.9m] {n_target_zone} 点  "
                  f"x=[{tz_pts[:,0].min():.3f},{tz_pts[:,0].max():.3f}]  "
                  f"y=[{tz_pts[:,1].min():.3f},{tz_pts[:,1].max():.3f}]  "
                  f"z=[{tz_pts[:,2].min():.3f},{tz_pts[:,2].max():.3f}]")
        else:
            print(f"  [目标区 z=0.3~0.9m] 0 点!")

    def _remove_ground(self, pts):
        empty = np.empty((0, 3), dtype=np.float32)
        if len(pts) < 50:
            return pts, empty

        ground_max = abs(GROUND_MAX_H)
        yc = pts[:, 1]
        gnd_mask = yc >= (ground_max - GROUND_MARGIN)
        pot_gnd = pts[gnd_mask]

        if len(pot_gnd) < 30:
            return pts, empty

        # RANSAC
        if len(pot_gnd) > 2000:
            idx = np.random.choice(len(pot_gnd), 2000, replace=False)
            ransac_in = pot_gnd[idx]
        else:
            ransac_in = pot_gnd

        normal, d_plane = self._ransac(ransac_in)

        if normal is not None:
            signed = pot_gnd @ normal - d_plane
            inliers = (signed >= GND_INLIER_MIN) & (signed < GND_INLIER_MAX)
        else:
            inliers = None

        if inliers is None:
            gm = yc >= ground_max
            return pts[~gm], pts[gm]

        ground_pts = pot_gnd[inliers]
        near_non = pot_gnd[~inliers]
        far_pts = pts[~gnd_mask]

        if len(near_non) > 0 and len(far_pts) > 0:
            obs = np.vstack([far_pts, near_non])
        elif len(far_pts) > 0:
            obs = far_pts
        else:
            obs = near_non

        return obs, ground_pts

    def _ransac(self, pts):
        n = len(pts)
        if n < 10:
            return None, None
        best_n, best_d, best_count = None, 0.0, 0
        for _ in range(RANSAC_ITER):
            idx = np.random.choice(n, 3, replace=False)
            p1, p2, p3 = pts[idx]
            nrm = np.cross(p2 - p1, p3 - p1)
            nl = np.linalg.norm(nrm)
            if nl < 1e-6:
                continue
            nrm /= nl
            if abs(nrm[1]) < 0.7:
                continue
            if nrm[1] < 0:
                nrm = -nrm
            d = float(np.dot(nrm, p1))
            cnt = int(np.sum(np.abs(pts @ nrm - d) < RANSAC_DIST))
            if cnt > best_count:
                best_count = cnt
                best_n = nrm.copy()
                best_d = d
        if best_n is None or best_count < n * RANSAC_MIN_R:
            return None, None
        return best_n, best_d

    def _cluster(self, pts):
        if len(pts) < CLUSTER_MIN:
            return []
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN,
                        algorithm='ball_tree', n_jobs=1).fit_predict(pts)
        clusters = []
        for lb in np.unique(labels):
            if lb == -1:
                continue
            mask = labels == lb
            cnt = int(np.sum(mask))
            if CLUSTER_MIN <= cnt <= CLUSTER_MAX:
                clusters.append(pts[mask])
        return clusters


def main():
    rclpy.init()
    node = DepthClusterTest()

    print("等待深度数据 + 内参...")
    # 等待数据就绪
    timeout = time.time() + 10.0
    while rclpy.ok() and time.time() < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.has_intrinsics and node.depth_image is not None:
            break
    else:
        if not node.has_intrinsics or node.depth_image is None:
            print("超时: 未收到深度数据或内参")
            node.destroy_node()
            rclpy.shutdown()
            return

    print("开始处理 (Ctrl+C 退出)...\n")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.process_once()
            time.sleep(0.5)  # 2Hz 输出
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
