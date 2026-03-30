#!/usr/bin/env python3
"""Compute Piper arm precise workspace via Monte Carlo FK sampling.

Uses SDK DH parameters + joint limits to sample the full reachable space
of the gripper center, and outputs workspace statistics + recommended config.

Usage:
    python3 scripts/_cc_arm_workspace.py              # Default 1M samples
    python3 scripts/_cc_arm_workspace.py --live        # Query arm firmware limits
    python3 scripts/_cc_arm_workspace.py --plot        # Save scatter plots
    python3 scripts/_cc_arm_workspace.py --samples 5000000  # 5M samples
"""

import sys
import math
import time
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# SDK default joint limits (radians) from piper_param_manager.py
# ---------------------------------------------------------------------------
DEFAULT_JOINT_LIMITS = [
    (-2.6179,  2.6179),   # J1: +/-150 deg
    ( 0.0,     3.14),     # J2:   0 ~ 180 deg
    (-2.967,   0.0),      # J3: -170 ~ 0 deg
    (-1.745,   1.745),    # J4: +/-100 deg
    (-1.22,    1.22),     # J5: +/-70 deg
    (-2.09439, 2.09439),  # J6: +/-120 deg
]

# ---------------------------------------------------------------------------
# DH parameters (dh_is_offset=0x01, default for real arm)
# From piper_sdk/kinematics/piper_fk.py
# ---------------------------------------------------------------------------
DH_A     = np.array([0.0,  0.0,             285.03, -21.98,  0.0,       0.0])
DH_ALPHA = np.array([0.0, -math.pi / 2,     0.0,    math.pi / 2, -math.pi / 2, math.pi / 2])
DH_THETA = np.array([0.0, -172.22 * math.pi / 180, -102.78 * math.pi / 180, 0.0, 0.0, 0.0])
DH_D     = np.array([123.0, 0.0,            0.0,    250.75,  0.0,      91.0])

GRIPPER_OFFSET_MM = 135.03  # flange-to-gripper-center along flange Z


def dh_transform_batch(alpha, a, theta_arr, d):
    """Vectorised DH transform for N samples.

    Returns (N,4,4) homogeneous matrices.
    """
    N = theta_arr.shape[0]
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct = np.cos(theta_arr)
    st = np.sin(theta_arr)

    T = np.zeros((N, 4, 4))
    T[:, 0, 0] = ct;         T[:, 0, 1] = -st;         T[:, 0, 3] = a
    T[:, 1, 0] = st * ca;    T[:, 1, 1] = ct * ca;     T[:, 1, 2] = -sa;  T[:, 1, 3] = -sa * d
    T[:, 2, 0] = st * sa;    T[:, 2, 1] = ct * sa;     T[:, 2, 2] = ca;   T[:, 2, 3] = ca * d
    T[:, 3, 3] = 1.0
    return T


def compute_workspace(joints):
    """Batch FK: (N,6) joint angles -> (N,3) gripper-center positions.

    Also returns flange positions and flange Z axis.
    """
    N = joints.shape[0]
    T_acc = np.broadcast_to(np.eye(4), (N, 4, 4)).copy()

    for i in range(6):
        Ti = dh_transform_batch(DH_ALPHA[i], DH_A[i],
                                joints[:, i] + DH_THETA[i], DH_D[i])
        T_acc = np.matmul(T_acc, Ti)

    flange_xyz = T_acc[:, :3, 3]
    flange_z   = T_acc[:, :3, 2]   # Z-column of rotation
    gc_xyz     = flange_xyz + GRIPPER_OFFSET_MM * flange_z
    return flange_xyz, gc_xyz, flange_z


def verify_fk():
    """Cross-check against SDK C_PiperForwardKinematics."""
    from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics
    fk_sdk = C_PiperForwardKinematics(dh_is_offset=0x01)

    cases = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.3, -0.5, 0.2, 0.1, -0.3],
        [1.0, 1.5, -1.0, 0.5, -0.5, 1.0],
        [-1.0, 0.8, -2.0, -0.5, 0.8, 0.0],
    ]

    print("\n--- FK 验证 (vs SDK) ---")
    max_err = 0.0
    for q in cases:
        sdk_pos = fk_sdk.CalFK(q)[5][:3]  # link6 xyz (mm)
        our_flange, _, _ = compute_workspace(np.array([q]))
        err = np.linalg.norm(np.array(sdk_pos) - our_flange[0])
        max_err = max(max_err, err)
        print(f"  q={[f'{v: .2f}' for v in q]}  "
              f"SDK=({sdk_pos[0]:7.1f},{sdk_pos[1]:7.1f},{sdk_pos[2]:7.1f})  "
              f"err={err:.4f}mm")

    ok = max_err < 0.01
    print(f"  Max error: {max_err:.6f} mm  {'PASS' if ok else 'FAIL'}")
    return ok


def try_query_firmware_limits(can_port):
    """Query joint limits from live arm firmware. Returns list of (lo,hi) or None."""
    try:
        from piper_sdk import C_PiperInterface_V2
        print(f"  连接 {can_port} ...")
        arm = C_PiperInterface_V2(can_port, judge_flag=False, can_auto_init=False)
        arm.ConnectPort()
        time.sleep(0.5)

        arm.SearchAllMotorMaxAngleSpd()
        time.sleep(1.0)

        info = arm.GetAllMotorAngleLimitMaxSpd()
        limits = []
        for i in range(1, 7):
            m = info.all_motor_angle_limit_max_spd.motor[i]
            lo_deg = m.min_angle_limit * 0.1   # unit: 0.1 deg
            hi_deg = m.max_angle_limit * 0.1
            lo_rad = math.radians(lo_deg)
            hi_rad = math.radians(hi_deg)
            limits.append((lo_rad, hi_rad))
            print(f"  J{i}: [{lo_deg:7.1f}, {hi_deg:7.1f}] deg  = [{lo_rad:.4f}, {hi_rad:.4f}] rad")

        # Sanity check: all limits non-zero?
        if all(abs(hi - lo) > 0.01 for lo, hi in limits):
            return limits
        print("  WARNING: 固件返回零限位, 回退到 SDK 默认值")
        return None
    except Exception as e:
        print(f"  无法连接机械臂: {e}")
        return None


def print_stats(label, pos, xy):
    """Print XYZ + XY reach stats."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    print(f"\n--- {label} ---")
    print(f"  X  range: [{x.min():8.1f}, {x.max():8.1f}] mm")
    print(f"  Y  range: [{y.min():8.1f}, {y.max():8.1f}] mm")
    print(f"  Z  range: [{z.min():8.1f}, {z.max():8.1f}] mm")
    print(f"  XY reach: [{xy.min():8.1f}, {xy.max():8.1f}] mm")


def main():
    parser = argparse.ArgumentParser(description="Piper arm workspace calculator")
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--live", action="store_true", help="Query firmware limits")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--plot", action="store_true", help="Save scatter plots")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(" Piper Arm  workspace calculator")
    print("=" * 60)

    # -- Joint limits -------------------------------------------------------
    limits = None
    if args.live:
        print("\n[1] 查询固件关节限位...")
        limits = try_query_firmware_limits(args.can)

    if limits is None:
        limits = DEFAULT_JOINT_LIMITS
        print("\n[1] SDK 默认关节限位:")
        for i, (lo, hi) in enumerate(limits):
            print(f"  J{i+1}: [{math.degrees(lo):7.1f}, {math.degrees(hi):7.1f}] deg"
                  f"  = [{lo:.4f}, {hi:.4f}] rad")

    # -- FK verification ----------------------------------------------------
    if not args.no_verify:
        if not verify_fk():
            print("FK verification FAILED -- aborting")
            sys.exit(1)

    # -- Monte Carlo sampling -----------------------------------------------
    N = args.samples
    print(f"\n[2] 蒙特卡洛采样 {N:,} 个关节配置...")

    rng = np.random.default_rng(42)
    joints = np.column_stack([
        rng.uniform(lo, hi, N) for lo, hi in limits
    ])

    t0 = time.time()
    flange_pos, gc_pos, _ = compute_workspace(joints)
    dt = time.time() - t0
    print(f"  FK 完成: {dt:.2f}s  ({N / dt:.0f} samples/s)")

    # -- Statistics ---------------------------------------------------------
    gc_x, gc_y, gc_z = gc_pos[:, 0], gc_pos[:, 1], gc_pos[:, 2]
    gc_xy = np.sqrt(gc_x**2 + gc_y**2)

    fl_xy = np.sqrt(flange_pos[:, 0]**2 + flange_pos[:, 1]**2)

    print("\n" + "=" * 60)
    print(" Workspace Statistics")
    print("=" * 60)

    print_stats("Flange (法兰)", flange_pos, fl_xy)
    print_stats("Gripper Center (夹爪中心, +135mm)", gc_pos, gc_xy)

    # Percentiles
    print("\n--- GC XY Reach 百分位 ---")
    for p in [50, 90, 95, 99, 99.5, 99.9]:
        print(f"  P{p:>5}: {np.percentile(gc_xy, p):7.1f} mm")
    print(f"  P  100: {gc_xy.max():7.1f} mm")

    print("\n--- GC Z 百分位 ---")
    for p in [0.1, 1, 5, 50, 95, 99, 99.9]:
        print(f"  P{p:>5}: {np.percentile(gc_z, p):7.1f} mm")

    # Grasping subset: X>0, -300<Z<400
    mask = (gc_x > 0) & (gc_z > -300) & (gc_z < 400)
    n_ok = mask.sum()
    print(f"\n--- 前方抓取区域 (X>0, -300<Z<400) ---")
    print(f"  有效样本: {n_ok:,} / {N:,}  ({100 * n_ok / N:.1f}%)")
    if n_ok > 0:
        print(f"  X:  [{gc_x[mask].min():.1f}, {gc_x[mask].max():.1f}] mm")
        print(f"  Y:  [{gc_y[mask].min():.1f}, {gc_y[mask].max():.1f}] mm")
        print(f"  Z:  [{gc_z[mask].min():.1f}, {gc_z[mask].max():.1f}] mm")
        print(f"  XY: [{gc_xy[mask].min():.1f}, {gc_xy[mask].max():.1f}] mm")

    # -- Downward-grasping analysis (gripper pointing ~down) ----------------
    # Flange Z-axis dot with (0,0,-1): measures how "downward" the gripper is
    # cos(angle) = -flange_z[:,2]; angle < 30deg => cos > 0.866
    _, _, flange_z_all = compute_workspace(joints)  # reuse joints
    down_cos = -flange_z_all[:, 2]  # cos of angle to (0,0,-1)

    for max_tilt_deg in [15, 30, 45]:
        cos_thresh = math.cos(math.radians(max_tilt_deg))
        dmask = (down_cos >= cos_thresh) & (gc_x > 0)
        n_d = dmask.sum()
        print(f"\n--- 朝下抓取 (tilt<{max_tilt_deg}deg, X>0) ---")
        print(f"  有效样本: {n_d:,} / {N:,}  ({100 * n_d / N:.1f}%)")
        if n_d > 100:
            xy_d = gc_xy[dmask]
            z_d = gc_z[dmask]
            print(f"  XY reach: [{xy_d.min():.1f}, {xy_d.max():.1f}] mm  "
                  f"P95={np.percentile(xy_d, 95):.1f}  P99={np.percentile(xy_d, 99):.1f}")
            print(f"  Z  range: [{z_d.min():.1f}, {z_d.max():.1f}] mm  "
                  f"P5={np.percentile(z_d, 5):.1f}  P95={np.percentile(z_d, 95):.1f}")

    # -- Recommendation ----------------------------------------------------
    # Use the 30-deg tilt mask as the practical grasping envelope
    grasp_cos = math.cos(math.radians(30))
    grasp_mask_full = (down_cos >= grasp_cos) & (gc_x > 0)
    if grasp_mask_full.sum() > 100:
        grasp_xy = gc_xy[grasp_mask_full]
        grasp_p99 = np.percentile(grasp_xy, 99)
        grasp_p999 = np.percentile(grasp_xy, 99.9)
        grasp_max = grasp_xy.max()
    else:
        grasp_p99 = gc_xy.max()
        grasp_p999 = gc_xy.max()
        grasp_max = gc_xy.max()

    p999 = np.percentile(gc_xy, 99.9)
    p100 = gc_xy.max()

    print("\n" + "=" * 60)
    print(" 建议 working_area 配置")
    print("=" * 60)
    print(f"  当前 max_reach:          435 mm  (经验值)")
    print(f"  理论最大 XY reach (max): {p100:.1f} mm  (所有构型)")
    print(f"  朝下抓取 XY reach (P99): {grasp_p99:.1f} mm  (tilt<30deg, X>0)")
    print(f"  朝下抓取 XY reach (P99.9): {grasp_p999:.1f} mm")
    print(f"  朝下抓取 XY reach (max): {grasp_max:.1f} mm")
    recommended = int(grasp_p99)
    print(f"  建议 max_reach:          {recommended} mm  (抓取P99)")
    print()
    print(f"  GC X 全范围: [{gc_x.min():.1f}, {gc_x.max():.1f}] mm")
    print(f"  GC Y 全范围: [{gc_y.min():.1f}, {gc_y.max():.1f}] mm")
    print(f"  GC Z 全范围: [{gc_z.min():.1f}, {gc_z.max():.1f}] mm")
    print()
    print("  # 建议 YAML (基于 FK + 抓取姿态约束):")
    print("  working_area:")
    print(f"    max_reach: {recommended}    # mm, 朝下抓取 P99 XY reach")
    print(f"    # 无约束理论极限: {p100:.0f}mm")

    # -- Optional plot ------------------------------------------------------
    if args.plot:
        _save_plots(gc_pos, gc_xy, int(p999), rng, N)


def _save_plots(gc_pos, gc_xy, max_reach, rng, N):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib 未安装, 跳过绘图")
        return

    n_plot = min(50_000, N)
    idx = rng.choice(N, n_plot, replace=False)
    gx, gy, gz = gc_pos[idx, 0], gc_pos[idx, 1], gc_pos[idx, 2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Top view XY
    ax = axes[0]
    sc = ax.scatter(gx, gy, s=0.2, alpha=0.15, c=gz, cmap="viridis", rasterized=True)
    plt.colorbar(sc, ax=ax, label="Z (mm)")
    circle = plt.Circle((0, 0), max_reach, fill=False, color="r", lw=1.5,
                         label=f"max_reach={max_reach}mm")
    ax.add_patch(circle)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.set_title("Top View (XY)"); ax.set_aspect("equal"); ax.legend(fontsize=8)

    # Side view XZ
    ax = axes[1]
    ax.scatter(gx, gz, s=0.2, alpha=0.15, c="steelblue", rasterized=True)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Z (mm)")
    ax.set_title("Side View (XZ)"); ax.set_aspect("equal")

    # Front view YZ
    ax = axes[2]
    ax.scatter(gy, gz, s=0.2, alpha=0.15, c="seagreen", rasterized=True)
    ax.set_xlabel("Y (mm)"); ax.set_ylabel("Z (mm)")
    ax.set_title("Front View (YZ)"); ax.set_aspect("equal")

    for ax in axes:
        ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "/data/workspace/MobileManipulator2/scripts/arm_workspace.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  散点图已保存: {out}")


if __name__ == "__main__":
    main()
