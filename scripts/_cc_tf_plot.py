#!/usr/bin/env python3
"""TF 实时录制 + 绘图工具

用法:
  # 录制 10 秒后自动绘图
  python3 scripts/_cc_tf_plot.py --duration 10

  # 录制 30 秒，指定 TF 对
  python3 scripts/_cc_tf_plot.py --duration 30 --frames map:base_link camera_init:body

  # 只保存数据不画图
  python3 scripts/_cc_tf_plot.py --duration 10 --no-plot

  # 从已有数据绘图
  python3 scripts/_cc_tf_plot.py --load scripts/tf_log_xxx.jsonl
"""
import argparse, json, math, os, signal, sys, time
from pathlib import Path

def record(frames_pairs, duration, rate_hz, outfile):
    """录制 TF 数据到 jsonl 文件"""
    import rclpy
    from rclpy.node import Node
    from tf2_ros import Buffer, TransformListener

    rclpy.init()

    class Recorder(Node):
        def __init__(self):
            super().__init__('tf_recorder')
            self.buf = Buffer()
            self.listener = TransformListener(self.buf, self)

    node = Recorder()
    import threading
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # warmup
    time.sleep(0.5)

    f = open(outfile, 'w')
    t0 = time.time()
    dt = 1.0 / rate_hz
    count = 0

    print(f"Recording TF for {duration}s at {rate_hz}Hz → {outfile}")
    print(f"  Frames: {['→'.join(p) for p in frames_pairs]}")

    try:
        while time.time() - t0 < duration:
            now = node.get_clock().now()
            wall = time.time() - t0

            for parent, child in frames_pairs:
                try:
                    tf = node.buf.lookup_transform(parent, child, rclpy.time.Time())
                    t = tf.transform.translation
                    r = tf.transform.rotation
                    # yaw
                    siny = 2.0 * (r.w * r.z + r.x * r.y)
                    cosy = 1.0 - 2.0 * (r.y * r.y + r.z * r.z)
                    yaw = math.atan2(siny, cosy)
                    # pitch
                    sinp = 2.0 * (r.w * r.y - r.z * r.x)
                    pitch = math.asin(max(-1, min(1, sinp)))
                    # roll
                    sinr = 2.0 * (r.w * r.x + r.y * r.z)
                    cosr = 1.0 - 2.0 * (r.x * r.x + r.y * r.y)
                    roll = math.atan2(sinr, cosr)
                    # stamp age
                    stamp_ns = tf.header.stamp.sec * 1_000_000_000 + tf.header.stamp.nanosec
                    age_ms = (now.nanoseconds - stamp_ns) / 1e6

                    rec = {
                        'wall': round(wall, 4),
                        'frame': f'{parent}→{child}',
                        'x': round(t.x, 5), 'y': round(t.y, 5), 'z': round(t.z, 5),
                        'roll': round(math.degrees(roll), 3),
                        'pitch': round(math.degrees(pitch), 3),
                        'yaw': round(math.degrees(yaw), 3),
                        'age_ms': round(age_ms, 2),
                    }
                    f.write(json.dumps(rec) + '\n')
                    count += 1
                except Exception:
                    pass

            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass

    print(f"Done: {count} samples in {time.time()-t0:.1f}s → {outfile}")
    return outfile


def load_data(filepath):
    """从 jsonl 文件加载数据"""
    records = {}
    with open(filepath) as f:
        for line in f:
            rec = json.loads(line)
            frame = rec['frame']
            if frame not in records:
                records[frame] = []
            records[frame].append(rec)
    return records


def plot(records, title, save_path):
    """绘制 TF 曲线"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frames = list(records.keys())
    n_frames = len(frames)

    # 5 行: x, y, z, yaw, age
    fig, axes = plt.subplots(5, n_frames, figsize=(7 * n_frames, 14),
                              squeeze=False, sharex='col')
    fig.suptitle(title, fontsize=14, fontweight='bold')

    labels = ['X (m)', 'Y (m)', 'Z (m)', 'Yaw (°)', 'Age (ms)']
    keys = ['x', 'y', 'z', 'yaw', 'age_ms']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    for col, frame in enumerate(frames):
        data = records[frame]
        wall = [d['wall'] for d in data]

        for row, (key, label, color) in enumerate(zip(keys, labels, colors)):
            ax = axes[row][col]
            vals = [d[key] for d in data]
            ax.plot(wall, vals, color=color, linewidth=0.8, alpha=0.9)

            # 均值线
            if len(vals) > 0:
                import statistics
                mean_v = statistics.mean(vals)
                ax.axhline(y=mean_v, color=color, linestyle='--', alpha=0.4, linewidth=0.6)
                ax.text(0.98, 0.95, f'avg={mean_v:.3f}',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=8, color=color, alpha=0.7)

            if row == 0:
                ax.set_title(frame, fontsize=11, fontweight='bold')
            ax.set_ylabel(label, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

        axes[-1][col].set_xlabel('Time (s)', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {save_path}")

    # 也画一个 XY 轨迹图
    if any('x' in records[f][0] for f in frames):
        fig2, axes2 = plt.subplots(1, n_frames, figsize=(7 * n_frames, 6), squeeze=False)
        fig2.suptitle(f'{title} - XY Trajectory', fontsize=14, fontweight='bold')

        for col, frame in enumerate(frames):
            data = records[frame]
            xs = [d['x'] for d in data]
            ys = [d['y'] for d in data]
            wall = [d['wall'] for d in data]

            ax = axes2[0][col]
            sc = ax.scatter(xs, ys, c=wall, cmap='viridis', s=2, alpha=0.8)
            ax.plot(xs[0], ys[0], 'go', markersize=8, label='start')
            ax.plot(xs[-1], ys[-1], 'r^', markersize=8, label='end')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(frame, fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            plt.colorbar(sc, ax=ax, label='Time (s)', shrink=0.8)

        plt.tight_layout()
        xy_path = save_path.replace('.png', '_xy.png')
        plt.savefig(xy_path, dpi=150, bbox_inches='tight')
        print(f"XY plot saved: {xy_path}")


def main():
    parser = argparse.ArgumentParser(description='TF curve recorder & plotter')
    parser.add_argument('--duration', type=float, default=10, help='Recording duration (s)')
    parser.add_argument('--rate', type=float, default=50, help='Sampling rate (Hz)')
    parser.add_argument('--frames', nargs='+', default=['map:base_link', 'camera_init:body'],
                        help='TF pairs as parent:child')
    parser.add_argument('--load', type=str, help='Load existing jsonl instead of recording')
    parser.add_argument('--no-plot', action='store_true', help='Only record, no plot')
    parser.add_argument('--output', type=str, help='Output file prefix')
    args = parser.parse_args()

    frames_pairs = [f.split(':') for f in args.frames]

    ts = time.strftime('%Y%m%d_%H%M%S')
    prefix = args.output or f'scripts/tf_log_{ts}'

    if args.load:
        data_file = args.load
    else:
        data_file = f'{prefix}.jsonl'
        record(frames_pairs, args.duration, args.rate, data_file)

    if not args.no_plot:
        records = load_data(data_file)
        plot(records, f'TF Curves ({os.path.basename(data_file)})', f'{prefix}.png')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    main()
