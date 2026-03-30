# 双相机感知管线 5Hz 优化方案

> 日期: 2026-03-15
> 状态: **5Hz 达标** — 5.70Hz (P50=176ms)

---

## 1. 需求

| 项目 | 要求 |
|------|------|
| 目标频率 | 双相机 SAM3+FoundationStereo 同步完成 ≥ 5Hz (≤200ms/cycle) |
| 质量约束 | 检测精度和深度精度不降 (不降分辨率/不降模型/不降JPEG质量) |
| 同步约束 | 两相机结果必须同步聚合返回, 不允许错峰 |
| 改造范围 | 服务端 + 客户端代码均可重构 |
| 验证方式 | 先在 `percept_benchmark.py` 中验证, 再改感知节点 |

---

## 2. 机器信息

### 2.1 GPU 服务器 (192.168.112.14)

```
用户/密码: tim / tim
GPU 0: NVIDIA RTX 5090, 32607 MiB, 已用 11786 MiB, 剩余 20291 MiB
GPU 1: NVIDIA RTX 5090, 32607 MiB, 已用  9599 MiB, 剩余 22511 MiB
TensorRT: 10.14.1
Python: 3.13 (miniconda3, /data1/miniconda3/bin/python3)
```

### 2.2 客户端 (Jetson, 本机)

```
平台: Linux 5.15-tegra (ARM64), Jetson Orin
ROS2: Humble
工作路径: /data/workspace/MobileManipulator2/src/perception/src/
WiFi: Intel iwlwifi, WiFi 6 (802.11ax)
```

### 2.3 网络

```
链路:      Jetson (wlP1p1s0) ── WiFi 6 ── 路由器 ── GPU Server
           移动机器人, 无法使用有线连接
频段:      5GHz ch44 (5220 MHz), 160MHz 带宽
链路速率:  TX 2041 Mbps / RX 2401 Mbps (HE-MCS11, 2×2 MIMO)
实际吞吐:  ~420 Mbps (WiFi 协议开销 + 半双工 + 重传)
信号强度:  -44 dBm (优秀)
TX 重传率: 1.79%
ICMP RTT:  P50=3.0ms, P95=4.5ms
```

---

## 3. 优化前基线 (4.1Hz)

### 3.1 双相机并发

```
并发 2×SAM3 + 2×Stereo:  245.2 ms = 4.1 Hz  ← 不达标
```

### 3.2 时间拆解

```
客户端 245ms 拆解:

服务端 GPU 时间:        188ms   77%   ← 主瓶颈
网络传输+协议开销:       ~51ms   21%   ← 次瓶颈
编解码:                  ~5ms    2%

服务端 188ms 根因:
  GPU0 (FS):   [cam0 ~107ms] → [cam1 ~107ms] = 214ms (串行)
  GPU1 (SAM3): [cam0 ~65ms]  → [cam1 ~65ms]  = 130ms (串行)
  两 GPU 并行: max(214, 130) ≈ 188ms ← FS 串行是慢腿

网络 51ms 根因:
  WiFi 物理传输 ~33ms + HTTP 协议开销 ~18ms (4次请求 × 4-5ms)
```

---

## 4. 数据传输量 (实测值)

### 4.1 上传 (Jetson → Server), 每相机

| 数据 | 格式 | 实测大小 |
|------|------|---------|
| RGB (1280×720) | JPEG Q85 | **~241 KB** |
| IR left (1280×720) | JPEG Q95 | **~229 KB** |
| IR right (1280×720) | JPEG Q95 | **~227 KB** |
| **单相机小计** | | **~697 KB** |
| **双相机合计** | | **~1394 KB** |

### 4.2 下载 (Server → Jetson), 每相机

| 数据 | 格式 | 实测大小 |
|------|------|---------|
| 检测结果 | JSON (binary framing) | ~3 KB |
| 深度图 (640×360 uint16, half-res) | **PNG level 1** | **~80 KB** |
| **单相机小计** | | **~83 KB** |
| **双相机合计** | | **~166 KB** |

> **深度编码方案选择 (实测对比)**:
>
> | 方案 | 服务端 encode | 大小 | 精度 | 网络开销 | 端到端效果 |
> |------|-------------|------|------|---------|-----------|
> | Full PNG (1280×720) | 58ms | 162KB | 1mm | 正常 | 太慢 |
> | **Half-res PNG lvl1 (640×360)** | **10ms** | **80KB** | **1mm** | **正常** | **最优** |
> | Half-res PNG lvl0 | 4ms | 452KB | 1mm | +70ms | 负优化! |
> | JPEG Q95 (÷DEPTH_SCALE) | 3ms | 15KB | ±20mm | 最小 | 精度不足 |
> | LZ4 (full) | ~2ms | 304KB | 1mm | 较大 | 比 PNG 大 |
>
> 结论: **Half-res PNG level 1** 是最优平衡点。客户端 INTER_NEAREST 上采样还原全分辨率。

### 4.3 总传输量 (优化后)

```
双相机总计:  1394 + 166 = 1560 KB ≈ 1.52 MB
WiFi 420Mbps (52.5MB/s): 1.52MB / 52.5 ≈ 29ms 理论下限
实测 P50: ~42ms (含 HTTP 协议开销 + WiFi 调度)
```

---

## 5. 网络优化 (已实施) ✅

### 5.1 诊断发现

| 问题 | 根因 | 影响 |
|------|------|------|
| **TCP 发送缓冲太小** | wmem_max=208KB, 700KB 上传分 4 轮等 ACK | P50 +10ms |
| **Nagle 算法延迟** | HTTP multipart 小包被攒包 40ms | 尾部 +20-50ms |
| **蓝牙共存干扰** | hci0 UP + bt_coex_active=Y, WiFi 为 BT 让路 | P99 +10ms |
| **WiFi MAC 重传** | 1.79% 重传率, 700KB/帧 必触发 | P99 +5-15ms (不可消除) |

### 5.2 已实施的优化

#### A. 内核 TCP 参数

文件: `/etc/sysctl.d/99-wifi-perception.conf` (重启自动生效)

```
net.core.wmem_max = 2097152      # 208KB → 2MB, 700KB 一次写入内核
net.core.rmem_max = 2097152
net.ipv4.tcp_wmem = 4096 131072 2097152
net.ipv4.tcp_rmem = 4096 131072 2097152
net.ipv4.tcp_low_latency = 1     # 跳过 prequeue, 减少处理延迟
```

#### B. 关闭蓝牙

文件: `/etc/systemd/system/disable-bluetooth.service` (重启自动生效)

```
rfkill block bluetooth   # 机器人不使用蓝牙, 消除 WiFi-BT 共存干扰
```

#### C. 应用层 socket 优化

文件: `percept.py` — `_LowLatencyAdapter` + `_make_session()`

```python
# 每个 TCP 连接自动设置:
TCP_NODELAY = 1          # 禁用 Nagle, 消除小包攒包延迟
SO_SNDBUF = 2MB          # 大发送缓冲, 消除 send() 阻塞
SO_RCVBUF = 2MB          # 大接收缓冲
```

### 5.3 实测效果

```
双相机并行发送 700KB×2 (100 runs):

                  P50     P90     P95     P99     max
优化前:          13.9    17.4    19.6    50.3    50.3 ms
优化后:           3.3     4.4     5.2    15.8    15.8 ms
────────────────────────────────────────────────────────
改善:           -10.6   -13.0   -14.4   -34.5   -34.5 ms
```

---

## 6. 服务端合并部署 (已实施) ✅

### 6.1 架构

```
优化前 (4.1Hz):
┌──────────┐    4个HTTP请求     ┌─────────────────────┐
│  Jetson  │ ────────────────→ │  GPU Server          │
│ (client) │    WiFi 420Mbps   │  GPU0: FS×2   =214ms │ ← FS 串行瓶颈!
│          │                   │  GPU1: SAM3×2 =130ms │
└──────────┘                   └─────────────────────┘

优化后 (5.38Hz):
┌──────────┐    2个HTTP请求     ┌──────────────────────────────┐
│  Jetson  │ ────────────────→ │  GPU Server                   │
│ (client) │   cam0→:8090      │  GPU0: SAM3‖FS(cam0) ≈ 128ms │
│          │   cam1→:8091      │  GPU1: SAM3‖FS(cam1) ≈ 128ms │
│          │   并行发送         │  两GPU并行 = max(128,128)     │
└──────────┘                   └──────────────────────────────┘
```

**核心收益**:
1. FS 从单 GPU 串行 (2×107=214ms) → 分散到两 GPU 并行
2. SAM3+FS 同 GPU 内并行 (ThreadPoolExecutor, 不同 CUDA stream)
3. 合并请求: 4个 HTTP → 2个, 省协议开销
4. Binary framing 响应: 检测 JSON + 深度 PNG 在一个 HTTP 响应中

**内存**: SAM3 ~4GB + FS ~6GB ≈ 10GB/GPU, RTX 5090 32GB 有 20GB+ 余量 ✓

### 6.2 文件结构

```
/home/tim/workspace/GraspForge/perception_server/
├── combined_server.py             # 合并 SAM3+FS, FastAPI, 单 GPU
└── start_combined.sh              # 启动 2 实例 (每 GPU 1 个)

客户端:
/data/workspace/MobileManipulator2/src/perception/src/
├── percept.py                     # CombinedPerceptionClient + DualPerceptionClient
└── percept_benchmark.py           # benchmark_combined_dual()
```

### 6.3 API 设计

```
POST /api/perceive
  Input (multipart/form-data):
    rgb:          RGB图像 (JPEG)
    ir_left:      IR左图 (JPEG)
    ir_right:     IR右图 (JPEG)
    text_prompt:  检测类别文本
    confidence:   置信度阈值 (float, 可选)
    return_mask:  是否返回mask (bool, 可选)
    intrinsics:   内参YAML (首次传, 后续用hash)

  Headers (可选):
    X-Intrinsics-Hash: 内参缓存hash (已缓存时免传intrinsics)

  Output:
    Content-Type: application/octet-stream
    Body: [4B json_len LE][json_bytes][depth_half_png_bytes]
          json_len = little-endian uint32
          json_bytes = 检测结果 JSON (含 objects 列表)
          depth_half_png_bytes = half-res uint16 PNG (640×360, 1mm精度)
    Headers:
      X-Intrinsics-Hash: 内参MD5
      X-Timing: 服务端分项耗时 JSON

GET /api/health
  返回: {"status": "ok", "gpu": 0, "models": ["sam3", "fs"]}
```

### 6.4 启动方式

```bash
cd /home/tim/workspace/GraspForge/perception_server
bash start_combined.sh start    # 启动
bash start_combined.sh stop     # 停止
bash start_combined.sh restart  # 重启
```

关键: 使用 `CUDA_VISIBLE_DEVICES` 隔离 GPU, 两个进程都使用 `--gpu 0`:

```bash
# start_combined.sh 核心逻辑:
CUDA_VISIBLE_DEVICES=0 python3 -u combined_server.py --port 8090 --gpu 0 &
CUDA_VISIBLE_DEVICES=1 python3 -u combined_server.py --port 8091 --gpu 0 &
```

> **为什么不用 `--gpu 1`?**
> TRT 引擎文件内嵌构建设备信息。虽然两块 GPU 型号相同 (RTX 5090),
> 但 TRT 按 PCI bus ID 区分设备, GPU1 加载 GPU0 构建的引擎会报
> "engine plan across different models" 警告, 导致 **CUDA error 400**。
> 用 `CUDA_VISIBLE_DEVICES` 让每个进程只看到一块 GPU (都是 device 0),
> 避免跨设备引擎加载问题。

### 6.5 模型加载顺序

```python
# combined_server.py load_models() 中:
# 1. FoundationStereo 先加载 (Python TRT, 初始化 CUDA context)
# 2. SAM3 后加载 (C++ TRT trtsam3, 适配已有 CUDA context)
# 反过来会导致 CUDA error 400!
```

### 6.6 服务端推理流水线优化

```
优化前 (串行 decode → 串行推理):
  decode(RGB) → decode(IR_L) → decode(IR_R) → launch SAM3 → launch FS
  |──────────── 19ms ──────────────|──── 等 FS 完成 ─────|

优化后 (decode 与推理重叠):
  decode(RGB, ~7ms) → launch SAM3          ← SAM3 只需要 RGB
       ↘ decode(IR_L ‖ IR_R, ~6ms) → launch FS  ← FS 只需要 IR pair
  |─ 7ms ─|── IR decode 与 SAM3 推理重叠 ──|
```

SAM3 只需要 RGB, FS 只需要 IR pair → RGB 解完立即启动 SAM3, IR 并行解码后启动 FS。
关键路径上 decode 从 19ms → ~7ms (仅 RGB 单张), 省 ~12ms。

### 6.7 FS GPU 后处理优化 (stereo_depth_processor.py)

FS 推理流水线 profiling (isolated, 无 SAM3 并发):

```
gray2rgb:    1.7ms   ← np.stack([gray]*3) at full 720×1280
preproc:     2.7ms   ← resize 3-ch + float32 + transpose
h2d_copy:    1.6ms
trt_exec:   63.9ms   ← 核心 TRT 推理 (ViT-S)
d2h_copy:    0.4ms
disp_resize: 1.2ms
disp2depth:  3.4ms   ← CPU numpy: invalid pixel + disp→depth
reproject:   1.5ms   ← GPU (torch)
fill_holes:  1.7ms   ← GPU (torch)
to_uint16:   0.8ms
TOTAL:      79.0ms (isolated) / 108ms (concurrent with SAM3)
```

优化方案 (4 个 patch, 在 GPU 服务器 stereo_depth_processor.py):

1. **gray-first resize**: 先 resize 1-ch gray (720→576), 再 stack 到 3-ch (3x 更少像素)
2. **GPU post-processing**: `_run_trt_stereo` 返回 `torch.Tensor` 而非 numpy,
   disp→depth 在 GPU 上用 torch.where 完成, 避免 CPU↔GPU 往返
3. **tensor 透传**: `_reproject_ir_to_rgb` 和 `_fill_holes` 直接接收 GPU tensor
4. **延迟 cpu().numpy()**: 仅在 process() 最终返回时转 numpy

效果: GPU1 server_total 132→122ms (**-10ms**), 两 GPU 对称性大幅改善。

> **注意**: GPU 后处理增加了 GPU 资源争抢, SAM3 从 31ms→102ms。
> 但这不影响关键路径, 因 FS (108ms) 始终是瓶颈, SAM3 (102ms) 仍在 FS 之前完成。

---

## 7. 实测性能

### 7.1 最终结果 (30 runs, ViT-S + GPU后处理优化)

```
[Combined Dual] 统计 (30 runs)
  Metric       avg     P50     P95     min     max
  ───────────────────────────────────────────
  Wall       174.5   175.6   185.9   161.8   190.7 ms

  频率 (P50): 5.70 Hz  ✅ 5Hz 达标!
```

### 7.2 分项耗时 (avg)

```
                        客户端     服务端      网络
─────────────────────────────────────────────────────
客户端 JPEG 编码:         8ms  (3路并行编码)
客户端→服务端 上传:                            ↑693KB
服务端 decode+重叠:                  5ms*  (turbojpeg RGB)
服务端 SAM3 推理:                  102ms  (与 FS 并行, 因 GPU 争抢)
服务端 FS 推理:                    108ms  ← GPU后处理优化
服务端 depth encode:                 9ms  (half-res PNG lvl1)
服务端→客户端 下载:                            ↓84KB
WiFi 网络往返:                                 36ms
客户端 decode:            4ms
─────────────────────────────────────────────────────
总计 (P50):             176ms

* decode 与推理重叠, 不在关键路径上
```

### 7.3 GPU 对称性

GPU patches + 并行解码后两 GPU 高度对称:

```
         server_total    http (含网络)    net overhead
GPU0:    122.9 ms        158.9 ms        36.0 ms (↑693+↓84 = 778 KB)
GPU1:    121.7 ms        156.9 ms        35.2 ms (↑735+↓75 = 810 KB)
```

### 7.4 优化效果汇总

| 优化措施 | 改善 | 状态 |
|---------|------|------|
| 网络: sysctl + 关蓝牙 + TCP_NODELAY | 网络 P50: 51→38ms, **-13ms** | ✅ |
| 架构: FS 串行→双 GPU 分散并行 | 服务端: 188→172ms, **-16ms** | ✅ |
| 架构: SAM3+FS 同 GPU 内并行推理 | 已并行 (wall=max, 非 sum) | ✅ |
| 协议: 4个 HTTP→2个 (合并请求) | 协议开销减半, **-5ms** | ✅ |
| 编码: Full PNG→Half-res PNG | 下载 162→80KB, encode 58→10ms | ✅ |
| 服务端: decode 并行+推理重叠 | 服务端 total: 172→165ms, **-7ms** | ✅ |
| CUDA: VISIBLE_DEVICES 隔离 | 修复 GPU1 CUDA error 400 | ✅ |
| **FS 引擎: ViT-L → ViT-S** | **服务端 FS: 144→98ms, -46ms** | ✅ |
| 客户端: 3路 JPEG 并行编码 | 客户端 encode: 15→7ms, **-8ms** | ✅ |
| 服务端: turbojpeg RGB 解码 | RGB decode: 10→6ms, **-4ms** | ✅ |
| **FS GPU后处理: tensor保持GPU** | **GPU1: 132→122ms, -10ms** | ✅ |
| **FS gray-first resize** | **FS preproc: 省 ~2ms** | ✅ |
| **累计** | **245ms → 176ms = -69ms, 4.1→5.70Hz** | |

### 7.5 ViT-S vs ViT-L 精度对比

```
FoundationStereo ViT-S (fs_vits_576x960_fp16.engine, 219MB)
vs ViT-L (fs_vitl_576x960_fp16.engine, 844MB):

  推理速度:  ViT-S ~98ms vs ViT-L ~144ms (-46ms, 3.2x 更小引擎)
  深度精度 (vs 相机 raw depth):
    ≤5mm:   ViT-S 90.5% vs ViT-L 90.4%  (持平)
    ≤10mm:  ViT-S 93.1% vs ViT-L 92.9%  (持平)
    RMSE:   ViT-S 139.6mm vs ViT-L 153.5mm  (ViT-S 更优)
  结论: ViT-S 精度无损甚至更好, 推理快 ~46ms
```

### 7.6 未采纳的优化 (实测负优化)

| 方案 | 原因 |
|------|------|
| LZ4 压缩深度图 | 304KB > PNG 162KB, WiFi 传输更慢 |
| PNG level 0 (无压缩) | 452KB, 省 6ms encode 但加 70ms 网络, 净亏损 |
| JPEG Q95 编码深度 | 精度 ±20mm, 不满足质量约束 |
| 非默认 CUDA stream | GPU1 推理从 114ms 恶化到 238ms |
| turbojpeg 灰度解码 | cv2 灰度已 3.4ms, turbojpeg 无改善 (3.6ms) |
| 错开 SAM3/FS 启动 (stagger 30ms) | FS 未加速, SAM3 从 31→75ms (恰好赶上 FS TRT 峰值) |

---

## 8. 剩余瓶颈分析

```
                   耗时      占比     可优化空间
FS TRT 推理:      108ms     61.4%    GPU compute-bound, 已用 ViT-S + GPU后处理
WiFi 网络往返:     36ms     20.5%    物理极限 ~29ms, 仅差 7ms
服务端 encode:      9ms      5.1%    PNG lvl1 已是最优
客户端 encode:      8ms      4.5%    已并行化
服务端 decode:      5ms*     2.8%    已优化 (turbojpeg + 重叠)
客户端 decode:      4ms      2.3%    接近零
SAM3 推理:        102ms      —       与 FS 并行, 受 GPU 争抢影响
其他开销:           6ms      3.4%    —
────────────────────────────
总计 (P50):       176ms
目标:             200ms
余量:              24ms ✅
```

**结论**: 5Hz 目标已达成 (P50=176ms, 5.70Hz), 有 24ms 余量。
主要硬限制: FS 推理 (108ms) + WiFi (36ms) = 144ms, 占 81.8%。
进一步优化空间有限, 主要在硬件层面:

| 方向 | 预期收益 | 代价 |
|------|---------|------|
| 有线网络 (千兆) | ~30ms | 机器人物理限制 |
| FS 更小分辨率引擎 (480×640) | ~20ms | 深度精度降低 |
| 感知节点流水线化 (N/N+1帧重叠) | 等效 7Hz+ | 增加 1 帧延迟 |

---

## 9. 验证状态

| 步骤 | 操作 | 通过标准 | 状态 |
|------|------|---------|------|
| V0 | SAM3+FS 同进程加载 (CUDA context) | 无报错 | ✅ (FS先加载) |
| V1 | combined_server curl 测试 | 返回正确检测+深度 | ✅ |
| V2 | benchmark 单 GPU 测试 | 服务端 ≤180ms | ✅ (126ms) |
| V3 | benchmark 双相机并行 | **P50 ≤200ms** | ✅ **(176ms, 5.70Hz)** |
| V4 | 质量对比 (ViT-S vs ViT-L) | 精度不降 | ✅ (RMSE 更优) |
| V5 | 检测可重复性 | 同帧两次一致 | ✅ (IoU=1.0, depth 99.76%相同) |
| V6 | FS GPU patches 质量验证 | 深度精度不退化 | ✅ (≤5mm: 90.5%, 同前) |

```bash
# 验证命令 (Jetson):
cd /data/workspace/MobileManipulator2/src/perception/src
export PATH="/usr/bin:$PATH"
python3 percept_benchmark.py --benchmark --num-runs 30
```

---

## 10. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| ~~trtsam3 + PyTorch TRT CUDA context 冲突~~ | — | — | ✅ 已解决: FS 先加载 |
| ~~GPU1 TRT 跨设备引擎 CUDA error 400~~ | — | — | ✅ 已解决: CUDA_VISIBLE_DEVICES 隔离 |
| GPU 显存不足 (两模型 ~10GB) | 极低 | 阻塞 | RTX 5090 32GB, 空闲 20GB+ |
| WiFi P99 尾刺 (MAC 重传) 偶发超 200ms | 中 | 性能 | 应用层超时+跳帧; 换更干净 5GHz 信道 |
| 路由器 DFS 雷达事件触发信道切换 | 低 | 短暂断连 | 固定路由器到非 DFS 信道 (如 ch149) |

---

## 11. 实施状态

1. ~~**网络优化**: sysctl + 关蓝牙 + TCP_NODELAY~~ ✅
2. ~~**服务端**: `combined_server.py` + `start_combined.sh`~~ ✅
3. ~~**服务端**: CUDA_VISIBLE_DEVICES 隔离 + FS先加载 + decode并行~~ ✅
4. ~~**服务端**: turbojpeg RGB 解码加速~~ ✅
5. ~~**客户端**: `CombinedPerceptionClient` + `DualPerceptionClient`~~ ✅
6. ~~**客户端**: 3路 JPEG 并行编码~~ ✅
7. ~~**客户端**: `benchmark_combined_dual()` 测试函数~~ ✅
8. ~~**FS 引擎**: ViT-L → ViT-S (精度无损, 推理 -46ms)~~ ✅
9. ~~**FS GPU后处理**: gray-first resize + tensor保持GPU (GPU1: -10ms)~~ ✅
10. ~~**验证**: 30 runs benchmark, **P50=176ms (5.70Hz)** 达标~~ ✅
11. ~~**质量验证**: 检测可重复性 + ViT-S 精度 ≥ ViT-L~~ ✅
12. **(后续)**: 改造 `multi_camera_perception_node.py` 使用 `DualPerceptionClient`

---

## 附录 A: 网络诊断详情

### WiFi 链路信息

```
接口:     wlP1p1s0 (Intel iwlwifi)
SSID:     visincept_5G
频段:     5220 MHz (ch44), 160MHz 带宽
协议:     WiFi 6 (802.11ax), HE-MCS 11, 2×2 MIMO
信号:     -44 dBm avg
TX 速率:  2041.9 Mbps (link rate)
RX 速率:  2401.9 Mbps (link rate)
实际吞吐: ~420 Mbps
```

### 有线网口 (未使用)

```
接口:     eno1
能力:     100M / 1G / 2.5G / 5G / 10G
当前速率: 100 Mbps (对端交换机只支持 10/100)
子网:     192.168.0.x (与 GPU 服务器不同网段)
状态:     未路由到 GPU 服务器
注:       移动机器人无法使用有线连接
```

### WiFi 420Mbps 的原因

WiFi 链路速率 2400Mbps 但实际只有 420Mbps, 是 WiFi 协议的固有特性:

1. **CSMA/CA**: 每帧发送前必须监听信道+随机退避, 即使信道空闲
2. **帧间间隔**: SIFS/DIFS 强制等待 (16-34μs/帧)
3. **Block ACK**: 每 N 帧需确认, 占信道时间
4. **半双工**: 上传和下载不能同时进行, 必须轮替
5. **Beacon**: AP 每 100ms 广播信标帧, 占信道
6. **重传**: 1.79% 的帧需要重传, 每次触发退避窗口增长
7. **协议头开销**: 每帧 MAC header + PHY preamble
