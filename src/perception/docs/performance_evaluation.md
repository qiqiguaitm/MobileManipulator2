# 多传感器感知系统性能评测方案

> **版本**: v1.0
> **日期**: 2026-01-30
> **类型**: 性能评测文档
> **目的**: 验证设计文档中的性能声明

---

## 目录

1. [评测目标](#1-评测目标)
2. [测试环境](#2-测试环境)
3. [评测指标](#3-评测指标)
4. [测试场景](#4-测试场景)
5. [基准测试](#5-基准测试)
6. [对比测试](#6-对比测试)
7. [统计分析](#7-统计分析)
8. [预期结果](#8-预期结果)
9. [实施计划](#9-实施计划)

---

## 1. 评测目标

### 1.1 需要验证的性能声明

从设计文档中提取的待验证性能声明：

| 编号 | 声明 | 来源文档 | 状态 |
|------|------|---------|------|
| **P1** | 全局预处理：210ms → 50ms（4.2x加速） | gpu_acceleration.md | 🔴 待验证 |
| **P2** | GPU加速：110ms → 10ms（11x加速） | gpu_acceleration.md | 🔴 待验证 |
| **P3** | 系统级提升：425ms → 325ms（23%） | gpu_acceleration.md | 🔴 待验证 |
| **P4** | 帧率提升：2.35 FPS → 3.08 FPS（31%） | gpu_acceleration.md | 🔴 待验证 |
| **P5** | 匈牙利算法：比贪心提升4-6%精度 | multi_sensor_perception_design.md | 🔴 待验证 |
| **P6** | WLS融合：比单传感器提升25-40%精度 | multi_sensor_perception_design.md | 🔴 待验证 |
| **P7** | 深度测距GPU加速：60ms → 3ms（20x） | gpu_acceleration.md | 🔴 待验证 |
| **P8** | LiDAR聚类GPU加速：30ms → 3ms（10x） | gpu_acceleration.md | 🔴 待验证 |
| **P9** | WLS融合GPU加速：15ms → 2ms（7.5x） | gpu_acceleration.md | 🔴 待验证 |

### 1.2 评测原则

**Linus准则**：
```text
"Talk is cheap. Show me the code... and the data."
```

1. ✅ **可复现性**：提供完整的测试脚本和数据集
2. ✅ **统计显著性**：每个测试至少30次，报告均值、标准差、P95
3. ✅ **控制变量**：每次只改变一个优化项
4. ✅ **真实数据**：使用实际传感器数据，不是模拟
5. ✅ **失败案例**：报告不适用场景和失败案例
6. ✅ **固定随机种子**：消除随机性影响

---

## 2. 测试环境

### 2.1 硬件配置

**基准测试环境**：
```yaml
机器人: AGILEX Mobile Manipulator
CPU: ARM Cortex-A78AE (8核)
内存: 16GB
GPU: NVIDIA Jetson AGX Orin (可选，用于GPU测试)
存储: NVMe SSD 256GB

传感器:
  - Chassis Camera: RealSense D435
  - Top Camera: RealSense D435
  - LiDAR: RSHELIOS 16P
```

**对比测试环境（GPU）**：
- 相同机器人，启用GPU加速模块
- 安装CuPy、PyTorch、Open3D CUDA

### 2.2 软件环境

```yaml
OS: Ubuntu 20.04
ROS: Noetic
Python: 3.8
主要依赖:
  - numpy==1.24.0
  - scipy==1.10.0
  - opencv-python==4.7.0
  - scikit-learn==1.2.0
  - rospy
  - sensor_msgs

GPU依赖（可选）:
  - cupy-cuda11x==12.0.0
  - torch==2.0.0
  - open3d==0.17.0
```

### 2.3 数据集

**采集策略**：
1. **固定场景录制**：使用`rosbag record`记录真实传感器数据
2. **场景多样性**：覆盖简单/中等/复杂场景
3. **数据量**：每个场景至少100帧（20秒@5Hz）

**场景定义**（见第4节）

---

## 3. 评测指标

### 3.1 延迟指标

| 指标 | 定义 | 单位 | 计算方法 |
|------|------|------|---------|
| **E2E延迟** | 从传感器数据到输出结果 | ms | `t_output - t_sensor` |
| **检测延迟** | DINO-X推理时间 | ms | `t_detection_end - t_detection_start` |
| **测量延迟** | 3D测距时间 | ms | `t_measure_end - t_measure_start` |
| **匹配延迟** | 双相机去重时间 | ms | `t_match_end - t_match_start` |
| **融合延迟** | 传感器融合时间 | ms | `t_fusion_end - t_fusion_start` |
| **跟踪延迟** | 多目标跟踪时间 | ms | `t_track_end - t_track_start` |

**统计量**：
- Mean（均值）
- Std（标准差）
- P50（中位数）
- P95（95百分位）
- P99（99百分位）

### 3.2 精度指标

#### 3.2.1 测距精度

| 指标 | 定义 | 单位 | 计算方法 |
|------|------|------|---------|
| **绝对误差（AE）** | 测量值与真值之差 | cm | `\|measured - ground_truth\|` |
| **均方根误差（RMSE）** | 误差的平方根 | cm | `sqrt(mean(AE^2))` |
| **平均误差（MAE）** | 绝对误差的均值 | cm | `mean(AE)` |
| **精度@阈值** | 误差<T的比例 | % | `count(AE<T) / total` |

**阈值定义**：
- 高精度：AE < 3cm
- 中精度：AE < 5cm
- 低精度：AE < 10cm

#### 3.2.2 匹配精度

| 指标 | 定义 | 单位 | 计算方法 |
|------|------|------|---------|
| **精确匹配率** | 正确匹配的比例 | % | `TP / (TP + FP + FN)` |
| **误匹配率** | 错误匹配的比例 | % | `FP / (TP + FP)` |
| **漏匹配率** | 漏掉的真匹配比例 | % | `FN / (TP + FN)` |

**Ground Truth标注**：
- 方法1：手工标注（小数据集）
- 方法2：使用高置信度检测（confidence > 0.9）

#### 3.2.3 跟踪精度

| 指标 | 定义 | 单位 | 参考 |
|------|------|------|------|
| **MOTA** | 多目标跟踪准确度 | % | MOT Challenge |
| **MOTP** | 多目标跟踪精度 | cm | MOT Challenge |
| **ID切换率** | 轨迹ID错误切换次数 | 次/帧 | MOT Challenge |
| **漏跟踪率** | 真实物体未跟踪比例 | % | 自定义 |

### 3.3 吞吐量指标

| 指标 | 定义 | 单位 | 计算方法 |
|------|------|------|---------|
| **帧率（FPS）** | 每秒处理帧数 | Hz | `1 / mean(frame_time)` |
| **最大物体数** | 在<500ms延迟下支持的物体数 | 个 | 压力测试 |

### 3.4 资源占用指标

| 指标 | 单位 | 监控工具 |
|------|------|---------|
| **CPU占用率** | % | `top` / `htop` |
| **内存占用** | MB | `free -m` |
| **GPU占用率** | % | `nvidia-smi` |
| **GPU显存** | MB | `nvidia-smi` |

---

## 4. 测试场景

### 4.1 场景分类

| 场景编号 | 场景名称 | 物体数量 | 距离范围 | 复杂度 | 用途 |
|---------|---------|---------|---------|-------|------|
| **S1** | 简单场景 | 5-8个 | 0.5-1.5m | 低 | 快速验证 |
| **S2** | 典型场景 | 15-20个 | 0.5-3.0m | 中 | 主要评测 |
| **S3** | 复杂场景 | 25-30个 | 0.3-3.5m | 高 | 压力测试 |
| **S4** | 极端场景 | 35-40个 | 0.3-4.0m | 极高 | 边界测试 |

### 4.2 场景详细定义

#### S1: 简单场景（快速验证）

```yaml
场景描述: 地面散落的纸箱和瓶子
物体列表:
  - 3个纸箱（30×30×30cm）
  - 2个瓶子（高20cm，直径8cm）
  - 1个杯子（高10cm）
  - 2个其他小物体
物体分布:
  - 距离: 0.5m - 1.5m
  - 高度: 0cm - 40cm（地面到桌面）
  - 间隔: >30cm（无遮挡）
特点:
  - ✅ 无遮挡
  - ✅ LiDAR覆盖良好
  - ✅ 双相机视野都覆盖
用途:
  - 快速验证算法正确性
  - 调试用
```

#### S2: 典型场景（主要评测）

```yaml
场景描述: 仓库环境，货架+地面混合
物体列表:
  - 10个纸箱（大小混合）
  - 5个瓶子
  - 3个托盘
  - 2个桶
  - 5个其他物体
物体分布:
  - 距离: 0.5m - 3.0m
  - 高度: 0cm - 80cm
  - 间隔: 10-50cm（有轻微遮挡）
特点:
  - ⚠️ 部分遮挡（10-20%）
  - ✅ LiDAR覆盖80%
  - ⚠️ 部分物体只有单相机看到
用途:
  - 主要性能评测场景
  - 与设计目标对齐（20个物体）
```

#### S3: 复杂场景（压力测试）

```yaml
场景描述: 密集堆放，多层货架
物体列表:
  - 25-30个混合物体
物体分布:
  - 距离: 0.3m - 3.5m
  - 高度: 0cm - 120cm
  - 间隔: <10cm（严重遮挡）
特点:
  - ❌ 严重遮挡（30-40%）
  - ⚠️ LiDAR只覆盖50%
  - ⚠️ 双相机视角差异大
用途:
  - 测试算法鲁棒性
  - 发现边界问题
```

#### S4: 极端场景（边界测试）

```yaml
场景描述: 最大物体数量测试
物体列表:
  - 35-40个物体
物体分布:
  - 极端密集
特点:
  - ❌ 极端遮挡
  - ❌ 传感器视野饱和
用途:
  - 测试算法上限
  - 验证失败模式
```

### 4.3 Ground Truth采集

**方法1：手工标注（推荐用于S1、S2）**

```python
# 使用RViz手工标注物体真实位置
# 工具：scripts/benchmark/annotate_ground_truth.py

步骤：
1. 播放rosbag
2. 在RViz中使用3D marker标注物体中心
3. 保存为CSV文件

格式：
timestamp,object_id,x,y,z,category
1234567890.123,1,0.5,0.2,0.15,box
1234567890.123,2,1.2,0.0,0.30,bottle
```

**方法2：高置信度检测（用于大规模测试）**

```python
# 使用confidence > 0.9的检测作为伪Ground Truth
# 适用于对比测试（相对性能）
```

---

## 5. 基准测试

### 5.1 测试矩阵

| 测试编号 | 测试名称 | 优化项 | 场景 | 迭代次数 |
|---------|---------|--------|------|---------|
| **B1** | CPU基准-逐物体测距 | 无 | S1, S2 | 30 |
| **B2** | CPU基准-全局预处理 | 全局预处理 | S1, S2 | 30 |
| **B3** | GPU加速-深度测距 | GPU深度 | S1, S2 | 30 |
| **B4** | GPU加速-LiDAR聚类 | GPU LiDAR | S1, S2 | 30 |
| **B5** | GPU加速-完整Pipeline | GPU全部 | S1, S2 | 30 |
| **B6** | 贪心匹配 | 贪心算法 | S2 | 100 |
| **B7** | 匈牙利匹配 | 匈牙利算法 | S2 | 100 |
| **B8** | 单传感器（LiDAR） | 无融合 | S2 | 30 |
| **B9** | 单传感器（Chassis相机） | 无融合 | S2 | 30 |
| **B10** | WLS融合 | WLS融合 | S2 | 30 |

### 5.2 测试脚本框架

```python
#!/usr/bin/env python3
"""
性能基准测试框架
scripts/benchmark/run_benchmark.py
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List
import rospy
import rosbag

class PerformanceBenchmark:
    """性能评测基类"""

    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.results = []

    def run_test(self, test_name: str, test_func, num_iterations: int = 30):
        """
        运行单个测试

        参数:
            test_name: 测试名称
            test_func: 测试函数
            num_iterations: 迭代次数
        """
        print(f"\n{'='*60}")
        print(f"开始测试: {test_name}")
        print(f"迭代次数: {num_iterations}")
        print(f"{'='*60}\n")

        latencies = []

        for i in range(num_iterations):
            start_time = time.time()

            # 执行测试
            result = test_func()

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            latencies.append(latency_ms)

            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{num_iterations}, "
                      f"最近延迟: {latency_ms:.2f}ms")

        # 统计分析
        stats = self.compute_statistics(latencies)
        stats['test_name'] = test_name
        stats['num_iterations'] = num_iterations

        self.results.append(stats)

        self.print_stats(stats)

        return stats

    def compute_statistics(self, latencies: List[float]) -> Dict:
        """计算统计量"""
        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'median': np.median(latencies),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'min': np.min(latencies),
            'max': np.max(latencies),
        }

    def print_stats(self, stats: Dict):
        """打印统计结果"""
        print(f"\n结果统计:")
        print(f"  均值:   {stats['mean']:>8.2f} ms")
        print(f"  标准差: {stats['std']:>8.2f} ms")
        print(f"  中位数: {stats['median']:>8.2f} ms")
        print(f"  P95:    {stats['p95']:>8.2f} ms")
        print(f"  P99:    {stats['p99']:>8.2f} ms")
        print(f"  最小:   {stats['min']:>8.2f} ms")
        print(f"  最大:   {stats['max']:>8.2f} ms")

    def save_results(self, output_path: str):
        """保存结果到CSV"""
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        print(f"\n✅ 结果已保存: {output_path}")

    def compare(self, baseline_name: str, optimized_name: str):
        """对比两个测试的性能"""
        baseline = next(r for r in self.results if r['test_name'] == baseline_name)
        optimized = next(r for r in self.results if r['test_name'] == optimized_name)

        speedup = baseline['mean'] / optimized['mean']
        improvement_pct = (1 - optimized['mean'] / baseline['mean']) * 100

        print(f"\n{'='*60}")
        print(f"性能对比: {baseline_name} vs {optimized_name}")
        print(f"{'='*60}")
        print(f"基准延迟:   {baseline['mean']:.2f} ms")
        print(f"优化延迟:   {optimized['mean']:.2f} ms")
        print(f"加速比:     {speedup:.2f}x")
        print(f"性能提升:   {improvement_pct:.1f}%")
        print(f"{'='*60}\n")

        return {
            'baseline_name': baseline_name,
            'optimized_name': optimized_name,
            'baseline_latency': baseline['mean'],
            'optimized_latency': optimized['mean'],
            'speedup': speedup,
            'improvement_pct': improvement_pct,
        }


class MeasurementBenchmark(PerformanceBenchmark):
    """测距模块性能评测"""

    def __init__(self, config_path: str, rosbag_path: str):
        super().__init__(config_path)
        self.bag = rosbag.Bag(rosbag_path)
        self.load_data()

    def load_data(self):
        """从rosbag加载数据"""
        # 提取一帧数据用于测试
        # TODO: 实现完整的数据加载
        pass

    def test_cpu_per_object(self) -> Dict:
        """测试：CPU逐物体测距"""
        from perception.measurement import CPUMeasurerPerObject

        measurer = CPUMeasurerPerObject()
        result = measurer.measure_all(
            detections=self.detections,
            lidar_points=self.lidar_points,
            depth_map=self.depth_map
        )
        return result

    def test_cpu_global_preprocessing(self) -> Dict:
        """测试：CPU全局预处理"""
        from perception.measurement import CPUMeasurerGlobalPreprocess

        measurer = CPUMeasurerGlobalPreprocess()
        result = measurer.measure_all(
            detections=self.detections,
            lidar_points=self.lidar_points,
            depth_map=self.depth_map
        )
        return result

    def test_gpu_depth(self) -> Dict:
        """测试：GPU深度测距"""
        from perception.measurement_gpu import GPUDepthMeasurer

        measurer = GPUDepthMeasurer()
        result = measurer.measure_batch(
            detections=self.detections,
            depth_map=self.depth_map
        )
        return result

    def test_gpu_lidar(self) -> Dict:
        """测试：GPU LiDAR聚类"""
        from perception.measurement_gpu import GPULiDARClusterer

        clusterer = GPULiDARClusterer()
        result = clusterer.cluster_global(self.lidar_points)
        return result

    def test_gpu_full(self) -> Dict:
        """测试：GPU完整Pipeline"""
        from perception.measurement_gpu import GPUMultiSensorMeasurer

        measurer = GPUMultiSensorMeasurer()
        result = measurer.measure_all_gpu(
            detections=self.detections,
            lidar_points=self.lidar_points,
            depth_map=self.depth_map
        )
        return result


class MatchingBenchmark(PerformanceBenchmark):
    """匹配模块精度评测"""

    def __init__(self, config_path: str, ground_truth_path: str):
        super().__init__(config_path)
        self.ground_truth = pd.read_csv(ground_truth_path)

    def test_greedy_matching(self) -> Dict:
        """测试：贪心匹配"""
        from perception.matching import GreedyMatcher

        matcher = GreedyMatcher()
        matches = matcher.match(
            detections_chassis=self.dets_chassis,
            detections_top=self.dets_top
        )

        # 计算精度
        accuracy = self.evaluate_matches(matches, self.ground_truth)
        return accuracy

    def test_hungarian_matching(self) -> Dict:
        """测试：匈牙利匹配"""
        from perception.matching import HungarianMatcher

        matcher = HungarianMatcher()
        matches = matcher.match(
            detections_chassis=self.dets_chassis,
            detections_top=self.dets_top
        )

        accuracy = self.evaluate_matches(matches, self.ground_truth)
        return accuracy

    def evaluate_matches(self, matches, ground_truth) -> Dict:
        """评估匹配精度"""
        # TODO: 实现精度评估
        # 返回: TP, FP, FN, precision, recall, F1
        pass


class FusionBenchmark(PerformanceBenchmark):
    """融合模块精度评测"""

    def __init__(self, config_path: str, ground_truth_path: str):
        super().__init__(config_path)
        self.ground_truth = pd.read_csv(ground_truth_path)

    def test_lidar_only(self) -> Dict:
        """测试：仅LiDAR"""
        from perception.fusion import LiDAROnlyMeasurement

        measurer = LiDAROnlyMeasurement()
        positions = measurer.measure(self.detections, self.lidar_points)

        errors = self.compute_errors(positions, self.ground_truth)
        return errors

    def test_depth_only(self) -> Dict:
        """测试：仅深度相机"""
        from perception.fusion import DepthOnlyMeasurement

        measurer = DepthOnlyMeasurement()
        positions = measurer.measure(self.detections, self.depth_map)

        errors = self.compute_errors(positions, self.ground_truth)
        return errors

    def test_wls_fusion(self) -> Dict:
        """测试：WLS融合"""
        from perception.fusion import WLSFusion

        fuser = WLSFusion()
        positions = fuser.fuse(
            lidar_measurements=self.lidar_meas,
            depth_measurements=self.depth_meas
        )

        errors = self.compute_errors(positions, self.ground_truth)
        return errors

    def compute_errors(self, positions, ground_truth) -> Dict:
        """计算测距误差"""
        # TODO: 实现误差计算
        # 返回: MAE, RMSE, accuracy@3cm, accuracy@5cm
        pass


# ========== 主测试脚本 ==========

def main():
    """主测试流程"""

    print("="*60)
    print("多传感器感知系统性能评测")
    print("="*60)

    # ========== 测试1: 测距延迟 ==========
    print("\n\n【测试1：测距延迟对比】\n")

    bench_measure = MeasurementBenchmark(
        config_path='config/benchmark.yaml',
        rosbag_path='data/scenario_S2.bag'
    )

    # B1: CPU逐物体
    bench_measure.run_test(
        'B1_CPU_PerObject',
        bench_measure.test_cpu_per_object,
        num_iterations=30
    )

    # B2: CPU全局预处理
    bench_measure.run_test(
        'B2_CPU_GlobalPreprocess',
        bench_measure.test_cpu_global_preprocessing,
        num_iterations=30
    )

    # B3-B5: GPU测试
    bench_measure.run_test(
        'B3_GPU_Depth',
        bench_measure.test_gpu_depth,
        num_iterations=30
    )

    bench_measure.run_test(
        'B4_GPU_LiDAR',
        bench_measure.test_gpu_lidar,
        num_iterations=30
    )

    bench_measure.run_test(
        'B5_GPU_Full',
        bench_measure.test_gpu_full,
        num_iterations=30
    )

    # 对比分析
    print("\n\n【对比分析】\n")

    comp1 = bench_measure.compare('B1_CPU_PerObject', 'B2_CPU_GlobalPreprocess')
    comp2 = bench_measure.compare('B1_CPU_PerObject', 'B5_GPU_Full')

    bench_measure.save_results('results/measurement_latency.csv')

    # ========== 测试2: 匹配精度 ==========
    print("\n\n【测试2：匹配精度对比】\n")

    bench_match = MatchingBenchmark(
        config_path='config/benchmark.yaml',
        ground_truth_path='data/scenario_S2_gt.csv'
    )

    # B6: 贪心匹配
    bench_match.run_test(
        'B6_Greedy_Matching',
        bench_match.test_greedy_matching,
        num_iterations=100
    )

    # B7: 匈牙利匹配
    bench_match.run_test(
        'B7_Hungarian_Matching',
        bench_match.test_hungarian_matching,
        num_iterations=100
    )

    comp3 = bench_match.compare('B6_Greedy_Matching', 'B7_Hungarian_Matching')

    bench_match.save_results('results/matching_accuracy.csv')

    # ========== 测试3: 融合精度 ==========
    print("\n\n【测试3：融合精度对比】\n")

    bench_fusion = FusionBenchmark(
        config_path='config/benchmark.yaml',
        ground_truth_path='data/scenario_S2_gt.csv'
    )

    # B8-B10: 融合对比
    bench_fusion.run_test(
        'B8_LiDAR_Only',
        bench_fusion.test_lidar_only,
        num_iterations=30
    )

    bench_fusion.run_test(
        'B9_Depth_Only',
        bench_fusion.test_depth_only,
        num_iterations=30
    )

    bench_fusion.run_test(
        'B10_WLS_Fusion',
        bench_fusion.test_wls_fusion,
        num_iterations=30
    )

    comp4 = bench_fusion.compare('B8_LiDAR_Only', 'B10_WLS_Fusion')

    bench_fusion.save_results('results/fusion_accuracy.csv')

    # ========== 生成报告 ==========
    print("\n\n【生成评测报告】\n")

    generate_report([comp1, comp2, comp3, comp4])

    print("\n✅ 评测完成！")


def generate_report(comparisons: List[Dict]):
    """生成Markdown评测报告"""

    report = """# 性能评测报告

## 执行摘要

| 声明 | 预期结果 | 实际结果 | 状态 |
|------|---------|---------|------|
"""

    for comp in comparisons:
        # TODO: 填充报告内容
        pass

    with open('results/benchmark_report.md', 'w') as f:
        f.write(report)

    print("📊 报告已生成: results/benchmark_report.md")


if __name__ == '__main__':
    main()
```

---

## 6. 对比测试

### 6.1 测试组合

#### 组合1：测距延迟优化链

```
CPU逐物体 (210ms)
    ↓
CPU全局预处理 (50ms)  ← 声称4.2x加速
    ↓
GPU全部 (10ms)        ← 声称11x加速（相对CPU逐物体）
```

**验证方法**：
1. 使用相同输入数据（rosbag固定帧）
2. 逐步启用优化
3. 记录每步延迟
4. 计算实际加速比

**成功标准**：
- CPU全局预处理：3x-5x加速（允许±20%误差）
- GPU加速：9x-13x加速（允许±20%误差）

#### 组合2：匹配精度对比

```
贪心匹配 (94.2%)
    ↓
匈牙利匹配 (98.8%)    ← 声称提升4.6%
```

**验证方法**：
1. 使用S2场景（20个物体）
2. 手工标注Ground Truth
3. 运行100次（消除随机性）
4. 计算精确匹配率

**成功标准**：
- 匈牙利算法：比贪心提升3%-6%（允许±1%误差）

#### 组合3：融合精度对比

```
单传感器（LiDAR）
单传感器（相机）
    ↓
WLS融合              ← 声称精度提升25-40%
```

**验证方法**：
1. 使用S2场景
2. 标注Ground Truth（使用外部测量设备）
3. 计算MAE、RMSE
4. 对比单传感器 vs 融合

**成功标准**：
- 融合精度：比最优单传感器提升20%-45%

### 6.2 控制变量

**严格控制**：
- ✅ 使用相同的rosbag数据
- ✅ 使用相同的随机种子
- ✅ 使用相同的检测结果（固定检测输出）
- ✅ 关闭其他ROS节点（隔离环境）

**记录环境**：
- CPU频率（关闭动态调频）
- 温度（避免过热降频）
- 后台进程（最小化干扰）

---

## 7. 统计分析

### 7.1 显著性检验

**问题**：观察到的性能差异是真实的，还是随机波动？

**方法**：配对t检验

```python
from scipy import stats

def test_significance(baseline_latencies, optimized_latencies):
    """
    配对t检验：检验优化是否显著

    H0（零假设）：优化无效，均值相等
    H1（备择假设）：优化有效，均值不同
    """
    t_statistic, p_value = stats.ttest_rel(baseline_latencies, optimized_latencies)

    alpha = 0.05  # 显著性水平

    if p_value < alpha:
        print(f"✅ 显著差异 (p={p_value:.4f} < {alpha})")
        print(f"   拒绝零假设，优化有效")
    else:
        print(f"❌ 无显著差异 (p={p_value:.4f} >= {alpha})")
        print(f"   不能拒绝零假设，优化可能无效")

    return t_statistic, p_value


# 示例使用
baseline = np.random.normal(210, 10, 30)  # 210ms均值，10ms标准差，30次测试
optimized = np.random.normal(50, 5, 30)   # 50ms均值

test_significance(baseline, optimized)
```

**判断标准**：
- p < 0.05：显著（95%置信度）
- p < 0.01：非常显著（99%置信度）

### 7.2 效应量分析

**问题**：即使有显著性，效果有多大？

**方法**：Cohen's d

```python
def compute_cohens_d(baseline, optimized):
    """
    计算Cohen's d效应量

    d = (mean1 - mean2) / pooled_std

    解释：
    - d < 0.2: 小效应
    - 0.2 <= d < 0.5: 中效应
    - 0.5 <= d < 0.8: 大效应
    - d >= 0.8: 非常大效应
    """
    mean_diff = np.mean(baseline) - np.mean(optimized)

    pooled_std = np.sqrt(
        (np.std(baseline, ddof=1)**2 + np.std(optimized, ddof=1)**2) / 2
    )

    d = mean_diff / pooled_std

    if d >= 0.8:
        interpretation = "非常大效应"
    elif d >= 0.5:
        interpretation = "大效应"
    elif d >= 0.2:
        interpretation = "中效应"
    else:
        interpretation = "小效应"

    print(f"Cohen's d = {d:.2f} ({interpretation})")

    return d
```

### 7.3 置信区间

**目的**：量化不确定性

```python
def compute_confidence_interval(data, confidence=0.95):
    """
    计算均值的置信区间

    示例：[48.5, 51.5]ms意味着真实均值有95%概率在这个区间
    """
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)  # 标准误差

    # t分布临界值（自由度=n-1）
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)

    margin_of_error = t_critical * std_err

    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error

    print(f"{confidence*100}% 置信区间: [{ci_lower:.2f}, {ci_upper:.2f}]")

    return ci_lower, ci_upper
```

---

## 8. 预期结果

### 8.1 延迟优化验证

| 优化项 | 文档声称 | 预期验证结果 | 可接受范围 |
|--------|---------|-------------|-----------|
| 全局预处理 | 4.2x (210→50ms) | 3.5x - 5.0x | ±20% |
| GPU深度测距 | 20x (60→3ms) | 16x - 24x | ±20% |
| GPU LiDAR聚类 | 10x (30→3ms) | 8x - 12x | ±20% |
| GPU WLS融合 | 7.5x (15→2ms) | 6x - 9x | ±20% |
| GPU完整Pipeline | 11x (110→10ms) | 9x - 13x | ±20% |
| 系统级提升 | 23% (425→325ms) | 18% - 28% | ±5% |

**说明**：
- 允许±20%误差：因为硬件差异、测试条件不同
- 系统级提升误差更小：因为检测模块不变，测量模块占比固定

### 8.2 精度优化验证

| 优化项 | 文档声称 | 预期验证结果 | 可接受范围 |
|--------|---------|-------------|-----------|
| 匈牙利 vs 贪心 | +4.6% (94.2%→98.8%) | +3% - +6% | ±1% |
| WLS融合（场景1） | +25%精度 | +20% - +30% | ±5% |
| WLS融合（场景2） | +30%精度 | +25% - +35% | ±5% |

### 8.3 失败案例预期

**预期会失败的场景**（需要在报告中说明）：

1. **极端物体数（S4场景）**
   - 预期：延迟超过500ms，精度下降
   - 原因：算法设计目标是20个物体

2. **LiDAR盲区物体**
   - 预期：无法使用LiDAR融合
   - 解决：依赖深度相机

3. **纯黑/纯白/透明物体**
   - 预期：深度相机失效
   - 解决：依赖LiDAR（如果有点云）

---

## 9. 实施计划

### 9.1 阶段划分

**阶段0：环境准备（1天）**
```bash
任务：
- [ ] 搭建测试环境
- [ ] 安装依赖（CuPy、PyTorch、Open3D）
- [ ] 验证GPU环境

交付物：
- test_gpu_setup.py运行成功
```

**阶段1：数据采集（2天）**
```bash
任务：
- [ ] 布置测试场景（S1, S2, S3）
- [ ] 录制rosbag数据
- [ ] 手工标注Ground Truth（S1, S2）

交付物：
- data/scenario_S1.bag
- data/scenario_S2.bag
- data/scenario_S3.bag
- data/scenario_S2_gt.csv
```

**阶段2：基准测试（3天）**
```bash
任务：
- [ ] 实现测试框架（scripts/benchmark/run_benchmark.py）
- [ ] 运行B1-B5（测距延迟）
- [ ] 运行B6-B7（匹配精度）
- [ ] 运行B8-B10（融合精度）

交付物：
- results/measurement_latency.csv
- results/matching_accuracy.csv
- results/fusion_accuracy.csv
```

**阶段3：统计分析（1天）**
```bash
任务：
- [ ] 显著性检验
- [ ] 效应量分析
- [ ] 生成图表

交付物：
- results/statistical_analysis.csv
- results/plots/*.png
```

**阶段4：报告生成（1天）**
```bash
任务：
- [ ] 生成Markdown报告
- [ ] 对比预期 vs 实际
- [ ] 标注失败案例

交付物：
- results/benchmark_report.md
```

### 9.2 时间表

```
Week 1:
  Mon: 阶段0（环境准备）
  Tue-Wed: 阶段1（数据采集）
  Thu-Sat: 阶段2（基准测试）

Week 2:
  Sun: 阶段3（统计分析）
  Mon: 阶段4（报告生成）
  Tue: 缓冲时间

总计：~7天
```

### 9.3 成功标准

**必须达到（Blocking）**：
- ✅ 所有9个性能声明都有对应的测试
- ✅ 至少80%的声明在可接受范围内
- ✅ 显著性检验p < 0.05
- ✅ 报告包含失败案例分析

**期望达到（Nice to have）**：
- ✅ 90%的声明在±10%误差内
- ✅ 效应量Cohen's d > 0.8
- ✅ 压力测试（S3场景）

---

## 10. 附录

### 10.1 目录结构

```
src/perception/
├── docs/
│   ├── performance_evaluation.md         # 本文档
│   ├── multi_sensor_perception_design.md # 设计文档
│   └── gpu_acceleration.md               # GPU加速文档
│
├── scripts/
│   └── benchmark/
│       ├── run_benchmark.py              # 主测试脚本
│       ├── measurement_benchmark.py      # 测距测试
│       ├── matching_benchmark.py         # 匹配测试
│       ├── fusion_benchmark.py           # 融合测试
│       ├── annotate_ground_truth.py      # 标注工具
│       └── visualize_results.py          # 结果可视化
│
├── data/
│   ├── scenario_S1.bag                   # 简单场景数据
│   ├── scenario_S2.bag                   # 典型场景数据
│   ├── scenario_S3.bag                   # 复杂场景数据
│   └── scenario_S2_gt.csv                # Ground Truth标注
│
├── results/
│   ├── measurement_latency.csv           # 测距延迟结果
│   ├── matching_accuracy.csv             # 匹配精度结果
│   ├── fusion_accuracy.csv               # 融合精度结果
│   ├── statistical_analysis.csv          # 统计分析
│   ├── benchmark_report.md               # 评测报告
│   └── plots/                            # 结果图表
│       ├── latency_comparison.png
│       ├── accuracy_comparison.png
│       └── fusion_improvement.png
│
└── config/
    └── benchmark.yaml                    # 测试配置
```

### 10.2 配置文件示例

```yaml
# config/benchmark.yaml

benchmark:
  # 数据路径
  data_dir: 'data/'
  results_dir: 'results/'

  # 测试参数
  num_iterations: 30
  random_seed: 42

  # 场景配置
  scenarios:
    S1:
      name: '简单场景'
      bag_file: 'scenario_S1.bag'
      num_objects: 8

    S2:
      name: '典型场景'
      bag_file: 'scenario_S2.bag'
      num_objects: 20
      ground_truth: 'scenario_S2_gt.csv'

    S3:
      name: '复杂场景'
      bag_file: 'scenario_S3.bag'
      num_objects: 30

  # 性能阈值
  thresholds:
    max_e2e_latency_ms: 400
    max_measurement_latency_ms: 50
    min_matching_accuracy: 0.95
    max_position_error_cm: 5.0

  # GPU配置
  gpu:
    enabled: true
    device_id: 0
    batch_size: 20
```

### 10.3 报告模板

```markdown
# 多传感器感知系统性能评测报告

## 执行摘要

- 测试日期: 2026-01-XX
- 测试环境: AGILEX Mobile Manipulator
- 测试场景: S1（简单）, S2（典型）, S3（复杂）
- 测试迭代: 30次/测试

## 主要发现

### ✅ 验证通过的声明

1. **全局预处理优化**
   - 文档声称: 4.2x加速 (210ms → 50ms)
   - 实际测试: 3.8x加速 (212ms → 56ms)
   - 状态: ✅ 通过（在可接受范围内）
   - 显著性: p=0.001 < 0.05
   - 效应量: Cohen's d = 2.3（非常大效应）

...

### ⚠️ 部分验证的声明

...

### ❌ 未通过验证的声明

...

## 详细结果

### 1. 测距延迟对比

| 测试 | 均值 | 标准差 | P95 | 加速比 |
|------|------|--------|-----|--------|
| ...  | ...  | ...    | ... | ...    |

### 2. 匹配精度对比

...

### 3. 融合精度对比

...

## 失败案例分析

...

## 结论与建议

...
```

---

**文档结束**

下一步：开始实施阶段0（环境准备）
