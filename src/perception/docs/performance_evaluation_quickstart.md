# 性能评测 - 快速开始指南

> **状态**: ✅ 评测框架已就绪
> **创建日期**: 2026-01-30
> **相关文档**: [完整评测方案](./performance_evaluation.md)

---

## 已完成工作

### 1. 评测框架 ✅

创建了完整的性能基准测试框架：

```
scripts/benchmark/
├── performance_benchmark.py      # 核心评测框架
├── verify_framework.py           # 框架验证脚本
├── test_measurement_performance.py  # 测距模块评测
└── README.md                     # 使用指南
```

**核心功能**：
- ✅ 自动化性能测试（支持预热、多次迭代）
- ✅ 统计分析（均值、标准差、P95/P99）
- ✅ 对比分析（加速比、性能提升百分比）
- ✅ 显著性检验（t检验、Cohen's d效应量）
- ✅ 结果保存（CSV格式）
- ✅ 报告生成（Markdown摘要）

### 2. 评测方案 ✅

创建了完整的评测方案文档：
- **docs/performance_evaluation.md** (完整方案，2800+行)
  - 9个待验证性能声明
  - 4个测试场景定义（S1-S4）
  - 详细的测试指标和方法
  - 统计分析方法
  - 预期结果和成功标准

### 3. 测距模块评测 ✅

创建了测距模块性能测试：
- **scripts/benchmark/test_measurement_performance.py**
  - 模拟数据生成器
  - CPU逐物体基准实现
  - CPU全局预处理优化实现
  - GPU完整Pipeline实现（需要CuPy）

**可验证的声明**：
- P1: 全局预处理 4.2x加速 (210ms → 50ms)
- P2: GPU加速 11x加速 (110ms → 10ms)
- P7: GPU深度测距 20x加速 (60ms → 3ms)
- P8: GPU LiDAR聚类 10x加速 (30ms → 3ms)

### 4. 框架验证 ✅

```bash
$ cd /home/agilex/MobileManipulator/scripts/benchmark
$ python3 verify_framework.py

✅ 所有验证测试通过！
```

验证内容：
- ✅ 快速基准测试
- ✅ 对比测试
- ✅ 完整评测流程
- ✅ 统计显著性检验

---

## 立即使用

### 步骤1: 验证框架

```bash
cd /home/agilex/MobileManipulator/scripts/benchmark
python3 verify_framework.py
```

预期输出：
```
✅ 所有验证测试通过！
```

### 步骤2: 运行测距模块评测（模拟数据）

```bash
python3 test_measurement_performance.py
```

这将：
1. 生成模拟数据（20个物体）
2. 测试CPU逐物体方法（基准）
3. 测试CPU全局预处理（优化）
4. 测试GPU Pipeline（如果GPU可用）
5. 计算加速比并验证是否符合预期
6. 保存结果到 `results/` 目录

预期输出：
```
======================================================================
性能对比: CPU_PerObject_Baseline vs CPU_GlobalPreprocess
======================================================================
基准延迟:   XXX.XX ms
优化延迟:   XXX.XX ms
加速比:         X.XXx
期望加速比: 4.20x
验证结果:   ✅ 通过（在可接受范围内）
```

### 步骤3: 查看结果

```bash
# 查看原始数据
cat results/measurement_latency.csv

# 查看对比结果
cat results/measurement_comparisons.csv

# 查看摘要报告
cat results/measurement_summary.md
```

---

## 待完成工作

### 使用真实数据进行评测

**当前限制**：`test_measurement_performance.py` 使用模拟数据

**需要真实数据的原因**：
1. **精度评测**：需要Ground Truth来验证测距精度
2. **匹配评测**：需要实际双相机数据验证匹配精度
3. **融合评测**：需要真实传感器数据验证WLS融合效果

**数据采集计划**（参考 performance_evaluation.md 第4节）：

#### 场景S1（简单场景）
```bash
# 1. 布置场景：地面放置5-8个物体
#    - 距离: 0.5-1.5m
#    - 无遮挡
#    - 间隔>30cm

# 2. 录制rosbag
rosbag record \
  /chassis_camera/color/image_raw \
  /chassis_camera/aligned_depth_to_color/image_raw \
  /top_camera/color/image_raw \
  /top_camera/aligned_depth_to_color/image_raw \
  /rslidar_points \
  -O data/scenario_S1.bag

# 3. 手工标注Ground Truth
# 使用RViz标注物体真实3D位置
python3 scripts/benchmark/annotate_ground_truth.py \
  --bag data/scenario_S1.bag \
  --output data/scenario_S1_gt.csv
```

#### 场景S2（典型场景）
```bash
# 1. 布置场景：15-20个物体
#    - 距离: 0.5-3.0m
#    - 轻微遮挡（10-20%）
#    - 混合高度（地面+货架）

# 2. 录制rosbag
rosbag record [...] -O data/scenario_S2.bag

# 3. 标注Ground Truth
python3 scripts/benchmark/annotate_ground_truth.py \
  --bag data/scenario_S2.bag \
  --output data/scenario_S2_gt.csv
```

### 待实现的测试脚本

根据 `docs/performance_evaluation.md`，以下测试需要实际数据后实现：

#### 1. 匹配精度评测
```bash
# 文件: scripts/benchmark/test_matching_accuracy.py
# 验证: P5 - 匈牙利算法比贪心提升4-6%精度
# 需要: scenario_S2.bag + scenario_S2_gt.csv
```

**实现步骤**：
1. 从rosbag加载双相机检测结果
2. 运行贪心匹配算法
3. 运行匈牙利匹配算法
4. 对比Ground Truth，计算精确匹配率
5. 验证是否提升4-6%

#### 2. 融合精度评测
```bash
# 文件: scripts/benchmark/test_fusion_accuracy.py
# 验证: P6 - WLS融合比单传感器提升25-40%精度
# 需要: scenario_S2.bag + scenario_S2_gt.csv
```

**实现步骤**：
1. 测量LiDAR单独精度（MAE、RMSE）
2. 测量深度相机单独精度
3. 测量WLS融合精度
4. 计算精度提升百分比
5. 验证是否提升25-40%

#### 3. 端到端延迟评测
```bash
# 文件: scripts/benchmark/test_e2e_latency.py
# 验证: P3, P4 - 系统级提升23%，帧率提升31%
# 需要: scenario_S2.bag
```

**实现步骤**：
1. 运行完整感知Pipeline
2. 记录端到端延迟（传感器→输出）
3. 对比CPU baseline vs GPU优化
4. 计算系统级提升百分比
5. 验证是否符合23%提升

---

## 评测原则（Linus风格）

### 问题1: 这些性能声明可信吗？
❌ **目前不可信**：因为没有实际数据验证

✅ **框架已就绪**：可以立即开始验证

### 问题2: 如何确保测试可靠？
遵循以下原则：
1. ✅ **可复现性**：固定随机种子，保存原始数据
2. ✅ **统计显著性**：至少30次迭代，p < 0.05
3. ✅ **控制变量**：每次只改变一个优化项
4. ✅ **真实数据**：不依赖模拟
5. ✅ **失败案例**：报告不适用场景

### 问题3: 会破坏什么吗？
❌ **不会**：
- 纯测试代码，不修改现有系统
- 测试脚本放在独立目录
- 结果保存到独立的 `results/` 目录

---

## 验证检查清单

| 阶段 | 任务 | 状态 | 负责人 |
|------|------|------|--------|
| **阶段0** | **环境准备** | ✅ 完成 | Claude |
| | 创建评测框架 | ✅ | Claude |
| | 验证框架正常工作 | ✅ | Claude |
| | 编写完整文档 | ✅ | Claude |
| **阶段1** | **数据采集** | 🔴 待开始 | 用户 |
| | 布置S1场景并录制rosbag | 🔴 | 用户 |
| | 布置S2场景并录制rosbag | 🔴 | 用户 |
| | 手工标注Ground Truth | 🔴 | 用户 |
| **阶段2** | **基准测试** | 🟡 部分完成 | 用户 |
| | 运行测距模块评测（模拟数据） | ✅ | 可立即运行 |
| | 实现匹配精度评测 | 🔴 | 需要真实数据 |
| | 实现融合精度评测 | 🔴 | 需要真实数据 |
| | 实现端到端延迟评测 | 🔴 | 需要真实数据 |
| **阶段3** | **统计分析** | 🟡 框架就绪 | 自动化 |
| | 显著性检验 | ✅ | 框架已实现 |
| | 效应量分析 | ✅ | 框架已实现 |
| | 生成图表 | 🔴 | 待实现 |
| **阶段4** | **报告生成** | 🟡 部分完成 | 自动化 |
| | Markdown摘要 | ✅ | 框架已实现 |
| | 对比预期 vs 实际 | ✅ | 框架已实现 |
| | 标注失败案例 | 🔴 | 需要真实测试 |

---

## 快速命令参考

```bash
# 目录
cd /home/agilex/MobileManipulator/scripts/benchmark

# 验证框架
python3 verify_framework.py

# 运行测距评测（模拟数据）
python3 test_measurement_performance.py

# 查看结果
ls -lh results/
cat results/measurement_summary.md

# 清理结果
rm -rf results/
```

---

## 下一步建议

### 选项1：立即体验（使用模拟数据）
```bash
cd /home/agilex/MobileManipulator/scripts/benchmark
python3 test_measurement_performance.py
```

**优点**：
- 立即可用，无需真实数据
- 验证框架工作正常
- 了解测试流程

**局限**：
- 无法验证精度指标
- 无法测试匹配和融合

### 选项2：完整评测（使用真实数据）
1. 采集数据（参考上面的数据采集计划）
2. 实现待完成的测试脚本
3. 运行完整评测
4. 生成最终报告

**优点**：
- 全面验证所有性能声明
- 可信的测试结果
- 发现实际问题

**成本**：
- 需要1-2周时间
- 需要布置测试场景
- 需要手工标注Ground Truth

---

## 相关文档

- [完整评测方案](./performance_evaluation.md) - 详细的评测计划和方法
- [系统设计文档](./multi_sensor_perception_design.md) - 性能声明的来源
- [GPU加速文档](./gpu_acceleration.md) - GPU优化的详细说明
- [评测工具README](../../../scripts/benchmark/README.md) - 工具使用指南

---

**最后更新**: 2026-01-30
**创建者**: Claude
**状态**: ✅ 框架就绪，等待真实数据评测
