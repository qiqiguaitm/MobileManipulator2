# 实时质量检查采集工具

## 问题背景

当前标定数据存在质量问题：
- **重投影误差高**：平均RMS 4.38px（理想值 <1.5px）
- **标定板姿态不当**：前俯/后仰30-45°（推荐 <20°）
- **采集时无质量反馈**：无法实时判断数据是否可用

## 改进方案

### 新增工具：`collect_data_improved.py`

实时显示：
- ✅ 重投影误差（RMS/Mean/Max）
- ✅ 标定板倾斜角度
- ✅ 质量评级（良好/中等/差）
- ✅ 误差可视化（黄色箭头标注>1px的点）

## 使用方法

### 1. 启动相机和LiDAR

```bash
# Terminal 1: 启动相机
roslaunch camera_driver camera_driver.launch top_enable:=true

# Terminal 2: 启动LiDAR
roslaunch lidar_driver rslidar_driver.launch
```

### 2. 运行实时质量检查

```bash
cd /Users/tim/Documents/work/MobileManipulator/src/cam_lidar_calibrate

# 使用默认参数（Top Camera）
python3 collect_data_improved.py

# 或指定参数
python3 collect_data_improved.py \
  --image-topic /camera/top/color/image_raw \
  --intrinsic ./calib_data_top/conf/camera_intrinsic.yaml \
  --pattern 7x6 \
  --square 0.072
```

### 3. 操作流程

1. **摆放标定板** - 参考guidance_to_collect_data.md的姿态要求
2. **按Enter** - 启动实时质量检测窗口
3. **查看指标**：
   - 绿色文字：RMS < 1.5px（质量良好，推荐录制）
   - 橙色文字：RMS 1.5-3px（质量中等，可接受）
   - 红色文字：RMS > 3px（质量差，建议重调）
4. **调整姿态** - 根据提示调整标定板：
   - 减小倾斜角度（推荐 <20°）
   - 确保标定板完整可见
   - 避免运动模糊
5. **确认录制**：
   - 按 `q` - 质量合格，继续
   - 按 `r` - 重新调整
   - 按 `s` - 跳过此姿态

## 质量标准

| 指标 | 推荐值 | 最大容忍值 | 说明 |
|------|--------|------------|------|
| RMS重投影误差 | <1.5px | <3px | 反映角点检测精度 |
| 标定板倾斜角度 | <15° | <25° | 过大会导致透视畸变 |
| 可见角点数 | 42/42 | 42/42 | 必须检测到所有角点 |

## 可视化说明

![质量检测界面](example_quality_check.png)

- **绿色圆圈**：检测到的角点位置
- **红色叉号**：重投影位置（理想情况应与绿色重合）
- **黄色箭头**：误差>1px的点，箭头长度=误差大小
- **底部面板**：实时质量指标

## 常见问题

### Q: 为什么RMS一直>3px？

**A**: 可能原因：
1. **标定板倾斜太大** → 减小倾斜角度到<20°
2. **内参不准确** → 确认使用正确分辨率的内参
3. **标定板遮挡** → 确保所有角点可见

### Q: 倾斜角度如何控制？

**A**: 经验法则：
- 0-15°：优秀（绿色）
- 15-25°：可接受（橙色）
- >25°：过大（红色）- 重投影误差会急剧增大

参考姿态（推荐角度）：
- 垂直：2-3°微倾（确保LiDAR有≥5条扫描线）
- 前俯：10-15°
- 后仰：10-20°

### Q: 如何快速判断质量？

**A**: 看3个指标：
1. RMS < 2px → ✓
2. 倾斜角度 < 20° → ✓
3. 黄色箭头很少或没有 → ✓

都满足就可以录制。

## 重新采集建议

基于当前数据分析（平均RMS 4.38px），建议：

1. **全部重新采集** - 当前数据质量不足以支撑高精度标定
2. **使用本工具** - 每个姿态都进行质量检查
3. **目标质量**：
   - 所有帧RMS < 2px
   - 至少50%的帧RMS < 1.5px
   - 倾斜角度都 < 20°

预计采集时间：15-20分钟（9个姿态，每个2分钟调整+检查）

## 下一步

采集完成后，使用原流程提取和标定：

```bash
# 提取数据
python extract_data.py calib_bags_top/data.bag top

# 标定（使用改进版交互式标定）
python interactive_calibrate.py --data-dir ./calib_data_top
```

改进版标定工具已支持：
- ✅ 投影引导的3D点云选择
- ✅ 标定板倾斜检测（允许60°以内）
- ✅ 浏览器3D可视化（macOS兼容）
