# RViz 配置不生效问题排查

## 问题描述

修改了 `perception_3d.rviz` 配置文件，但启动 RViz 后发现配置没有生效。

## 根本原因

**RViz 的行为**：
1. RViz 启动时加载配置文件到内存
2. 用户在 RViz 中做的任何修改（添加/删除显示项、调整参数等）都保存在内存中
3. **关闭 RViz 时，默认会将当前配置保存回配置文件**

**结果**：
- 你修改了配置文件
- 但 RViz 关闭时用旧配置覆盖了你的修改
- 或者 RViz 还在运行，使用的是内存中的旧配置

## 解决方案

### 方法 1: 使用修复脚本（推荐）

```bash
cd ~/MobileManipulator/scripts
./fix_rviz_config.sh
```

**脚本功能**：
1. ✅ 停止所有 RViz 进程
2. ✅ 验证配置文件正确性
3. ✅ 备份当前配置
4. ✅ **将配置文件设为只读**（防止 RViz 覆盖）

### 方法 2: 手动修复

```bash
# 1. 停止 RViz
killall -9 rviz

# 2. 验证配置文件
cat /home/agilex/MobileManipulator/src/perception/config/perception_3d.rviz

# 3. 设为只读
chmod 444 /home/agilex/MobileManipulator/src/perception/config/perception_3d.rviz

# 4. 重新启动
roslaunch perception perception_rviz.launch
```

### 方法 3: 在 RViz 中手动加载

```bash
# 1. 启动 RViz（空配置）
rviz

# 2. 在 RViz 中: File → Open Config
# 3. 选择: /home/agilex/MobileManipulator/src/perception/config/perception_3d.rviz
# 4. ⚠️ 关闭时选择 "Don't Save"
```

## 验证配置已生效

启动 RViz 后，检查以下项：

### 1. Camera Image
- **位置**: 左侧 Displays 面板
- **应该显示**: Camera Image (Enabled)
- **Topic**: `/camera/top/color/image_raw`

### 2. Distance Labels
- **位置**: 左侧 Displays 面板
- **应该显示**: Distance Labels (Enabled)
- **Topic**: `/perception_rviz_node/distance_labels`

### 3. Object Markers
- **位置**: 左侧 Displays 面板
- **应该显示**: Object Markers (Enabled)
- **Topic**: `/perception_rviz_node/distance_labels`

### 4. TF Frames
- **位置**: 左侧 Displays 面板 → TF → Frames
- **应该显示** (Value: true):
  - base_link
  - arm_base_link
  - rslidar
  - top_camera_link
  - hand_camera_link
  - chassis_camera_link
  - gripper_link

## 防止配置被覆盖

### 临时方案：设为只读

```bash
chmod 444 /home/agilex/MobileManipulator/src/perception/config/perception_3d.rviz
```

**优点**:
- ✅ RViz 无法保存修改
- ✅ 配置永远保持正确

**缺点**:
- ❌ 无法在 RViz 中修改配置
- ❌ 需要修改时要先改回可写：`chmod 644 <file>`

### 永久方案：使用 launch 参数

```xml
<!-- 在 launch 文件中禁止保存 -->
<node name="rviz" pkg="rviz" type="rviz"
      args="-d $(find perception)/config/perception_3d.rviz --splash-screen 0"/>
```

RViz 没有官方参数禁止保存，但可以通过只读权限实现。

## 常见问题

### Q1: 为什么配置文件看起来是对的，但 RViz 显示不对？

**A**: RViz 可能还在运行，使用的是内存中的旧配置。

**解决**:
```bash
killall -9 rviz
roslaunch perception perception_rviz.launch
```

### Q2: 修改配置后，关闭 RViz 时提示保存，应该选什么？

**A**: 选择 **"Don't Save"**，否则会覆盖配置文件。

**更好的方案**: 将配置文件设为只读，RViz 就无法保存。

### Q3: 配置文件是只读的，我想修改怎么办？

**A**: 临时改为可写，修改后再改回只读：

```bash
# 改为可写
chmod 644 /home/agilex/MobileManipulator/src/perception/config/perception_3d.rviz

# 修改配置
vim /home/agilex/MobileManipulator/src/perception/config/perception_3d.rviz

# 改回只读
chmod 444 /home/agilex/MobileManipulator/src/perception/config/perception_3d.rviz
```

### Q4: 如何恢复到备份？

**A**: 修复脚本会自动创建备份：

```bash
# 查看备份文件
ls -l /home/agilex/MobileManipulator/src/perception/config/*.backup.*

# 恢复备份
cp /home/agilex/MobileManipulator/src/perception/config/perception_3d.rviz.backup.YYYYMMDD_HHMMSS \
   /home/agilex/MobileManipulator/src/perception/config/perception_3d.rviz

# 改为只读
chmod 444 /home/agilex/MobileManipulator/src/perception/config/perception_3d.rviz
```

## 配置文件结构

```yaml
Panels:
  - Class: rviz/Displays
    Property Tree Widget:
      Expanded:
        - /Camera Image1       # 相机图像面板

Visualization Manager:
  Displays:
    - Class: rviz/TF
      Frames:
        All Enabled: false
        base_link:
          Value: true          # ← 控制是否显示
        arm_base_link:
          Value: true
        # ... 其他坐标系

    - Class: rviz/Image
      Name: Camera Image
      Topic: /camera/top/color/image_raw  # ← 相机 topic

    - Class: rviz/MarkerArray
      Name: Object Markers
      Topic: /perception_rviz_node/distance_labels  # ← Object Markers topic

    - Class: rviz/MarkerArray
      Name: Distance Labels
      Topic: /perception_rviz_node/distance_labels  # ← Distance Labels topic
```

## 验证脚本

运行以下脚本验证配置：

```bash
cd /home/agilex/MobileManipulator/src/perception
bash /tmp/verify_rviz_full.sh
```

**预期输出**:
```
2. Camera Image Topic:
      Topic: /camera/top/color/image_raw

3. Distance Labels Topic:
      Topic: /perception_rviz_node/distance_labels

4. Object Markers Topic:
      Topic: /perception_rviz_node/distance_labels

5. TF 显示的坐标系:
  - base_link
  - arm_base_link
  - rslidar
  - top_camera_link
  - hand_camera_link
  - chassis_camera_link
  - gripper_link
```

## 完整启动流程

```bash
# 1. 修复配置（首次或遇到问题时）
cd ~/MobileManipulator/scripts
./fix_rviz_config.sh

# 2. 启动完整系统
./start_perception.sh --rviz --test

# 3. 验证 RViz 配置
# - 检查左侧 Displays 面板
# - 确认所有显示项正确
# - 确认 TF 坐标系正确
```

## 文件权限说明

| 权限 | 数字 | 说明 | RViz 行为 |
|------|------|------|-----------|
| `rw-r--r--` | 644 | 可读写 | ✅ 可加载 ✅ 可保存 |
| `r--r--r--` | 444 | 只读 | ✅ 可加载 ❌ 无法保存 |

**推荐**: 使用 `444` (只读) 防止配置被意外覆盖。

## 相关文档

- `perception_3d_rviz_update.md` - 配置更新记录
- `fix_rviz_config.sh` - 配置修复脚本
- `QUICK_START.md` - 快速启动指南

## 总结

**问题**: RViz 关闭时会保存配置覆盖文件
**解决**: 将配置文件设为只读 (`chmod 444`)
**验证**: 使用 `fix_rviz_config.sh` 脚本

现在配置已经修复并生效！🎉
