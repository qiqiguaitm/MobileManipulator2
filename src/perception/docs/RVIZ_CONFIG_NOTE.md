# RViz 配置注意事项

## ⚠️ 重要提醒

### RViz 配置文件行为

**RViz 会在关闭时自动保存配置**，这可能会覆盖你手动修改的内容。

### 避免配置被覆盖的方法

#### 方法 1: 关闭时不保存（推荐）

关闭 RViz 时，如果提示保存配置：
- ✅ 选择 **"Don't Save"**
- ❌ 不要选择 "Save"

#### 方法 2: 重新加载配置

如果配置被覆盖了，运行修复脚本：

```bash
cd ~/MobileManipulator/scripts
./fix_rviz_config.sh
```

脚本会：
1. 停止所有 RViz 进程
2. 验证配置文件
3. 创建备份
4. 恢复正确的配置

#### 方法 3: 使用命令行重启

如果 RViz 运行中配置不对：

```bash
# 1. 停止 RViz
killall -9 rviz

# 2. 重新启动（会重新加载配置文件）
roslaunch perception perception_rviz.launch
```

## 当前配置

所有配置项都已正确设置：

| 项目 | 配置值 |
|------|--------|
| Camera Image | `/camera/top/color/image_raw` |
| Distance Labels | `/perception_rviz_node/distance_labels` |
| Object Markers | `/perception_rviz_node/distance_labels` |
| TF Frames | 7 个（base_link, arm_base_link, rslidar, 3×camera, gripper_link） |

## 验证配置

```bash
# 快速验证
cd /home/agilex/MobileManipulator/src/perception
grep -n "Topic:" config/perception_3d.rviz

# 应该看到：
# 88:      Topic: /camera/top/color/image_raw
# 101:     Topic: /perception_rviz_node/rgb_pointcloud
# 117:     Topic: /lidar/chassis/point_cloud
# 142:     Topic: /perception_rviz_node/object_clouds
# 157:     Topic: /perception_rviz_node/distance_labels
# 166:     Topic: /perception_rviz_node/distance_labels
```

## 配置文件位置

```
/home/agilex/MobileManipulator/src/perception/config/perception_3d.rviz
```

**权限**: `rw-r--r--` (644) - 可读写

## 如果配置又不生效了

```bash
# 运行修复脚本
cd ~/MobileManipulator/scripts
./fix_rviz_config.sh

# 重新启动
./start_perception.sh --rviz --test
```

就这么简单！✅
