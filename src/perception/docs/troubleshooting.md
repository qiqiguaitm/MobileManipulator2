# Perception 模块故障排除

## 常见问题

### 1. DINO-X 服务不可用

**问题表现：**
```
[ERROR] [ScenePerception3D] DINO-X 服务不可用: http://192.168.112.14:10086
RuntimeError: DINO-X 服务不可用
```

节点启动时健康检查失败，即使 `curl` 命令可以正常访问服务。

**原因分析：**

Python `requests` 库可能读取系统代理设置（`http_proxy`, `https_proxy` 环境变量），导致内网服务请求被路由到代理服务器。

**解决方案：**

1. **环境变量层面**：在启动脚本中彻底清除所有代理变量并设置 no_proxy
   ```bash
   # 清除所有代理变量
   unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

   # 设置内网 IP 不走代理
   export no_proxy="localhost,127.0.0.1,192.168.0.0/16,10.0.0.0/8"
   export NO_PROXY="$no_proxy"
   ```

2. **curl 层面**：使用 `--noproxy '*'` 显式禁用代理
   ```bash
   curl -s --noproxy '*' --connect-timeout 2 --max-time 3 http://192.168.112.14:10086
   ```

3. **Python requests 层面**：显式禁用代理
   ```python
   resp = requests.get(url, timeout=timeout, proxies={'http': None, 'https': None})
   ```

4. **Session 层面**：配置 requests.Session 不使用环境代理
   ```python
   session = requests.Session()
   session.trust_env = False  # 不读取环境变量
   session.proxies = {'http': None, 'https': None}
   ```

**关键点**：仅 `unset http_proxy` 不够，还需要清除 `all_proxy` 并设置 `no_proxy`。

**已修复文件：**
- `scripts/build_perception.sh` - curl 添加 `--noproxy '*'`
- `scripts/start_perception.sh` - 完整代理清除 + no_proxy 设置
- `src/scene_perception_3d_node.py` - requests.get 添加 proxies 参数
- `src/percept.py` - Session 配置 trust_env=False

---

### 2. cv_bridge OpenCV ABI 不兼容

**问题表现：**
```
TypeError: img is not a numpy array, neither a scalar
```
或
```
Segmentation fault (core dumped)
```

**原因分析：**

系统 cv_bridge 使用 OpenCV 4.2，但用户环境可能有其他版本的 OpenCV。

**解决方案：**

使用 `passthrough` 编码避免 cv_bridge 内部转换：
```python
# 修改前
rgb = bridge.imgmsg_to_cv2(msg, 'bgr8')

# 修改后
rgb = bridge.imgmsg_to_cv2(msg, 'passthrough')
if msg.encoding == 'rgb8':
    rgb = rgb[:, :, ::-1]  # RGB -> BGR
```

详见 `docs/cv_bridge_opencv_issue.md`

---

### 3. LiDAR 投影点数为 0

**问题表现：**

可视化图中没有 LiDAR 投影点，距离显示 `L:N/A`。

**可能原因：**

1. **物理遮挡**：LiDAR 被障碍物遮挡
2. **坐标系错误**：外参标定文件不正确
3. **分辨率不匹配**：相机分辨率与标定时不一致

**排查步骤：**

```bash
# 1. 检查 LiDAR 点云是否有数据
rostopic hz /lidar/chassis/point_cloud

# 2. 检查点云数量
rostopic echo /lidar/chassis/point_cloud/width -n 1

# 3. 确认相机分辨率 (应为 1280x720)
rostopic echo /camera/top/color/camera_info/width -n 1
rostopic echo /camera/top/color/camera_info/height -n 1
```

**解决方案：**

- 确保 LiDAR 前方无遮挡
- 相机分辨率设为 1280x720（与 LiDAR 标定一致）
- 检查外参文件 `config/extrinsics_*.yaml`

---

### 4. 启动脚本卡住

**问题表现：**

`./start_perception.sh` 或 `./build_perception.sh` 执行时卡在检查外部服务。

**原因分析：**

curl 请求通过代理访问内网服务，代理超时。

**解决方案：**

确保 curl 使用 `--noproxy '*' --max-time 3` 参数（已在脚本中修复）。

临时解决：
```bash
# 手动清除代理环境变量
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# 然后运行脚本
./start_perception.sh --test
```

---

## 日志位置

| 日志 | 路径 |
|------|------|
| ROS launch 日志 | `/tmp/perception_launch.log` |
| RViz 日志 | `/tmp/perception_rviz.log` |
| ROS 系统日志 | `~/.ros/log/` |
| 可视化结果 | `src/perception/results/` |

---

## 调试命令

```bash
# 检查节点状态
rosnode list | grep perception
rosnode info /scene_perception_3d

# 检查服务
rosservice list | grep perception
rosservice call /scene_perception_3d/detect "{prompt: 'bottle', enable_lidar: true}"

# 检查 topics
rostopic list | grep perception
rostopic hz /perception_rviz_node/rgb_pointcloud

# 测试外部服务
curl --noproxy '*' -s -o /dev/null -w "%{http_code}" http://192.168.112.14:10086
curl --noproxy '*' -s -o /dev/null -w "%{http_code}" http://192.168.112.14:8086
```
