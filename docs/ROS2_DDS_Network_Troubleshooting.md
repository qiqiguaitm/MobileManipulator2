# ROS2 DDS 网络故障排查指南

## 问题概述

**症状**: `ros2 topic list` 只显示 `/parameter_events` 和 `/rosout`，尽管相机节点正在运行，但相机话题不可见。

**日期**: 2026-02-11

---

## 已确认的根本原因

### 1. ROS_DOMAIN_ID 不匹配（主要问题）

**问题**: 相机 launch 从 `~/.bashrc` 继承了 `ROS_DOMAIN_ID=42`，但终端使用默认域 0。

**证据**:
```bash
# 相机进程
$ cat /proc/<camera_pid>/environ | tr '\0' '\n' | grep DOMAIN
ROS_DOMAIN_ID=42

# 终端
$ echo $ROS_DOMAIN_ID
(空 - 默认为 0)
```

**影响**: DDS 为每个域使用不同的多播端口：
- 域 0: 端口 7400
- 域 42: 端口 17900

不同域的节点无法互相发现。

**解决方案**:
```bash
export ROS_DOMAIN_ID=42
```

---

### 2. 多网络接口（次要问题）

**问题**: 系统有多个活动的网络接口，导致 DDS 绑定混乱。

**网络配置**:
```
wlP1p1s0 (WiFi):   192.168.112.153/24
eno1 (以太网):     192.168.0.10/24
lo (回环):         127.0.0.1
```

**影响**: 不同的 ROS2 进程随机绑定到不同接口，阻止单播通信。

**解决方案**: 通过 FastRTPS 配置强制指定接口。

**配置文件**: `/home/didi/.ros/fastdds.xml`
```xml
<?xml version="1.0" encoding="UTF-8"?>
<dds xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <profiles>
        <transport_descriptors>
            <transport_descriptor>
                <transport_id>custom_udp</transport_id>
                <type>UDPv4</type>
                <interfaceWhiteList>
                    <address>192.168.0.10</address>
                </interfaceWhiteList>
            </transport_descriptor>
        </transport_descriptors>
        <participant profile_name="participant_profile" is_default_profile="true">
            <rtps>
                <userTransports>
                    <transport_id>custom_udp</transport_id>
                </userTransports>
                <useBuiltinTransports>false</useBuiltinTransports>
            </rtps>
        </participant>
    </profiles>
</dds>
```

**环境变量**:
```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/didi/.ros/fastdds.xml
```

---

## 最终配置

添加到 `~/.bashrc`:
```bash
export ROS_DOMAIN_ID=42
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/didi/.ros/fastdds.xml
```

---

## 诊断命令

### 检查运行中的进程
```bash
ps aux | grep realsense2_camera | grep -v grep
```

### 检查进程网络绑定
```bash
PID=<camera_pid>
ss -ulnp | grep $PID
```

### 检查进程环境变量
```bash
cat /proc/<pid>/environ | tr '\0' '\n' | grep -E "DOMAIN|FAST"
```

### 重启 ROS2 Daemon（更改环境变量后必须执行）
```bash
ros2 daemon stop && ros2 daemon start
```

### 使用 Python 测试发现机制
```python
import rclpy
rclpy.init()
node = rclpy.create_node('test')
print(node.get_topic_names_and_types())
print(node.get_node_names_and_namespaces())
```

---

## 关键经验总结

1. **ROS_DOMAIN_ID 必须匹配** - 所有终端和进程必须使用相同的域 ID
2. **ROS2 daemon 会缓存环境** - 更改环境变量后必须重启 daemon
3. **多网卡系统需要显式接口配置** - 通过 FastRTPS XML 文件指定
4. **FastRTPS（非 CycloneDDS）** 是本系统的默认 RMW - 检查命令：
   ```bash
   python3 -c "from rclpy.utilities import get_rmw_implementation_identifier; print(get_rmw_implementation_identifier())"
   ```
5. **interfaceWhiteList 需要 IP 地址**，而非接口名称

---

## 备选方案（未采用）

| 方案 | 命令 | 权衡 |
|------|------|------|
| 仅本地通信 | `export ROS_LOCALHOST_ONLY=1` | 无法跨机器通信 |
| 禁用 WiFi | `sudo ip link set wlP1p1s0 down` | 失去 WiFi 连接 |
| 使用 CycloneDDS | `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` | 需要不同的配置格式 |

---

## 参考资料

- [FastDDS XML 配置文档](https://fast-dds.docs.eprosima.com/en/latest/fastdds/xml_configuration/xml_configuration.html)
- [ROS2 DDS 调优指南](https://docs.ros.org/en/humble/How-To-Guides/DDS-tuning.html)
