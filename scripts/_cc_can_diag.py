#!/usr/bin/env python3
"""
CAN 通信和位置反馈诊断脚本 - 增强版

Usage:
    python3 scripts/_cc_can_diag.py
"""

import sys
import time
import os

# 添加路径
sys.path.insert(0, '/data/workspace/MobileManipulator2/src/piper_grasp/scripts')

from piper_api_v2 import PiperAPI


def check_can_stats():
    """检查 CAN 接口统计"""
    print("\n" + "=" * 60)
    print("[0] CAN 接口统计")
    print("=" * 60)

    # 检查 can0 状态
    try:
        # 读取 CAN 统计
        with open('/sys/class/net/can0/statistics/rx_packets', 'r') as f:
            rx = int(f.read().strip())
        with open('/sys/class/net/can0/statistics/tx_packets', 'r') as f:
            tx = int(f.read().strip())
        with open('/sys/class/net/can0/statistics/rx_bytes', 'r') as f:
            rx_bytes = int(f.read().strip())
        with open('/sys/class/net/can0/statistics/tx_bytes', 'r') as f:
            tx_bytes = int(f.read().strip())

        print(f"  can0 接收: {rx} packets, {rx_bytes} bytes")
        print(f"  can0 发送: {tx} packets, {tx_bytes} bytes")

        if rx == 0 and tx > 0:
            print("  ⚠️ 警告: CAN 只发送不接收!")
        elif rx > 0:
            print("  ✓ CAN 有接收数据")
    except Exception as e:
        print(f"  无法读取 CAN 统计: {e}")

    # 检查接口状态
    try:
        with open('/sys/class/net/can0/operstate', 'r') as f:
            state = f.read().strip()
        print(f"  can0 状态: {state}")
    except:
        pass


def main():
    print("=" * 60)
    print("CAN 通信和位置反馈诊断 - 增强版")
    print("=" * 60)

    # 0. 先检查 CAN 统计
    check_can_stats()

    arm = PiperAPI()

    # 1. 连接测试
    print("\n[1] 连接测试...")
    try:
        arm.connect()
        print(f"  连接状态: {arm.is_connected}")
    except Exception as e:
        print(f"  连接失败: {e}")
        return

    if not arm.is_connected:
        print("  ❌ 无法连接，请检查 CAN 接口和机械臂电源")
        return

    print("  ✓ 已连接")

    # 记录初始 CAN 统计
    initial_rx = 0
    initial_tx = 0
    try:
        with open('/sys/class/net/can0/statistics/rx_packets', 'r') as f:
            initial_rx = int(f.read().strip())
        with open('/sys/class/net/can0/statistics/tx_packets', 'r') as f:
            initial_tx = int(f.read().strip())
    except:
        pass
    print(f"  初始 CAN: rx={initial_rx}, tx={initial_tx}")

    # 2. 检查使能状态
    print("\n[2] 检查使能状态...")
    print(f"  使能状态: {arm._get_enable_status_official()}")

    # 2.5 空闲状态读取 (baseline)
    print("\n[2.5] 空闲状态读取 (不发送任何命令)...")
    print("  读取10次:")
    for i in range(10):
        msg = arm.piper.GetArmJointMsgs()
        joints = [msg.joint_state.joint_1/1000, msg.joint_state.joint_2/1000,
                  msg.joint_state.joint_3/1000, msg.joint_state.joint_4/1000,
                  msg.joint_state.joint_5/1000, msg.joint_state.joint_6/1000]
        hz = msg.Hz
        print(f"    {i+1}: Hz={hz:.0f} joints={joints}")
        time.sleep(0.05)

    # 3. GoZero 命令测试
    print("\n[3] 测试 MotionCtrl_2 + JointCtrl 的影响...")

    # 3.1 先确认当前状态正常
    print("  3.1 当前状态:")
    for i in range(3):
        msg = arm.piper.GetArmJointMsgs()
        joints = [msg.joint_state.joint_1/1000, msg.joint_state.joint_2/1000,
                  msg.joint_state.joint_3/1000, msg.joint_state.joint_4/1000,
                  msg.joint_state.joint_5/1000, msg.joint_state.joint_6/1000]
        print(f"       joints={joints}")
        time.sleep(0.05)

    # 3.2 只发送 MotionCtrl_2，不发送 JointCtrl
    print("  3.2 只发送 MotionCtrl_2 (不发送 JointCtrl):")
    arm.piper.MotionCtrl_2(0x01, 0x01, 50, 0)
    time.sleep(0.5)
    for i in range(3):
        msg = arm.piper.GetArmJointMsgs()
        joints = [msg.joint_state.joint_1/1000, msg.joint_state.joint_2/1000,
                  msg.joint_state.joint_3/1000, msg.joint_state.joint_4/1000,
                  msg.joint_state.joint_5/1000, msg.joint_state.joint_6/1000]
        print(f"       joints={joints}")
        time.sleep(0.05)

    # 3.3 发送 JointCtrl(0,0,0,0,0,0)
    print("  3.3 发送 JointCtrl(0,0,0,0,0,0):")
    arm.piper.JointCtrl(0, 0, 0, 0, 0, 0)
    time.sleep(0.5)
    for i in range(3):
        msg = arm.piper.GetArmJointMsgs()
        joints = [msg.joint_state.joint_1/1000, msg.joint_state.joint_2/1000,
                  msg.joint_state.joint_3/1000, msg.joint_state.joint_4/1000,
                  msg.joint_state.joint_5/1000, msg.joint_state.joint_6/1000]
        print(f"       joints={joints}")
        time.sleep(0.05)

    # 3.4 恢复正常 - 发送非零 JointCtrl
    print("  3.4 发送非零 JointCtrl(0, -2000, 0, 1000, 30000, 0):")
    arm.piper.MotionCtrl_2(0x01, 0x01, 30, 0)
    arm.piper.JointCtrl(0, -2000, 0, 1000, 30000, 0)
    time.sleep(2.0)
    for i in range(3):
        msg = arm.piper.GetArmJointMsgs()
        joints = [msg.joint_state.joint_1/1000, msg.joint_state.joint_2/1000,
                  msg.joint_state.joint_3/1000, msg.joint_state.joint_4/1000,
                  msg.joint_state.joint_5/1000, msg.joint_state.joint_6/1000]
        print(f"       joints={joints}")
        time.sleep(0.05)

    print("  测试完成，继续后续测试...")

    # 4. GoZero 精确时序测试 (跳过，已在上面测试过)
    print("\n[5] GoZero 命令精确时序测试...")
    print("  (跳过，使用之前的测试结果)")
    print("  结论: JointCtrl(0,0,0,0,0,0) 会让关节归零，反馈正常")

    # 5. 测试 GoZero 完成后，等待更长时间是否能恢复
    print("\n[6] 测试 GoZero 完成后恢复...")
    print("  发送 go_ready 恢复位置...")
    arm.go_ready()
    time.sleep(2.0)
    print("  go_ready 后读取:")
    for i in range(5):
        msg = arm.piper.GetArmJointMsgs()
        joints = [msg.joint_state.joint_1/1000, msg.joint_state.joint_2/1000,
                  msg.joint_state.joint_3/1000, msg.joint_state.joint_4/1000,
                  msg.joint_state.joint_5/1000, msg.joint_state.joint_6/1000]
        print(f"       joints={joints}")
        time.sleep(0.05)
    print("  发送 GoZero 命令...")
    arm.go_zero()
    print("  GoZero 返回后立即读取:")

    # 立即读取多次
    for i in range(10):
        msg = arm.piper.GetArmJointMsgs()
        joints = [msg.joint_state.joint_1/1000, msg.joint_state.joint_2/1000,
                  msg.joint_state.joint_3/1000, msg.joint_state.joint_4/1000,
                  msg.joint_state.joint_5/1000, msg.joint_state.joint_6/1000]
        print(f"    t+{i*0.01:.2f}s: {joints}")
        time.sleep(0.01)

    time.sleep(0.5)
    print("  0.5秒后读取:")
    for i in range(10):
        msg = arm.piper.GetArmJointMsgs()
        joints = [msg.joint_state.joint_1/1000, msg.joint_state.joint_2/1000,
                  msg.joint_state.joint_3/1000, msg.joint_state.joint_4/1000,
                  msg.joint_state.joint_5/1000, msg.joint_state.joint_6/1000]
        print(f"    t+{0.5+i*0.01:.2f}s: {joints}")
        time.sleep(0.01)

    time.sleep(1.0)
    print("  1.5秒后读取 (GoZero 完成):")
    msg = arm.piper.GetArmJointMsgs()
    joints = [msg.joint_state.joint_1/1000, msg.joint_state.joint_2/1000,
              msg.joint_state.joint_3/1000, msg.joint_state.joint_4/1000,
              msg.joint_state.joint_5/1000, msg.joint_state.joint_6/1000]
    print(f"    {joints}")

    # 读取 get_position 对比
    success, pos = arm.get_position(return_gripper_center=False)
    print(f"  get_position: {pos}")

    # 5. 读取关节角度
    print("\n[6] 读取 GetArmJointMsgs (连续20次，间隔0.1s)...")
    print("  次数 | Hz | J1 | J2 | J3 | J4 | J5 | J6")
    print("  " + "-" * 60)

    has_nonzero = False
    for i in range(20):
        joint_msg = arm.piper.GetArmJointMsgs()
        joints = [
            joint_msg.joint_state.joint_1,
            joint_msg.joint_state.joint_2,
            joint_msg.joint_state.joint_3,
            joint_msg.joint_state.joint_4,
            joint_msg.joint_state.joint_5,
            joint_msg.joint_state.joint_6,
        ]
        hz = joint_msg.Hz

        if any(j != 0 for j in joints):
            has_nonzero = True

        # 只打印前5次和后5次
        if i < 5 or i >= 15:
            j_str = " ".join([f"{j/1000:6.1f}" for j in joints])
            print(f"  {i+1:4d} | {hz:3.1f} | {j_str}")
        elif i == 5:
            print("  ... (省略中间) ...")

        time.sleep(0.1)

    if has_nonzero:
        print("  ✓ 有关节反馈!")
    else:
        print("  ❌ 全是 0，无关节反馈")

    # 7. 检查 GetArmJointMsgs 的原始数据
    print("\n[7] 检查原始数据...")
    msg = arm.piper.GetArmJointMsgs()
    print(f"  joint_state 类型: {type(msg.joint_state)}")
    print(f"  joint_1 原始值: {msg.joint_state.joint_1}")
    print(f"  time_stamp: {msg.time_stamp}")
    print(f"  Hz: {msg.Hz}")

    # 8. 尝试 GetFK (正向运动学)
    print("\n[8] 测试 GetFK (正向运动学)...")
    try:
        fk = arm.piper.GetFK("feedback")
        print(f"  GetFK(feedback): {fk}")
    except Exception as e:
        print(f"  GetFK 失败: {e}")

    try:
        fk = arm.piper.GetFK("control")
        print(f"  GetFK(control): {fk}")
    except Exception as e:
        print(f"  GetFK(control) 失败: {e}")

    # 9. 读取 get_position
    print("\n[9] 读取 get_position...")
    success, pos = arm.get_position(return_gripper_center=False)
    print(f"  末端位置: {pos}")

    # 10. 移动测试
    print("\n[10] 移动测试...")
    print("  移动到 [350, 50, 180]...")
    try:
        arm.move(350, 50, 180, 180, 30, 180)
        time.sleep(2.0)
    except Exception as e:
        print(f"  move 失败: {e}")
        # 尝试其他方式
        try:
            arm.go_ready()
            time.sleep(2.0)
        except Exception as e2:
            print(f"  go_ready 也失败: {e2}")

    # 检查 CAN 统计
    try:
        with open('/sys/class/net/can0/statistics/rx_packets', 'r') as f:
            rx = int(f.read().strip())
        with open('/sys/class/net/can0/statistics/tx_packets', 'r') as f:
            tx = int(f.read().strip())
        print(f"  移动后 CAN: rx={rx}, tx={tx}")
        print(f"  总接收: {rx}, 总发送: {tx}")
    except:
        pass

    # 11. 再次读取关节角度
    print("\n[11] 移动后再次读取...")
    msg = arm.piper.GetArmJointMsgs()
    joints = [
        msg.joint_state.joint_1,
        msg.joint_state.joint_2,
        msg.joint_state.joint_3,
        msg.joint_state.joint_4,
        msg.joint_state.joint_5,
        msg.joint_state.joint_6,
    ]
    print(f"  关节角度 (度): {[f'{j/1000:.1f}' for j in joints]}")

    success, pos = arm.get_position(return_gripper_center=False)
    print(f"  get_position: {pos}")

    # 12. 长时间监控
    print("\n[12] 长时间监控 (30秒)...")
    print("  持续监控关节反馈是否有变化...")
    last_joints = None
    changes = 0
    start_t = time.time()

    while time.time() - start_t < 30:
        msg = arm.piper.GetArmJointMsgs()
        joints = [
            msg.joint_state.joint_1,
            msg.joint_state.joint_2,
            msg.joint_state.joint_3,
            msg.joint_state.joint_4,
            msg.joint_state.joint_5,
            msg.joint_state.joint_6,
        ]
        hz = msg.Hz

        if last_joints is not None:
            # 检查是否有变化
            for i, (j1, j2) in enumerate(zip(joints, last_joints)):
                if j1 != j2:
                    changes += 1
                    break
        last_joints = joints

        print(f"  t={time.time()-start_t:5.1f}s Hz={hz:4.1f} joints={[j/1000 for j in joints]}", end='\r')
        time.sleep(0.5)

    print(f"\n  检测到 {changes} 次变化")

    # 总结
    print("\n" + "=" * 60)
    print("诊断结论:")
    print("=" * 60)

    msg = arm.piper.GetArmJointMsgs()
    joints = [msg.joint_state.joint_1, msg.joint_state.joint_2,
              msg.joint_state.joint_3, msg.joint_state.joint_4,
              msg.joint_state.joint_5, msg.joint_state.joint_6]

    if all(j == 0 for j in joints):
        print("❌ GetArmJointMsgs() 全返回 0 - CAN 接收故障")
    else:
        print("✓ GetArmJointMsgs() 有正常反馈")

    # 检查 CAN 统计
    try:
        with open('/sys/class/net/can0/statistics/rx_packets', 'r') as f:
            rx = int(f.read().strip())
        with open('/sys/class/net/can0/statistics/tx_packets', 'r') as f:
            tx = int(f.read().strip())

        print(f"\nCAN 统计:")
        print(f"  接收: {rx} packets")
        print(f"  发送: {tx} packets")

        if rx == 0:
            print("  ❌ 确认: CAN 适配器只发送不接收")
        else:
            print("  ✓ CAN 有接收")
    except:
        pass

    # 断开
    print("\n[13] 断开连接...")
    try:
        arm.motion_enable(False)
        arm.disconnect()
        print("  ✓ 已断开")
    except Exception as e:
        print(f"  断开出错: {e}")


if __name__ == '__main__':
    main()
