# coding=utf-8
import math
from xarm.wrapper import XArmAPI
import json
import logging,os
import socket
import time
import sys
import numpy as np
import cv2
import pyrealsense2 as rs
from datetime import datetime
import yaml

from libs.log_setting import CommonLog
from libs.auxiliary import create_folder_with_date, get_ip, popup_message

# Camera intrinsics
from calibration_common import CameraIntrinsicsManager

# Global variables
data_path = None  # Will be created based on mode
calibration_mode = None  # 'eye_in_hand' or 'eye_to_hand'
collected_frames_data = []  # Store frame metadata
camera_serial = None  # RealSense camera serial number
tcp_offset = None  # TCP offset configuration [x, y, z, roll, pitch, yaw]

# Chessboard parameters - loaded from config
BOARD_SIZE = None  # Will be loaded from config
CHESSBOARD_SIZE_MM = None  # Will be loaded from config
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
config_data = None  # Store loaded config

logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)

def load_config(mode=None):
    """Load calibration configuration from YAML file

    Args:
        mode: Calibration mode ('eye_in_hand' or 'eye_to_hand').
              If None, will try to load from global calibration_mode or config file.

    Returns:
        dict: Configuration data or None if failed
    """
    global BOARD_SIZE, CHESSBOARD_SIZE_MM, tcp_offset, config_data, calibration_mode

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config", "calibration_config_xarm.yaml")

    if not os.path.exists(config_file):
        logger_.error(f"配置文件不存在: {config_file}")
        return None

    try:
        with open(config_file, 'r') as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)

        # Determine which mode to use
        if mode is None:
            mode = calibration_mode  # Use global if set
        if mode is None:
            # Try to get from config file
            mode = config_data.get('calibration_mode', 'eye_in_hand')

        # Load chessboard parameters from mode-specific section
        if mode in config_data and 'chessboard' in config_data[mode]:
            # New structure: chessboard is under each mode
            BOARD_SIZE = tuple(config_data[mode]['chessboard']['board_size'])
            CHESSBOARD_SIZE_MM = config_data[mode]['chessboard']['square_size_mm']
            logger_.info(f"从 {mode} 配置中加载棋盘格参数")
        else:
            # Fallback: try old structure (top-level chessboard)
            if 'chessboard' in config_data:
                BOARD_SIZE = tuple(config_data['chessboard']['board_size'])
                CHESSBOARD_SIZE_MM = config_data['chessboard']['square_size_mm']
                logger_.warning(f"使用旧版配置结构（顶级 chessboard）")
            else:
                # Default values
                logger_.error(f"未找到 {mode} 模式的棋盘格配置")
                BOARD_SIZE = (6, 4)
                CHESSBOARD_SIZE_MM = 50.0

        # Load TCP offset if available
        if 'robot' in config_data and 'tcp_offset' in config_data['robot']:
            tcp_offset = config_data['robot']['tcp_offset']
            logger_.info(f"配置加载成功:")
            logger_.info(f"  - 棋盘格尺寸: {BOARD_SIZE}")
            logger_.info(f"  - 方格大小: {CHESSBOARD_SIZE_MM}mm")
            logger_.info(f"  - TCP偏移: {tcp_offset} (mm, mm, mm, deg, deg, deg)")
        else:
            tcp_offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            logger_.info(f"配置加载成功:")
            logger_.info(f"  - 棋盘格尺寸: {BOARD_SIZE}")
            logger_.info(f"  - 方格大小: {CHESSBOARD_SIZE_MM}mm")
            logger_.warning(f"  - 未找到TCP偏移配置，使用默认值: [0, 0, 0, 0, 0, 0]")

        return config_data

    except Exception as e:
        logger_.error(f"加载配置文件失败: {e}")
        return None

def detect_chessboard(image):
    """Detect chessboard in image and refine corners

    Args:
        image: Color image (BGR format)

    Returns:
        tuple: (success, corners) where corners is refined corner positions
    """
    if image is None:
        return False, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners with enhanced flags
    ret, corners = cv2.findChessboardCorners(
        gray, BOARD_SIZE,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
    )

    if ret:
        # Refine corner positions to subpixel accuracy
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)

    return ret, corners


def callback(frame, xarm):

    scaling_factor = 1.0
    global count, collected_frames_data

    # Keep original frame for saving
    original_frame = frame.copy()

    # Scale only for display
    cv_img = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # Detect chessboard for real-time visualization (on scaled image)
    display_img = cv_img.copy()
    ret, corners = detect_chessboard(original_frame)

    if ret:
        # Draw chessboard corners
        cv2.drawChessboardCorners(display_img, BOARD_SIZE, corners, ret)
        cv2.putText(display_img, "Chessboard: DETECTED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        cv2.putText(display_img, "Chessboard: NOT DETECTED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Display collected count
    cv2.putText(display_img, f"Collected: {count-1}", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Display instructions
    cv2.putText(display_img, "[S] Save  [ESC] Finish", (10, display_img.shape[0]-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Capture_Video", display_img)  # 窗口显示，显示名为 Capture_Video

    k = cv2.waitKey(30) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧

    if k == ord('s'):  # 若检测到按键 's'，打印字符串
        state,pose = get_pos(xarm)
        logger_.info(f'获取状态：{"成功" if state else "失败"}，{f"当前位姿为{pose}" if state else None}')
        if state:
            # Save original image (not scaled) with new naming format: frame_id_Color.png
            image_path = os.path.join(data_path, f"{count}_Color.png")
            cv2.imwrite(image_path, original_frame)

            # Detect chessboard on the original image (not scaled)
            chessboard_ret, chessboard_corners = detect_chessboard(original_frame)

            if chessboard_ret:
                logger_.info(f"===采集第{count}次数据！棋盘格检测成功")
            else:
                logger_.info(f"===采集第{count}次数据！棋盘格未检测到")

            # Store frame metadata (pose already in m and rad from get_pos)
            frame_data = {
                'frame_id': count,
                'pose': pose,  # [x, y, z, roll, pitch, yaw] in meters and radians
                'chessboard_detected': chessboard_ret
            }

            # Add corners if detected
            if chessboard_ret and chessboard_corners is not None:
                frame_data['corners'] = chessboard_corners.reshape(-1, 2).tolist()

            collected_frames_data.append(frame_data)

        count += 1

    elif k == 27:  # ESC key to finish
        return True  # Signal to stop collection

    return False  # Continue collection

def set_tcp_offset(xarm):
    """Set TCP offset to xArm robot

    Args:
        xarm: xArm robot client

    Returns:
        bool: True if successful, False otherwise
    """
    global tcp_offset

    if tcp_offset is None:
        logger_.warning("TCP偏移未配置，跳过设置")
        return False

    try:
        # xArm API: set_tcp_offset(offset, is_radian=None, wait=True)
        # offset: list of 6 values [x, y, z, roll, pitch, yaw]
        # Units: mm, mm, mm, deg, deg, deg (when is_radian=False or None)
        # Note: wait=False to avoid blocking, set_tcp_offset doesn't need to wait
        code = xarm.set_tcp_offset(tcp_offset, is_radian=False, wait=False)

        if code != 0:
            logger_.error(f"设置TCP偏移失败，错误码: {code}")
            return False

        # Give the robot time to apply the TCP offset
        time.sleep(0.5)

        logger_.info(f"✅ TCP偏移已设置: {tcp_offset}")

        # Verify the setting by reading the property
        current_offset = xarm.tcp_offset
        logger_.info(f"   当前TCP偏移: {current_offset}")

        return True

    except Exception as e:
        logger_.error(f"设置TCP偏移异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_pos(xarm):
    """Get robot arm position

    Returns:
        tuple: (success, pose) where pose is [x(m), y(m), z(m), roll(rad), pitch(rad), yaw(rad)]
    """
    code, pos = xarm.get_position()

    if code != 0:
        logger_.error(f"获取位姿失败，错误码: {code}")
        return False, None

    if not pos or len(pos) < 6:
        logger_.error(f"位姿数据无效: {pos}")
        return False, None

    # Convert from xArm units (mm, degrees) to standard units (m, radians)
    x, y, z, roll, pitch, yaw = pos
    x = x / 1000.0  # mm -> m
    y = y / 1000.0  # mm -> m
    z = z / 1000.0  # mm -> m
    roll = roll * math.pi / 180.0  # deg -> rad
    pitch = pitch * math.pi / 180.0  # deg -> rad
    yaw = yaw * math.pi / 180.0  # deg -> rad

    return True, [x, y, z, roll, pitch, yaw]
#
def save_calibration_metadata():
    """Save all calibration metadata after collection completes"""
    global data_path, calibration_mode, collected_frames_data, camera_serial

    if not collected_frames_data:
        logger_.info("没有采集到数据，不保存元数据")
        return

    # Save poses.txt in piper format
    poses_file = os.path.join(data_path, "poses.txt")
    with open(poses_file, 'w') as f:
        f.write("# Trajectory for replay - Robot arm end-effector poses\n")
        f.write("# Format: frame_id roll(rad) pitch(rad) yaw(rad) x(m) y(m) z(m)\n")
        timestamp = os.path.basename(data_path).replace('manual_calibration_eyeinhand_', '').replace('manual_calibration_eyetohand_', '')
        f.write(f"# Collection time: {timestamp}\n")
        f.write(f"# Collection mode: manual\n")
        f.write("#" + "-"*70 + "\n")
        for data in collected_frames_data:
            frame_id = data['frame_id']
            pose = data['pose']  # [x, y, z, roll, pitch, yaw] in meters and radians
            # Write in piper format: frame_id roll pitch yaw x y z
            f.write(f"{frame_id} {pose[3]:.6f} {pose[4]:.6f} {pose[5]:.6f} "
                   f"{pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f}\n")

    logger_.info(f"已保存 poses.txt: {len(collected_frames_data)} 个位姿")

    # Save calibration_data.json
    data_file = os.path.join(data_path, "calibration_data.json")
    with open(data_file, 'w') as f:
        json.dump(collected_frames_data, f, indent=4)

    logger_.info(f"已保存 calibration_data.json: {len(collected_frames_data)} 帧数据")

    # Save calibration_metadata.json
    timestamp = os.path.basename(data_path).replace('manual_calibration_eyeinhand_', '').replace('manual_calibration_eyetohand_', '')
    metadata = {
        'calibration_mode': calibration_mode,
        'collection_mode': 'manual',
        'camera_id': camera_serial if camera_serial else 'unknown',
        'timestamp': timestamp,
        'version': '3.5.2'  # 改进 replay 模式错误处理和运动规划
    }

    metadata_file = os.path.join(data_path, "calibration_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    logger_.info(f"已保存 calibration_metadata.json")
    logger_.info(f"  - 相机ID: {camera_serial if camera_serial else 'unknown'}")


def save_camera_intrinsics(pipeline):
    """Extract and save camera intrinsics to YAML file and get camera serial number"""
    global data_path, camera_serial

    try:
        profile = pipeline.get_active_profile()

        # Get camera serial number
        device = profile.get_device()
        camera_serial = device.get_info(rs.camera_info.serial_number)
        logger_.info(f"📹 相机序列号: {camera_serial}")

        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float32)

        dist_coeffs = np.array(intrinsics.coeffs[:5], dtype=np.float32)

        # Save to YAML file
        intrinsics_file = os.path.join(data_path, "realsense_intrinsics.yaml")
        CameraIntrinsicsManager.save_to_file(camera_matrix, dist_coeffs, intrinsics_file)

        logger_.info(f"已保存相机内参到 realsense_intrinsics.yaml")
        logger_.info(f"  - fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
        logger_.info(f"  - cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")

    except Exception as e:
        logger_.error(f"保存相机内参失败: {e}")
        camera_serial = None


def display_realsense(xarm, pipeline):
    """Display RealSense camera stream and collect calibration data

    Args:
        xarm: xArm robot client
        pipeline: RealSense pipeline (already started)
    """
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Create resizable window
    cv2.namedWindow("Capture_Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Capture_Video", 1280, 720)  # Set initial size

    global count
    count = 1

    logger_.info(f"开始手眼标定程序，当前程序版号V1.0.0")
    logger_.info(f"操作说明：")
    logger_.info(f"  - 按 S 键采集当前帧")
    logger_.info(f"  - 按 ESC 键结束采集")
    logger_.info(f"  - 窗口支持鼠标拖拽调整大小")

    try:
        should_stop = False
        while not should_stop:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            should_stop = callback(color_image, xarm)

    except KeyboardInterrupt:
        logger_.info("\n用户中断采集")

    finally:
        # Save all metadata before exiting
        save_calibration_metadata()

        cv2.destroyAllWindows()

        logger_.info(f"\n数据采集完成，共采集 {len(collected_frames_data)} 帧")
        if collected_frames_data:
            chessboard_count = sum(1 for d in collected_frames_data if d.get('chessboard_detected', False))
            logger_.info(f"  - 成功检测到棋盘格: {chessboard_count}/{len(collected_frames_data)} 帧")
            logger_.info(f"  - 数据保存路径: {data_path}")


def load_trajectory(trajectory_file):
    """Load trajectory from poses.txt file

    Args:
        trajectory_file: Path to poses.txt

    Returns:
        list: List of poses [x, y, z, roll, pitch, yaw] in meters and radians
    """
    if not os.path.exists(trajectory_file):
        logger_.error(f"轨迹文件不存在: {trajectory_file}")
        return []

    poses = []

    try:
        with open(trajectory_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Format: frame_id roll(rad) pitch(rad) yaw(rad) x(m) y(m) z(m)
                parts = line.split()

                if len(parts) != 7:
                    logger_.warning(f"无效的行格式 (期望7个字段): {line}")
                    continue

                frame_id = int(parts[0])
                roll_rad = float(parts[1])
                pitch_rad = float(parts[2])
                yaw_rad = float(parts[3])
                x_m = float(parts[4])
                y_m = float(parts[5])
                z_m = float(parts[6])

                # Return as [x, y, z, roll, pitch, yaw]
                pose = [x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]
                poses.append(pose)

        if not poses:
            logger_.error("轨迹文件中没有找到有效的位姿")

        return poses

    except Exception as e:
        logger_.error(f"加载轨迹文件失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def replay_trajectory(xarm, pipeline, trajectory_file, data_path):
    """Replay trajectory and collect calibration data

    Args:
        xarm: xArm robot client
        pipeline: RealSense pipeline
        trajectory_file: Path to trajectory file (poses.txt)
        data_path: Directory to save collected data
    """
    logger_.info("\n" + "="*60)
    logger_.info("重放轨迹采集模式")
    logger_.info("="*60)

    # Load trajectory
    poses = load_trajectory(trajectory_file)
    if not poses:
        logger_.error("加载轨迹失败")
        return

    logger_.info(f"从轨迹文件加载了 {len(poses)} 个位姿")

    # Create resizable window
    cv2.namedWindow("Replay_Mode", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Replay_Mode", 1280, 720)

    global collected_frames_data
    collected_frames_data = []

    # Motion config
    stability_wait = 3.0  # seconds to wait for stability (increased for better settling)
    capture_speed = 20  # movement speed (slower for better accuracy)

    # Configure robot for safer motion planning
    #logger_.info("配置机械臂运动参数...")
    #xarm.set_tcp_jerk(1000)  # 降低加加速度，使运动更平滑
    #xarm.set_tcp_maxacc(500)  # 降低最大加速度
    #logger_.info("运动参数配置完成")

    # Check and clear any existing errors before starting
    logger_.info("\n检查并清理机械臂状态...")
    xarm.clean_error()
    xarm.clean_warn()
    time.sleep(0.5)

    # Get and display current state
    code, state = xarm.get_state()
    logger_.info(f"当前状态码: {state}")

    # Ensure robot is in motion state
    if state != 0:
        logger_.info("重新设置机械臂为运动状态...")
        xarm.set_mode(0)
        xarm.set_state(0)
        time.sleep(0.5)

    try:
        for idx, pose_data in enumerate(poses):
            logger_.info(f"\n[{idx+1}/{len(poses)}] 移动到位姿...")

            # Unpack pose (in meters and radians from file)
            x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad = pose_data

            # Convert to mm and degrees (xArm API units)
            x_mm = x_m * 1000.0
            y_mm = y_m * 1000.0
            z_mm = z_m * 1000.0
            roll_deg = np.degrees(roll_rad)
            pitch_deg = np.degrees(pitch_rad)
            yaw_deg = np.degrees(yaw_rad)

            logger_.info(f"  目标: X={x_mm:.1f}mm Y={y_mm:.1f}mm Z={z_mm:.1f}mm")
            logger_.info(f"        Roll={roll_deg:.1f}° Pitch={pitch_deg:.1f}° Yaw={yaw_deg:.1f}°")

            # Move to target pose
            # Try motion_type=1 first (linear with fallback), then motion_type=2 (pure joint) if fails
            code = xarm.set_position(
                x=x_mm, y=y_mm, z=z_mm,
                roll=roll_deg, pitch=pitch_deg, yaw=yaw_deg,
                wait=True, speed=capture_speed,
                motion_type=0
            )
            
            if code != 0:
                logger_.warning(f"  移动到目标位置失败，错误码: {code}")
                if code == 9:
                    logger_.warning(f"  错误原因: 目标位置超出工作空间范围")
                    logger_.warning(f"  问题位置: X={x_mm:.1f} Y={y_mm:.1f} Z={z_mm:.1f}")
                    logger_.warning(f"  姿态角度: Roll={roll_deg:.1f}° Pitch={pitch_deg:.1f}° Yaw={yaw_deg:.1f}°")

                logger_.info(f"  清理机械臂错误状态...")
                xarm.clean_error()
                xarm.clean_warn()
                time.sleep(0.5)

                # Re-enable motion (error may put robot in stop state)
                logger_.info(f"  重新激活机械臂运动状态...")
                xarm.set_mode(0)  # Position control mode
                xarm.set_state(0)  # Sport state
                time.sleep(0.5)

                logger_.info(f"  跳过当前pose，继续执行下一个")
                continue

            # Wait for stability
            logger_.info(f"  等待 {stability_wait}s 以确保稳定...")
            time.sleep(stability_wait)

            # Get actual position
            success, actual_pose = get_pos(xarm)
            if not success:
                logger_.warning("  无法获取实际位置")
                logger_.info(f"  清理机械臂错误状态...")
                xarm.clean_error()
                xarm.clean_warn()
                time.sleep(0.5)

                # Re-enable motion (error may put robot in stop state)
                logger_.info(f"  重新激活机械臂运动状态...")
                xarm.set_mode(0)  # Position control mode
                xarm.set_state(0)  # Sport state
                time.sleep(0.5)

                logger_.info(f"  跳过当前pose，继续执行下一个")
                continue

            # Check position error (convert actual_pose from m to mm for comparison)
            pos_error = [
                abs(actual_pose[0] * 1000.0 - x_mm),
                abs(actual_pose[1] * 1000.0 - y_mm),
                abs(actual_pose[2] * 1000.0 - z_mm)
            ]
            max_error = max(pos_error)

            if max_error > 2.0:
                logger_.warning(f"  位置误差 {max_error:.2f}mm > 2mm")
            else:
                logger_.info(f"  到达目标位置 (误差 < 2mm)")

            # Capture image from camera - flush buffer to get latest frame
            logger_.info("  清空相机缓冲区，获取最新帧...")
            for _ in range(5):  # Discard old frames from buffer
                pipeline.wait_for_frames()

            frames = pipeline.wait_for_frames()  # Get fresh frame
            color_frame = frames.get_color_frame()
            if not color_frame:
                logger_.warning("  无法获取图像帧")
                continue

            original_frame = np.asanyarray(color_frame.get_data())

            # Scale for display
            scaling_factor = 1.0
            display_img = cv2.resize(original_frame, None, fx=scaling_factor, fy=scaling_factor,
                                    interpolation=cv2.INTER_AREA)

            # Detect chessboard on original image
            chessboard_ret, chessboard_corners = detect_chessboard(original_frame)

            # Draw on display image (scaled)
            if chessboard_ret:
                # Scale corners for display
                corners_scaled = chessboard_corners * scaling_factor
                cv2.drawChessboardCorners(display_img, BOARD_SIZE, corners_scaled, True)
                cv2.putText(display_img, "DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                cv2.putText(display_img, "NOT DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # Display progress
            cv2.putText(display_img, f"Replay: {idx+1}/{len(poses)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display_img, "[ESC] Stop", (10, display_img.shape[0]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Replay_Mode", display_img)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                logger_.info("\n用户中断采集")
                break

            # Save image (original, not scaled)
            frame_id = idx + 1
            image_path = os.path.join(data_path, f"{frame_id}_Color.png")
            cv2.imwrite(image_path, original_frame)

            # actual_pose is already in meters and radians from get_pos()
            # No need to convert again

            # Store metadata
            frame_data = {
                'frame_id': frame_id,
                'pose': actual_pose,  # Already in m and rad
                'chessboard_detected': chessboard_ret
            }

            if chessboard_ret and chessboard_corners is not None:
                frame_data['corners'] = chessboard_corners.reshape(-1, 2).tolist()

            collected_frames_data.append(frame_data)

            if chessboard_ret:
                logger_.info(f"  ✅ 已采集第 {frame_id} 帧数据 (棋盘格检测成功)")
            else:
                logger_.info(f"  ⚠️  已采集第 {frame_id} 帧数据 (棋盘格未检测到)")

        cv2.destroyAllWindows()

        # Save metadata
        save_calibration_metadata()

        logger_.info(f"\n数据采集完成，共采集 {len(collected_frames_data)} 帧")
        if collected_frames_data:
            chessboard_count = sum(1 for d in collected_frames_data if d.get('chessboard_detected', False))
            logger_.info(f"  - 成功检测到棋盘格: {chessboard_count}/{len(collected_frames_data)} 帧")
            logger_.info(f"  - 数据保存路径: {data_path}")

    except KeyboardInterrupt:
        logger_.info("\n用户中断采集")
        cv2.destroyAllWindows()
    except Exception as e:
        logger_.error(f"重放轨迹失败: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    print("="*60)
    print("Hand-Eye Calibration Data Collector for xArm v3.4.1")
    print("="*60)

    # Load configuration first
    if load_config() is None:
        print("错误: 无法加载配置文件")
        sys.exit(1)

    # 选择采集模式
    print("\n采集模式:")
    print("  1. Manual (手动采集)")
    print("  2. Replay (重放轨迹)")
    collection_mode = input("选择采集模式 (1/2) [默认 1]: ").strip()

    # 选择标定模式
    print("\n标定模式:")
    print("  1. Eye-in-Hand (相机在末端，棋盘格固定)")
    print("  2. Eye-to-Hand (相机固定，棋盘格在末端)")
    mode_choice = input("选择标定模式 (1/2) [默认 1]: ").strip()

    if mode_choice == "2":
        calibration_mode = "eye_to_hand"
        print("\n📷 模式: Eye-to-Hand")
        print("   - 相机固定在基座附近")
        print("   - 棋盘格固定在机械臂末端")
    else:
        calibration_mode = "eye_in_hand"
        print("\n📷 模式: Eye-in-Hand")
        print("   - 相机固定在机械臂末端")
        print("   - 棋盘格固定在外部")

    # Reload configuration with selected mode to get correct chessboard parameters
    if load_config(calibration_mode) is None:
        print("错误: 无法加载配置文件")
        sys.exit(1)

    print("="*60)

    # Create calibration data directory with mode suffix
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_data_base = os.path.join(script_dir, "calibration_data")
    os.makedirs(calibration_data_base, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    calibration_mode_short = calibration_mode.replace('_', '')  # eyeinhand / eyetohand

    # Determine collection mode prefix
    if collection_mode == "2":
        mode_prefix = "replay_calibration"
    else:
        mode_prefix = "manual_calibration"

    data_path = os.path.join(
        calibration_data_base,
        f"{mode_prefix}_{calibration_mode_short}_{timestamp}"
    )
    os.makedirs(data_path, exist_ok=True)

    logger_.info(f"数据将保存到: {data_path}")

    # xArm connection
    xarm_ip = '192.168.1.236'
    logger_.info(f'xarm_ip: {xarm_ip}')

    try:
        xarm = XArmAPI(xarm_ip)
        logger_.info("机械臂连接成功")

        # Enable motion
        code = xarm.motion_enable(enable=True)
        if code != 0:
            logger_.error(f"使能机械臂失败，错误码: {code}")
            popup_message("错误", f"使能机械臂失败，错误码: {code}")
            sys.exit(1)
        logger_.info("机械臂已使能")

        # Set mode: 0=position control, 1=servo motion, 2=joint teaching
        code = xarm.set_mode(0)  # Position control mode
        if code != 0:
            logger_.warning(f"设置模式失败，错误码: {code}")

        # Set state: 0=sport, 3=pause, 4=stop
        code = xarm.set_state(0)  # Sport state
        if code != 0:
            logger_.warning(f"设置状态失败，错误码: {code}")

        logger_.info("机械臂初始化完成")

        

        # Re-enable motion after TCP offset (setting TCP may change robot state)
        logger_.info("\n重新确认机械臂状态...")
        code = xarm.set_mode(0)  # Position control mode
        if code != 0:
            logger_.warning(f"重新设置模式失败，错误码: {code}")

        code = xarm.set_state(0)  # Sport state
        if code != 0:
            logger_.warning(f"重新设置状态失败，错误码: {code}")

        # Set TCP offset if configured
        logger_.info("\n" + "="*60)
        logger_.info("设置TCP偏移")
        logger_.info("="*60)
        set_tcp_offset(xarm)
        
        logger_.info("✅ 机械臂准备就绪")

    except Exception as e:
        logger_.error(f"机械臂连接失败: {e}")
        popup_message("错误", f"机械臂连接失败: {e}")
        sys.exit(1)

    # Initialize camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    try:
        pipeline.start(config)
    except Exception as e:
        logger_.error(f"相机连接异常：{e}")
        popup_message("提醒", "相机连接异常")
        sys.exit(1)

    # Save camera intrinsics
    save_camera_intrinsics(pipeline)

    # Choose collection mode
    if collection_mode == "2":
        # Replay mode
        print("\n" + "="*60)
        print("重放模式 - 选择轨迹文件")
        print("="*60)

        # List available trajectory files
        available_trajectories = []
        for item in os.listdir(calibration_data_base):
            item_path = os.path.join(calibration_data_base, item)
            if os.path.isdir(item_path):
                poses_file = os.path.join(item_path, "poses.txt")
                if os.path.exists(poses_file):
                    available_trajectories.append((item, poses_file))

        if not available_trajectories:
            print("错误: 没有找到可用的轨迹文件")
            print("请先使用手动模式采集数据")
            sys.exit(1)

        print("\n可用的轨迹文件:")
        for idx, (dirname, filepath) in enumerate(available_trajectories, 1):
            print(f"  {idx}. {dirname}")

        traj_choice = input(f"\n选择轨迹 (1-{len(available_trajectories)}): ").strip()
        try:
            traj_idx = int(traj_choice) - 1
            if 0 <= traj_idx < len(available_trajectories):
                selected_trajectory = available_trajectories[traj_idx][1]
                print(f"已选择: {available_trajectories[traj_idx][0]}")
            else:
                print("无效选择")
                sys.exit(1)
        except ValueError:
            print("无效输入")
            sys.exit(1)

        # Start replay
        replay_trajectory(xarm, pipeline, selected_trajectory, data_path)

    else:
        # Manual mode
        display_realsense(xarm, pipeline)

    # Cleanup
    pipeline.stop()
