#!/usr/bin/env python3
"""
Camera Info Publisher - 从标定文件发布相机内参
覆盖 RealSense 驱动的默认内参，使用精确标定的参数
"""

import rospy
import yaml
from sensor_msgs.msg import CameraInfo

class CameraInfoPublisher:
    def __init__(self):
        rospy.init_node('camera_info_publisher', anonymous=False)

        # 参数
        self.camera_name = rospy.get_param('~camera_name', 'camera/top')
        self.intrinsics_file = rospy.get_param('~intrinsics_file')
        self.image_width = rospy.get_param('~image_width', 640)
        self.image_height = rospy.get_param('~image_height', 480)
        self.publish_rate = rospy.get_param('~publish_rate', 30.0)

        # 读取内参
        self.camera_info = self.load_intrinsics()

        # 发布器
        self.pub = rospy.Publisher(
            f'/{self.camera_name}/color/camera_info_calibrated',
            CameraInfo,
            queue_size=10,
            latch=True  # 保持最后一条消息
        )

        rospy.loginfo(f"[CameraInfoPublisher] Publishing calibrated camera info for {self.camera_name}")
        rospy.loginfo(f"[CameraInfoPublisher] Resolution: {self.image_width}x{self.image_height}")
        rospy.loginfo(f"[CameraInfoPublisher] fx={self.camera_info.K[0]:.2f}, fy={self.camera_info.K[4]:.2f}")
        rospy.loginfo(f"[CameraInfoPublisher] cx={self.camera_info.K[2]:.2f}, cy={self.camera_info.K[5]:.2f}")

    def load_intrinsics(self):
        """从 YAML 文件加载内参 - 支持多种格式"""
        with open(self.intrinsics_file, 'r') as f:
            content = f.read()

            # 移除第一行的 YAML 版本指令
            if content.startswith('%YAML'):
                content = '\n'.join(content.split('\n')[1:])

            # 方法1：尝试按 model_type 分割（多个标定在一个文件中）
            sections = content.split('\nmodel_type:')
            calib_data_list = []

            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                # 重新添加 model_type 行（除了第一个）
                if i > 0:
                    section = 'model_type:' + section
                # 移除开头的 ---
                section = section.replace('---\n', '').strip()
                try:
                    data = yaml.safe_load(section)
                    if data:
                        calib_data_list.append(data)
                except Exception as e:
                    rospy.logwarn(f"Failed to parse section {i}: {e}")
                    continue

            rospy.loginfo(f"Found {len(calib_data_list)} calibration entries in file")

            # 查找匹配分辨率的标定数据
            for data in calib_data_list:
                width = data.get('image_width')
                height = data.get('image_height')
                if width == self.image_width and height == self.image_height:
                    rospy.loginfo(f"✓ Found matching calibration: {width}x{height}")
                    return self.create_camera_info(data)

            # 如果没找到匹配的，使用第一个
            rospy.logwarn(f"No calibration found for {self.image_width}x{self.image_height}")
            rospy.logwarn(f"Available resolutions: {[(d.get('image_width'), d.get('image_height')) for d in calib_data_list]}")
            if calib_data_list:
                rospy.logwarn(f"Using first entry: {calib_data_list[0].get('image_width')}x{calib_data_list[0].get('image_height')}")
                return self.create_camera_info(calib_data_list[0])
            else:
                rospy.logerr("No calibration data found in file!")
                raise RuntimeError("Empty calibration file")

    def create_camera_info(self, calib_data):
        """创建 CameraInfo 消息"""
        msg = CameraInfo()
        msg.header.frame_id = f"{self.camera_name}_color_optical_frame"
        msg.width = self.image_width
        msg.height = self.image_height

        # 投影参数
        fx = calib_data['projection_parameters']['fx']
        fy = calib_data['projection_parameters']['fy']
        cx = calib_data['projection_parameters']['cx']
        cy = calib_data['projection_parameters']['cy']

        # 畸变参数
        k1 = calib_data['distortion_parameters']['k1']
        k2 = calib_data['distortion_parameters']['k2']
        p1 = calib_data['distortion_parameters']['p1']
        p2 = calib_data['distortion_parameters']['p2']

        # K 矩阵 (内参矩阵)
        msg.K = [fx,  0.0, cx,
                 0.0, fy,  cy,
                 0.0, 0.0, 1.0]

        # D 畸变系数 (plumb_bob 模型)
        msg.D = [k1, k2, p1, p2, 0.0]
        msg.distortion_model = "plumb_bob"

        # R 矩阵 (单目相机为单位矩阵)
        msg.R = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]

        # P 矩阵 (投影矩阵)
        msg.P = [fx,  0.0, cx,  0.0,
                 0.0, fy,  cy,  0.0,
                 0.0, 0.0, 1.0, 0.0]

        return msg

    def publish(self):
        """定期发布 camera_info"""
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            self.camera_info.header.stamp = rospy.Time.now()
            self.pub.publish(self.camera_info)
            rate.sleep()

if __name__ == '__main__':
    try:
        publisher = CameraInfoPublisher()
        publisher.publish()
    except rospy.ROSInterruptException:
        pass
