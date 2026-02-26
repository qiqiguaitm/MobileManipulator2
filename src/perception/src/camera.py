import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from mmengine.config import Config

logger = logging.getLogger(__name__)


class RealSenseCamera:
    """英特尔RealSense深度相机接口类
    
    负责管理RealSense D系列相机，提供RGB-D数据采集、
    深度对齐、3D点云反投影等功能。
    """
    def __init__(self, cfg):
        """初始化RealSense相机
        
        Args:
            cfg: 配置对象，包含：
                - device_id: 设备序列号
                - width/height: 图像分辨率
                - fps: 帧率
                - enable_depth: 是否启用深度流
                - use_calibrated_intrinsics: 是否使用标定文件内参(默认False)
                - intrinsics_file: 内参文件路径(可选)
        """
        self.device_id = cfg.device_id  # 相机序列号
        self.width = cfg.width  # 图像宽度（像素）
        self.height = cfg.height  # 图像高度（像素）
        self.fps = cfg.fps  # 采集帧率
        self.enable_depth = cfg.get('enable_depth', True)  # 是否启用深度
        self.use_calibrated_intrinsics = cfg.get('use_calibrated_intrinsics', True)  # 是否使用标定内参
        self.intrinsics_file = cfg.get('intrinsics_file', 'intrics_hand_camera.yaml')  # 内参文件路径
        self.pipeline = None  # RealSense管道对象
        self.scale = None  # 深度值缩放因子
        self.intrinsics = None  # 相机内参

    def _load_intrinsics_from_file(self, file_path, resolution):
        """从标定文件加载内参
        
        Args:
            file_path: YAML格式的内参文件路径
            resolution: 目标分辨率(width, height)
            
        Returns:
            rs.intrinsics对象或None
        """
        if not os.path.exists(file_path):
            logger.warning(f"Intrinsics file not found: {file_path}")
            return None
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # 解析YAML文件中对应分辨率的内参
            blocks = content.split('model_type: PINHOLE')
            
            for block in blocks[1:]:  # 跳过第一个空块
                lines = block.strip().split('\n')
                width = None
                height = None
                params = {}
                
                for line in lines:
                    if 'image_width:' in line:
                        width = int(line.split(':')[1].strip())
                    elif 'image_height:' in line:
                        height = int(line.split(':')[1].strip())
                    elif 'fx:' in line and 'fx' not in params:
                        params['fx'] = float(line.split(':')[1].strip())
                    elif 'fy:' in line and 'fy' not in params:
                        params['fy'] = float(line.split(':')[1].strip())
                    elif 'cx:' in line and 'cx' not in params:
                        params['cx'] = float(line.split(':')[1].strip())
                    elif 'cy:' in line and 'cy' not in params:
                        params['cy'] = float(line.split(':')[1].strip())
                    elif 'k1:' in line and 'k1' not in params:
                        params['k1'] = float(line.split(':')[1].strip())
                    elif 'k2:' in line and 'k2' not in params:
                        params['k2'] = float(line.split(':')[1].strip())
                    elif 'p1:' in line and 'p1' not in params:
                        params['p1'] = float(line.split(':')[1].strip())
                    elif 'p2:' in line and 'p2' not in params:
                        params['p2'] = float(line.split(':')[1].strip())
                
                # 检查是否匹配目标分辨率
                if width == resolution[0] and height == resolution[1]:
                    # 创建rs.intrinsics对象
                    intrinsics = rs.intrinsics()
                    intrinsics.width = width
                    intrinsics.height = height
                    intrinsics.ppx = params.get('cx', width/2)
                    intrinsics.ppy = params.get('cy', height/2)
                    intrinsics.fx = params.get('fx', 600)
                    intrinsics.fy = params.get('fy', 600)
                    
                    # 设置畸变系数
                    intrinsics.model = rs.distortion.brown_conrady
                    intrinsics.coeffs = [
                        params.get('k1', 0),
                        params.get('k2', 0),
                        params.get('p1', 0),
                        params.get('p2', 0),
                        0  # k3，通常为0
                    ]
                    
                    logger.info(f"Loaded intrinsics from file for {width}x{height}")
                    logger.info(f"  fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
                    logger.info(f"  cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
                    logger.info(f"  distortion: k1={intrinsics.coeffs[0]:.4f}, k2={intrinsics.coeffs[1]:.4f}")
                    
                    return intrinsics
            
            logger.warning(f"No matching intrinsics found for {resolution[0]}x{resolution[1]} in {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading intrinsics from file: {e}")
            return None
    
    def connect(self):
        """连接并启动相机
        
        配置相机参数，启动数据流，获取相机内参和深度缩放因子。
        """
        # 创建并配置管道
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(self.device_id))  # 指定设备
        
        # 配置深度流（可选）
        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        # 配置彩色流
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        
        try:
            cfg = self.pipeline.start(config)  # 启动流
        except RuntimeError as e:
            print(f"❌ Failed to start camera pipeline: {e}")
            raise
        
        # 重置相机设备以确保稳定状态（类似ROS系统的initial_reset）
        device = cfg.get_device()
        if device.supports(rs.camera_info.firmware_version):
            print(f"Camera firmware: {device.get_info(rs.camera_info.firmware_version)}")
        
        # 智能等待相机稳定 - 最多尝试30帧，但允许合理的失败
        stable_frames = 0
        max_attempts = 30
        timeout_ms = 1000  # 1秒超时，而不是默认的5秒
        
        for attempt in range(max_attempts):
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms)
                if frames:
                    stable_frames += 1
                    if stable_frames >= 5:  # 连续5帧成功就认为稳定
                        break
            except RuntimeError as e:
                if "Frame didn't arrive" in str(e):
                    print(f"⚠ Camera warming up... ({attempt+1}/{max_attempts})")
                    continue
                else:
                    print(f"❌ Camera connection failed: {e}")
                    raise
        
        if stable_frames < 5:
            print(f"⚠ Camera may be unstable (only got {stable_frames} frames)")
        
        # Determine intrinsics - 支持双模式
        if self.use_calibrated_intrinsics:
            # 尝试从文件加载内参
            loaded_intrinsics = self._load_intrinsics_from_file(
                self.intrinsics_file, 
                (self.width, self.height)
            )
            if loaded_intrinsics:
                self.intrinsics = loaded_intrinsics
                print(f"✅ Using calibrated intrinsics from {self.intrinsics_file}")
            else:
                # 回退到RealSense默认内参
                rgb_profile = cfg.get_stream(rs.stream.color)
                self.intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
                print(f"⚠ Failed to load calibrated intrinsics, using RealSense defaults")
        else:
            # 使用RealSense自动获取的内参
            rgb_profile = cfg.get_stream(rs.stream.color)
            self.intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
            print(f"ℹ Using RealSense auto-detected intrinsics")
        
        # 打印当前使用的内参信息
        print(f"Current intrinsics: fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}, "
              f"cx={self.intrinsics.ppx:.2f}, cy={self.intrinsics.ppy:.2f}")

        if self.enable_depth:
            dpt_profile = cfg.get_stream(rs.stream.depth)
            self.dpt_instrinsics = dpt_profile.as_video_stream_profile().get_intrinsics()
        else:
            self.dpt_instrinsics = None
        # Determine depth scale
        if self.enable_depth:
            self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()
        else:
            self.scale = None

    def get_image_rgb(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    
    def get_image_bundle(self):
        """获取对齐的RGB-D图像数据（默认使用原始简单版本）
        
        Returns:
            dict: 包含'rgb'和'depth'的字典
        """
        # 默认使用原始简单版本，保持向后兼容
        frames = self.pipeline.wait_for_frames()  # 等待新的帧

        # 将深度图对齐到彩色图
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.first(rs.stream.color)
        aligned_depth_frame = aligned_frames.get_depth_frame()

        # 转换为numpy数组，深度值从mm转换为m
        depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32)
        depth_image *= self.scale  # 应用深度缩放因子
        color_image = np.asanyarray(color_frame.get_data())

        return {
            'rgb': color_image,
            'depth': depth_image,
        }
    
    def get_image_bundle_enhanced(self, skip_frames=2, apply_filters=True):
        """获取对齐的RGB-D图像数据（增强版，确保图像清晰稳定）
        
        将深度图对齐到彩色图坐标系，返回同步的RGB和深度数据。
        添加了多种机制确保图像质量：
        1. 跳帧机制 - 丢弃缓冲区的旧帧，获取最新帧
        2. 时间滤波 - 减少深度噪声
        3. 空间滤波 - 平滑深度图
        4. 孔洞填充 - 填补深度图中的空洞
        
        Args:
            skip_frames: 跳过的帧数，确保获取最新图像
            apply_filters: 是否应用深度滤波器
            
        Returns:
            dict: 包含'rgb'和'depth'的字典
                - rgb: RGB图像 (H,W,3) uint8
                - depth: 深度图 (H,W) float32，单位米
        """
        # 跳过缓冲区中的旧帧，获取最新帧
        for _ in range(skip_frames):
            try:
                self.pipeline.wait_for_frames(timeout_ms=100)
            except:
                pass
        
        # 等待新的帧（带超时保护）
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        
        # 检查帧的有效性
        if not frames:
            raise RuntimeError("Failed to get valid frames from camera")

        # 将深度图对齐到彩色图
        if not hasattr(self, '_align'):
            self._align = rs.align(rs.stream.color)
        aligned_frames = self._align.process(frames)
        
        color_frame = aligned_frames.first(rs.stream.color)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        
        # 检查帧是否有效
        if not color_frame or not aligned_depth_frame:
            raise RuntimeError("Invalid color or depth frame")
        
        # 应用深度滤波器以提高质量
        if apply_filters and self.enable_depth:
            # 时间滤波器 - 减少时间噪声
            if not hasattr(self, '_temporal_filter'):
                self._temporal_filter = rs.temporal_filter()
                self._temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
                self._temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
            
            # 空间滤波器 - 平滑深度图
            if not hasattr(self, '_spatial_filter'):
                self._spatial_filter = rs.spatial_filter()
                self._spatial_filter.set_option(rs.option.filter_magnitude, 2)
                self._spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
                self._spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
            
            # 孔洞填充滤波器
            if not hasattr(self, '_hole_filling_filter'):
                self._hole_filling_filter = rs.hole_filling_filter()
            
            # 应用滤波器链
            filtered_depth = aligned_depth_frame
            filtered_depth = self._temporal_filter.process(filtered_depth)
            filtered_depth = self._spatial_filter.process(filtered_depth)
            filtered_depth = self._hole_filling_filter.process(filtered_depth)
            aligned_depth_frame = filtered_depth

        # 转换为numpy数组，深度值从mm转换为m
        depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32)
        depth_image *= self.scale  # 应用深度缩放因子
        color_image = np.asanyarray(color_frame.get_data())
        
        # 检查图像质量指标
        if color_image.mean() < 10:  # 图像过暗
            print("⚠ Warning: Image too dark, may affect detection quality")
        
        if self.enable_depth and np.isnan(depth_image).sum() > depth_image.size * 0.3:  # 深度图空洞过多
            print("⚠ Warning: Too many holes in depth image (>30%)")

        return {
            'rgb': color_image,
            'depth': depth_image,
        }

    def unprj(self, aff, rgb, depth):
        """将2D抓取点反投影到3D空间
        
        根据相机内参和深度值，将2D像素坐标转换为3D点坐标。
        
        Args:
            aff: 抓取点信息，包含'aff'(五参数)和'touching_points'(接触点)
            rgb: RGB图像
            depth: 深度图或深度帧
            
        Returns:
            dict: 包含3D点坐标、夹爪宽度等信息
        """
        aff5 = aff['aff']  # 五参数表示: (cx, cy, w, h, angle)
        tps = aff['touching_points']  # 夹爪接触点
        cx, cy, w, h, angle = aff5  # 解析抓取框参数

        # if cx >= depth.width or cy >= depth.height:
        #     print(f'{cx=} {cy=} out of image')
        #     return False, [0, 0, 0]
        if isinstance(depth, rs.depth_frame):
            depth = np.asarray(depth.get_data(), dtype=np.float32) / 1000.0  # from mm to m

        height, width = depth.shape
        if cx >= width and cy >= height:
            print(f'{cx=} {cy} out of depth image')
            return False, [0, 0, 0]
        center_depth = depth[int(cy), int(cx)]
        point3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [cx, cy], center_depth)
        # width3d = center_depth * w
        p1 = rs.rs2_deproject_pixel_to_point(self.intrinsics, tps[0], center_depth)
        p2 = rs.rs2_deproject_pixel_to_point(self.intrinsics, tps[1], center_depth)
        width3d = np.linalg.norm(np.array(p1) - np.array(p2))

        ret = dict()
        ret['width3d'] = width3d
        ret['point3d'] = point3d
        ret['status'] = True
        if center_depth < 1e-3 or center_depth > 2:
            print(f'no valid depth: {center_depth=} {cx=} {cy=}')
            ret['status'] = False
            # breakpoint()
        return ret

    def plot_image_bundle(self):
        images = self.get_image_bundle()

        rgb = images['rgb']
        depth = images['depth']

        fig, ax = plt.subplots(1, 2, squeeze=False)
        ax[0, 0].imshow(rgb)
        m, s = np.nanmean(depth), np.nanstd(depth)
        ax[0, 1].imshow(depth, vmin=m - s, vmax=m + s, cmap=plt.cm.gray)
        ax[0, 0].set_title('rgb')
        ax[0, 1].set_title('aligned_depth')

        plt.show()

    def plot_image_rgb(self):
        rgb = self.get_image_rgb()
        cv2.namedWindow('rgb', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('rgb', self.width, self.height)
        cv2.imshow('rgb', rgb)
        cv2.waitKey(1)

    def unprj_point(self, dpt, cx, cy):
        point3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [cx, cy], dpt)
        return point3d

    def unprj_points(self, dpts, ps):
        n = len(dpts)
        pp = []
        for i in range(n):
            point3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, ps[i], dpts[i])
            pp.append(point3d)
        return pp


if __name__ == '__main__':
    
    if 1:
        cfg = Config()
        cfg.device_id = 337122071190  # hand camera id 
        cfg.width = 1280 # 使用支持的分辨率
        cfg.height = 720
        cfg.fps = 6  # default 30
        cam = RealSenseCamera(cfg)
        cam.connect()
        while True:
            cam.plot_image_bundle()
    else:
        cfg = Config()
        cfg.device_id = 337122071540  # chassis camera id 
        cfg.enable_depth = True
        cfg.width = 1280
        cfg.height = 720
        cfg.fps = 6   # default 30
        cam = RealSenseCamera(cfg)
        cam.connect()
        while True:
            #cam.plot_image_rgb()
            cam.plot_image_bundle()
    
    #top camera id: 318122302992
