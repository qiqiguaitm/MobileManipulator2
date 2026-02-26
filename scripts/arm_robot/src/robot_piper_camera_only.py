import math
import os
import time
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from camera import RealSenseCamera
from percept import DinoXDetectorCloud
from mmengine.config import Config
from robot_piper import PiperRobot as ArmRobot
from scipy.spatial.transform import Rotation as R
from vis_cls import Visualizer
import pyrealsense2 as rs

class RobotPiperCam:
    def __init__(self, cfg, camera, arm, ref, vis=None, **kwargs):
        """简化机器人控制器 - 只保留ref模式核心功能"""
        self.cfg = cfg
        self.camera = camera
        self.arm = arm
        self.ref = ref
        self.vis = vis
        fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        self._font = ImageFont.truetype(fp, size=30)

        # 初始化数据存储目录
        self.frame_count = 0
        self.setup_data_storage()

    def draw_chinsese_text(self, rgb, pos, text):
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        img_w = pil_img.width
        bbox = draw.textbbox((0, 0), text, font=self._font)
        text_w = bbox[2] - bbox[0]
        x = (img_w - text_w) // 2
        y = pos[1]
        draw.text((x, y), text, font=self._font, fill='blue')
        return np.array(pil_img)

    def setup_data_storage(self):
        """初始化数据存储目录"""
        today = datetime.now().strftime("%Y%m%d")
        self.data_dir = os.path.join("samples", today)
        self.rgb_dir = os.path.join(self.data_dir, "rgb")
        self.depth_dir = os.path.join(self.data_dir, "depth")

        # 创建目录
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        print(f"数据存储目录: {self.data_dir}")

    def save_frame_data(self, rgb, depth):
        """保存RGB和深度数据"""
        try:
            timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]  # 精确到毫秒
            frame_id = f"{self.frame_count:06d}_{timestamp}"

            # 保存RGB图像 (BGR -> RGB for saving)
            rgb_save = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb_path = os.path.join(self.rgb_dir, f"{frame_id}.jpg")
            cv2.imwrite(rgb_path, rgb_save)

            # 保存深度图像 (16bit PNG)
            depth_path = os.path.join(self.depth_dir, f"{frame_id}.png")
            # 将深度值转换为16bit整数 (mm)
            depth_mm = (depth * 1000).astype(np.uint16)
            cv2.imwrite(depth_path, depth_mm)

            self.frame_count += 1

            if self.frame_count % 100 == 0:  # 每100帧打印一次
                print(f"已保存 {self.frame_count} 帧数据")

        except Exception as e:
            print(f"保存帧数据失败: {e}")

    def vis_frame(self, rgb, depth, state):
        """实时可视化 - 显示refer检测结果和3D信息，包括深度图"""
        targets = state['targets']
        cmd = state['cmd']
        best_target = state['best_target']
        point3d = state['point3d']
        offset = state['offset']
        eef_in_base = state['eef_in_base']
        point_in_base = state['point_in_base']

        # 绘制所有目标框
        if targets:
            for target in targets:
                bbox = target['bbox']
                bbox = [int(x) for x in bbox]
                cv2.rectangle(rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), [0, 0, 255], 2)
                # 显示类别和得分
                category = target.get('category', 'unknown')
                score = target.get('score', 0)
                cv2.putText(rgb, f'{category}: {score:.2f}',
                          (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 显示命令文本
        if cmd:
            rgb = self.draw_chinsese_text(rgb, pos=(100, 680), text=cmd)

        # 显示最高概率目标的中心点和3D信息
        text_y = 20
        if best_target is not None:
            bbox = best_target['bbox']
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            cv2.circle(rgb, (cx, cy), 8, [0, 255, 0], 3)
            cv2.rectangle(rgb, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [0, 255, 0], 3)

            # 显示2D中心点坐标
            cv2.putText(rgb, f'2D Center: ({cx}, {cy})', (100, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            text_y += 25

            # 显示3D信息
            if point3d is not None:
                cv2.putText(rgb, f'3D in Cam: {[round(x * 1000) for x in point3d]} mm',
                          (100, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                text_y += 25
            if point_in_base is not None:
                cv2.putText(rgb, f'3D in Base: {[round(x * 1000) for x in point_in_base]} mm',
                          (100, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                text_y += 25
            if offset is not None:
                cv2.putText(rgb, f'offset: {[round(x * 1000) for x in offset]} mm',
                          (100, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                text_y += 25
            if eef_in_base is not None:
                cv2.putText(rgb, f'eef: {[round(x * 1000) for x in eef_in_base]} mm',
                          (100, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                text_y += 25

            category = best_target.get('category', 'unknown')
            score = best_target.get('score', 0)
            cv2.putText(rgb, f'Best: {category} ({score:.3f})', (100, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            text_y += 25
        else:
            cv2.putText(rgb, 'no target detected', (100, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            text_y += 25

        cv2.putText(rgb, 'mode: ref', (100, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 创建彩色深度图
        depth_colormap = self.create_depth_colormap(depth)

        # 在深度图上也标记检测结果
        if best_target is not None:
            bbox = best_target['bbox']
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            cv2.circle(depth_colormap, (cx, cy), 8, [0, 255, 0], 3)
            cv2.rectangle(depth_colormap, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [0, 255, 0], 3)

            # 在深度图上显示深度值
            if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                depth_value = depth[cy, cx]
                cv2.putText(depth_colormap, f'Depth: {depth_value:.3f}m',
                          (cx-50, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 添加深度图颜色条说明
        depth_colormap = self.add_depth_colorbar(depth_colormap, depth)

        # 纵向拼接RGB和深度图
        combined = np.vstack((rgb, depth_colormap))

        return combined

    def create_depth_colormap(self, depth):
        """创建彩色深度图 - 放大0.2-0.5m区间的颜色差异"""
        # 复制深度图避免修改原始数据
        depth_vis = depth.copy()

        # 定义三个深度区间和对应的颜色映射范围
        # 0.0-0.2m -> 0-50 (蓝色区域)
        # 0.2-0.5m -> 50-200 (主要工作区间，绿-黄-橙)
        # 0.5-2.0m -> 200-255 (红色区域)

        depth_normalized = np.zeros_like(depth_vis, dtype=np.uint8)

        # 第一区间：0-0.2m -> 映射到0-50
        mask1 = (depth_vis > 0) & (depth_vis <= 0.2)
        depth_normalized[mask1] = (depth_vis[mask1] / 0.2 * 50).astype(np.uint8)

        # 第二区间：0.2-0.5m -> 映射到50-200（放大的主要工作区间）
        mask2 = (depth_vis > 0.2) & (depth_vis <= 0.5)
        depth_normalized[mask2] = ((depth_vis[mask2] - 0.2) / 0.3 * 150 + 50).astype(np.uint8)

        # 第三区间：0.5-2.0m -> 映射到200-255
        mask3 = depth_vis > 0.5
        depth_vis_clipped = np.clip(depth_vis[mask3], 0.5, 2.0)
        depth_normalized[mask3] = ((depth_vis_clipped - 0.5) / 1.5 * 55 + 200).astype(np.uint8)

        # 应用JET颜色映射
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # 将无效深度区域设为黑色
        depth_colormap[depth <= 0] = [0, 0, 0]

        return depth_colormap

    def add_depth_colorbar(self, depth_colormap, depth):
        """在深度图内部添加颜色条说明，不改变图像尺寸"""
        h, w = depth_colormap.shape[:2]

        # 在深度图右侧绘制颜色条（内嵌方式）
        colorbar_width = 30
        colorbar_x = w - colorbar_width - 10  # 距离右边缘10像素
        colorbar_height = h // 2
        colorbar_y = h // 4

        # 在深度图上绘制颜色条背景
        cv2.rectangle(depth_colormap, (colorbar_x, colorbar_y),
                     (colorbar_x + colorbar_width, colorbar_y + colorbar_height),
                     (0, 0, 0), -1)

        # 填充颜色条
        for i in range(colorbar_height):
            color_val = int(255 * (1 - i / colorbar_height))  # 从上到下：红到蓝
            color = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
            color = color.tolist()
            cv2.rectangle(depth_colormap,
                         (colorbar_x + 2, colorbar_y + i),
                         (colorbar_x + colorbar_width - 2, colorbar_y + i + 1),
                         color, -1)

        # 标注关键深度值（与非线性映射对应）
        # 顶部：2.0m（红色）
        cv2.putText(depth_colormap, '2.0m',
                   (colorbar_x - 5, colorbar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

        # 关键深度点标注（根据非线性映射）
        # 0.5m位置（映射值200/255 ≈ 78%位置）
        pos_05m = int(colorbar_y + colorbar_height * 0.22)  # 从上往下22%
        cv2.putText(depth_colormap, '0.5m',
                   (colorbar_x - 5, pos_05m),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)

        # 0.35m位置（工作区间中心）
        pos_035m = int(colorbar_y + colorbar_height * 0.49)  # 中间位置
        cv2.putText(depth_colormap, '0.35m',
                   (colorbar_x - 5, pos_035m),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        # 0.2m位置（映射值50/255 ≈ 80%位置）
        pos_02m = int(colorbar_y + colorbar_height * 0.80)  # 从上往下80%
        cv2.putText(depth_colormap, '0.2m',
                   (colorbar_x - 5, pos_02m),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)

        # 底部：0m（蓝色）
        cv2.putText(depth_colormap, '0m',
                   (colorbar_x - 5, colorbar_y + colorbar_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
        
        # 添加实际深度统计信息（左上角）
        valid_depth = depth[depth > 0]
        if len(valid_depth) > 0:
            actual_min = np.min(valid_depth)
            actual_max = np.max(valid_depth)
            actual_mean = np.mean(valid_depth)
            
            info_y = 20
            cv2.putText(depth_colormap, f'Actual range: [{actual_min:.2f}, {actual_max:.2f}]m',
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(depth_colormap, f'Mean: {actual_mean:.2f}m',
                       (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return depth_colormap

    def get_best_target(self, targets):
        """获取最高概率的目标"""
        if not targets:
            return None
        # 按score排序，选择最高的
        best_target = max(targets, key=lambda x: x.get('score', 0))
        return best_target

    def detect2base3d(self):
        cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
        # 调整窗口大小以适应RGB图和深度图纵向排布
        cv2.resizeWindow(self.cfg.window_name, self.cfg.img_w + 30, self.cfg.img_h * 2)  # 30是颜色条宽度

        text = 'box.pen.remote.bottle.eraser.battery.tape.rubik cube'

        while True:
            try:
                # 获取相机数据
                #out = self.camera.get_image_bundle_enhanced(skip_frames=5, apply_filters=False)
                out = self.camera.get_image_bundle()
                _, end_pos = self.arm.arm.get_position(return_gripper_center=True)
                print('EEF',end_pos)
                end_pos = np.array(end_pos)
                end_pos[0:3] /= 1000.0

                rgb = out['rgb']
                rgb_ori = rgb.copy()
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                depth = out['depth']
                
                # 深度数据处理 - 修复单位问题
                # camera.get_image_bundle()已经返回米单位的numpy array
                if hasattr(depth, 'get_data'):  # rs.depth_frame对象
                    depth = np.asarray(depth.get_data(), dtype=np.float32) / 1000.0
                elif not isinstance(depth, np.ndarray):
                    depth = np.asarray(depth, dtype=np.float32)
                elif depth.dtype != np.float32:
                    depth = depth.astype(np.float32)
                
                # 验证深度值范围（应该在米单位）
                valid_depth = depth[depth > 0]
                if len(valid_depth) > 0:
                    if valid_depth.max() > 100:  # 可能是毫米
                        print("警告: 检测到毫米单位深度值，自动转换为米")
                        depth = depth / 1000.0
                    elif valid_depth.max() < 0.01:  # 深度值太小
                        print(f"警告: 深度值异常偏小 (max={valid_depth.max():.6f}m)")

                # Refer检测
                targets = []
                try:
                    task = self.ref.forward(rgb=rgb_ori, text=text)
                    # task.result 直接是响应数据
                    if hasattr(task, 'result') and task.result:
                        if isinstance(task.result, dict):
                            if 'data' in task.result and 'result' in task.result['data']:
                                targets = task.result['data']['result'].get('objects', [])
                            else:
                                targets = task.result.get('objects', [])
                except Exception as e:
                    print(f"Refer检测失败: {e}")

                # 获取最高概率目标
                best_target = self.get_best_target(targets)

                # 计算3D坐标 - 使用bbox中心
                point3d = offset = point_in_base = eef_in_base = None
                if best_target:
                    try:
                        bbox = best_target['bbox']
                        cx = int((bbox[0] + bbox[2]) / 2)
                        cy = int((bbox[1] + bbox[3]) / 2)

                        # 获取中心点深度值
                        if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                            depth_value = depth[cy, cx]
                            #breakpoint()
                            if depth_value > 0:  # 有效深度值
                                # 使用camera的unprj_point函数计算3D点
                                point3d = self.camera.unprj_point(depth_value, cx, cy)
                                #breakpoint()
                                point3d = np.array(point3d)  # 转换为numpy数组

                                # 计算机械臂坐标
                                data = self.arm.offset_from_end(point3d=point3d, end_pos=end_pos, src_frame='cam')
                                offset = data['offset']
                                point_in_base = data['point_in_base']
                                eef_in_base = data['eef_in_base']
                           
                    except Exception as e:
                        print(f"3D坐标计算失败: {e}")
                        point3d = offset = point_in_base = eef_in_base = None

                # 可视化
                state = {
                    'targets': targets,
                    'best_target': best_target,
                    'cmd': text,
                    'point3d': point3d,
                    'offset': offset,
                    'point_in_base': point_in_base,
                    'eef_in_base': eef_in_base
                }

                # 打印state信息
                print("=" * 50)
                print(f"Frame: {self.frame_count}")
                print(f"Targets found: {len(targets)}")
                if best_target:
                    bbox = best_target['bbox']
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int((bbox[1] + bbox[3]) / 2)
                    category = best_target.get('category', 'unknown')
                    score = best_target.get('score', 0)
                    print(f"Best target: {category} (score: {score:.3f})")
                    print(f"2D center: ({cx}, {cy})")
                    if point3d is not None:
                        print(f"3D in camera: {[round(x * 1000) for x in point3d]} mm")
                    if point_in_base is not None:
                        print(f"3D in base: {[round(x * 1000) for x in point_in_base]} mm")
                    if offset is not None:
                        print(f"Offset: {[round(x * 1000) for x in offset]} mm")
                    if eef_in_base is not None:
                        print(f"eef in base: {[round(x * 1000) for x in eef_in_base]} mm")
                else:
                    print("No target detected")
                print("=" * 50)

                # 保存帧数据
                self.save_frame_data(rgb, depth)

                vis_rgb = self.vis_frame(rgb=rgb, depth=depth, state=state)
                cv2.imshow(self.cfg.window_name, vis_rgb)

                if self.vis:
                    self.vis.put_frame(rgb_ori, dpt=depth, msg=state)

                # 键盘控制
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    print("正在退出...")
                    if self.vis:
                        self.vis.stop()
                    self.arm.disconnect()
                    cv2.destroyAllWindows()
                    break

            except Exception as e:
                print(f"主循环错误: {e}")
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    print("正在退出...")
                    if self.vis:
                        self.vis.stop()
                    self.arm.disconnect()
                    cv2.destroyAllWindows()
                    break


def init_robot():
    img_width = 1280
    img_height = 720

    # 可视化模块
    cfg = Config()
    cfg.img_h = img_height
    cfg.img_w = img_width
    cfg.dpt_img_h = img_height
    cfg.dpt_img_w = img_width
    cfg.enable_dpt = True
    cfg.enable_write = True
    cfg.enable_show = False
    cfg.window_name = 'Demo'
    cfg.q_size = 100
    cfg.out_video = 'grasp.mp4'
    vis = Visualizer(cfg)
    vis.start(in_thread=True)

    # 机械臂模块
    cfg = Config()
    cfg.can_name = 'can0'
    #cfg.init_pos = dict(x=400, y=0, z=180, roll=180, pitch=30, yaw=180)
    cfg.init_pos = dict(x=390, y=0, z=155, roll=180, pitch=12, yaw=180)


    #cfg.T_cam2flan = [0.04010927904656854, 0.08283986339504643, 0.0025333968091171308]  # calibration
    #q =  [0.14206540330899609, -0.1423312973808512, 0.6810236905433535, 0.7041064947060541]  # [x,y,z,w]

    #cfg.T_cam2flan = [0.014751452411217648, 0.08057593892922113, -0.00631765125928978]   #urdf
    #q =  [0.10700593899442587, -0.10600588348980507, 0.7000388532345616, 0.6980387422253201]  # [x,y,z,w]


    ### 20260109
    cfg.T_cam2flan = [-0.07583874846136467,0.05292914229358449,0.05706895145407312]
    q = [0.1558761610824957,-0.16214154506130807,0.6975255175538586,-0.6803461575790205]

    cfg.R_cam2flan = R.from_quat(q).as_matrix().tolist()
    cfg.T_gripper2flan = [0.0, 0.0, 0.13503]
    #cfg.T_gripper2flan = [0.0, 0.0, 0.1000]
    cfg.gripper_as_end = True
    arm = ArmRobot(cfg)
    arm.connect()

    # 主相机
    cfg = Config()
    cfg.device_id = 337122071190
    cfg.width = img_width
    cfg.height = img_height
    cfg.fps = 6
    cfg.use_calibrated_intrinsics = True
    camera = RealSenseCamera(cfg=cfg)
    camera.connect()

    # 参考检测模块
    cfg = Config()
    cfg.uri = r'/v2/task/dinox/detection'
    cfg.status_uri = r'/v2/task_status'
    cfg.token = 'c4cdacb48bc4d1a1a335c88598a18e8c'
    ref = DinoXDetectorCloud(cfg)

    # 创建机器人实例
    cfg = Config()
    cfg.window_name = 'demo: detect_2_base3d'
    cfg.img_w = img_width
    cfg.img_h = img_height
    robot = RobotPiperCam(cfg=cfg, camera=camera, arm=arm, vis=vis, ref=ref)

    return robot


if __name__ == '__main__':
    robot = init_robot()
    robot.detect2base3d()
