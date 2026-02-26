import math
import os
import queue
import sys
import threading
import time
from queue import Queue

import cv2
import numpy as np

# import open3d as o3d
from camera import RealSenseCamera
from percept import GraspAnythingOnline, DinoXDetectorOnline
from mmengine.config import Config
from PIL import Image, ImageDraw, ImageFont
#from robot_xarm import XArmRobot as ArmRobot
from robot_piper import PiperRobot as ArmRobot
from scipy.spatial.transform import Rotation as R

# from track import Tracker
# from vis_cls import Visualizer  # 已禁用数据录制
#from voice_input import VoiceInput
from pycocotools import mask as coco_mask

# ROS navigation imports
try:
    import rospy
    import actionlib
    from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
    HAS_ROS_NAV = True
except ImportError:
    HAS_ROS_NAV = False
    print("警告: ROS导航模块未安装，navi_grasp_show功能将不可用")

IS_MOBILE = False
if IS_MOBILE:
    from simple_tracer import SimpleTracer
    



class DemoRobot:
    """主控制类，负责协调整个抓取系统的运行
    
    该类整合了相机、检测、机械臂、语音等所有模块，实现完整的抓取流程控制。
    支持多种工作模式：等待、确认、自动、参考、语音等。
    """
    def __init__(self, cfg, camera, det, arm, ref, voc=None, tracker=None, chassis=None, **kwargs):
        """初始化Demo控制器

        Args:
            cfg: 配置对象，包含系统参数
            camera: 主相机对象（RealSenseCamera），用于RGB-D采集
            det: 检测对象（GraspAnythingOnline），用于抓取点检测
            arm: 机械臂对象（ArmRobot），用于执行抓取
            ref: 参考定位对象（DinoXDetectorOnline），用于语义定位
            voc: 语音输入对象（VoiceInput），用于语音控制
            tracker: 跟踪器对象，用于多目标跟踪
            chassis: 底盘控制对象（SimpleTracer），用于移动底盘控制
        """
        self.cfg = cfg
        self.camera = camera  # 主相机（带深度）
        self.det = det  # 在线检测服务
        self.arm = arm  # xArm机械臂控制
        self.ref = ref  # 语义参考定位
        self.voc = voc  # 语音输入
        self.tracker = tracker  # 目标跟踪器
        self.chassis = chassis  # 底盘控制器

        self.kwargs = kwargs
        # 系统工作模式：wait(等待) | confirm(确认) | auto(自动) | ref(参考) | voice(语音) | arm_picking(执行中)
        self.mode = cfg.get('mode', 'wait')  
        self.arm_ops_q = Queue()  # 机械臂操作队列，确保线程安全
        fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        self._font = ImageFont.truetype(fp, size=30)  # 中文字体，用于显示
        self._last_arm_op_done_time = None  # 记录上次操作完成时间

    def is_at_init_position(self, tolerance=20):
        """检查机械臂是否在初始位置
        
        Args:
            tolerance: 位置容差（毫米）
            
        Returns:
            bool: True表示在初始位置，False表示不在
        """
        if not hasattr(self.arm, 'init_pos') or self.arm.init_pos is None:
            # 如果没有定义初始位置，始终返回True以保持兼容性
            return True
            
        # 获取当前位置（单位：mm）
        _, current_pos = self.arm.arm.get_position(return_gripper_center=True)
        
        # 检查xyz坐标是否在容差范围内
        init_pos = self.arm.init_pos
        if abs(current_pos[0] - init_pos['x']) > tolerance:
            return False
        if abs(current_pos[1] - init_pos['y']) > tolerance:
            return False
        if abs(current_pos[2] - init_pos['z']) > tolerance:
            return False
            
        return True

    def draw_chinsese_text(self, rgb, pos, text):
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        img_w = pil_img.width
        # text_w, text_h = draw.textsize(text, font=self._font)
        bbox = draw.textbbox((0, 0), text, font=self._font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        x = (img_w - text_w) // 2
        y = pos[1]
        draw.text((x, y), text, font=self._font, fill='blue')
        img = np.array(pil_img)
        return img

    def listen_on_arm_ops(self):
        """监听机械臂操作队列的线程函数
        
        该函数在独立线程中运行，从队列中取出操作命令并执行。
        确保机械臂操作的原子性和线程安全性。
        """
        while True:
            op = self.arm_ops_q.get()  # 阻塞等待新的操作命令
            action, para = op
            old_mode = self.mode
            
            # 处理模式切换逻辑
            if 'start_mode' in para:
                start_mode = para.get('start_mode', old_mode)
                self.mode = start_mode
            else:
                start_mode = old_mode
            end_mode = para.get('end_mode', start_mode)
            print(f'{action=} {para=} {old_mode=} {start_mode=} {end_mode}')
            
            # 参数验证
            if para['offset'] is None:
                #print('invalid ops parameters')
                continue
                
            try:
                if action == 'pick':
                    self.mode = 'arm_picking'  # 标记为执行中状态
                    # 打印抓取目标的详细信息
                    print("="*50)
                    print(f"开始抓取物体")
                    print(f"  类别: {para.get('category', 'unknown')}")
                    if para.get('point3d') is not None:
                        point3d_list = para['point3d'] if isinstance(para['point3d'], list) else para['point3d'].tolist()
                        print(f"  相机坐标系3D位置: {[round(x*1000) for x in point3d_list]} mm")
                    if para.get('point_in_base') is not None:
                        point_base_list = para['point_in_base'] if isinstance(para['point_in_base'], list) else para['point_in_base'].tolist()
                        print(f"  基座坐标系3D位置: {[round(x*1000) for x in point_base_list]} mm")
                    if para.get('offset') is not None:
                        offset_list = para['offset'] if isinstance(para['offset'], list) else para['offset'].tolist()
                        print(f"  偏移量: {[round(x*1000) for x in offset_list]} mm")
                    else:
                        print(f"  偏移量: None")
                    print(f"  旋转角度: {para['angle']:.1f}°" if para.get('angle') is not None else "  旋转角度: None")
                    gripper_width = para.get('gripper_width')
                    if gripper_width is not None:
                        if isinstance(gripper_width, (int, float)):
                            print(f"  夹爪宽度: {gripper_width:.4f}")
                        else:
                            print(f"  夹爪宽度: {gripper_width}")
                    else:
                        print(f"  夹爪宽度: default")
                    print("="*50)
                    # 执行抓取动作：移动到目标位置、调整角度、控制夹爪
                    self.arm.pick(offset=para['offset'], angle=para['angle'], gripper_width=para.get('gripper_width', None))
                    
                elif action == 'locating':
                    self.mode = 'grasping'
                elif action == 'grasping':  # simple picking after locating
                    self.mode = 'wait'
                else:
                    raise NotImplementedError(f'unexpected error: {op=}')
            except Exception as e:
                print(f'arm: {e}')
            finally:
                self.mode = end_mode  # 恢复到结束模式
                # 清空队列中的后续操作，避免累积
                with self.arm_ops_q.mutex:  
                    if len(self.arm_ops_q.queue) > 0:
                        print(f'warning: {len(self.arm_ops_q)} ops in arm_ops_q after one is DONE')
                        self.arm_ops_q.queue.clear()
                self._last_arm_op_done_time = time.time()

    def vis_frame(self, rgb, depth, state):
        affs = state['affs']
        fts = state['fts']
        cmd = state['cmd']
        chosen_aff = state['chosen_aff']
        point3d = state['point3d']
        offset = state['offset']
        eef_in_base = state['eef_in_base']
        point_in_base = state['point_in_base']
        img_h, img_w, _ = rgb.shape

        # 深度图可视化 (TURBO colormap)
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_TURBO)
        if 0:
            center_x = img_w // 2
            center_y = img_h // 2
            cv2.circle(rgb, (center_x, center_y), 3, [0, 0, 255], 3)
            cv2.putText(rgb, 'UV=(0,0)', (center_x - 30, center_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if 1:
            # tmp = affs[0]['affs']
            tmp = [obj['affs'] for obj in affs[0]]
            ps = [x for p in tmp for x in p]
            ps = [(int(x[0]), int(x[1])) for x in ps]  #
            ps = [x for x in ps if x[0] <= img_w and x[1] <= img_h]
            ps = [[x, x] for x in ps]
            ps = np.array(ps, dtype=np.int32)
            cv2.polylines(rgb, ps, isClosed=False, color=[0, 0, 255], thickness=7)

        if 1 and 'targets' in fts:
            for target in fts['targets']:
                bbox = target['bbox']
                bbox = [int(x) for x in bbox]
                cv2.rectangle(rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), [0, 0, 255], 2)

        if 1 and self.mode in ['ref', 'voice'] and cmd is not None:
            rgb = self.draw_chinsese_text(rgb, pos=(100, 680), text=cmd)

        text_offset_u = 100 #800
        text_offset_v = 20 #50
        delta_v = 30
        if 1 and chosen_aff is not None:
            rect = chosen_aff['aff']
            rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4] * 180 / math.pi)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(rgb, [box], 0, (0, 255, 0), 2)

            cx, cy = rect[0][0], rect[0][1]
            cx, cy = int(cx), int(cy)
            cv2.circle(rgb, (cx, cy), 3, [0, 255, 0], 3)

            cv2.putText(
                rgb,
                f'3D in Cam: {[round(x * 1000) for x in point3d]} mm',
                (text_offset_u, text_offset_v),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            text_offset_v += delta_v
            cv2.putText(
                rgb,
                f'3D in Base: {[round(x * 1000) for x in point_in_base]} mm',
                (text_offset_u, text_offset_v),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            text_offset_v += delta_v
            cv2.putText(
                rgb,
                f'offset in Base: {[round(x * 1000) for x in offset]} mm',
                (text_offset_u, text_offset_v),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            text_offset_v += delta_v
            cv2.putText(
                rgb,
                f'eef in Base: {[round(x * 1000) for x in eef_in_base]} mm',
                (text_offset_u, text_offset_v),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )


            text_offset_v += delta_v
            cv2.putText(rgb, f'UV in RGB: {(cx, cy)} px', (text_offset_u, text_offset_v), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            text_offset_v += delta_v
            cv2.putText(rgb, f'aff score: {chosen_aff["score"]: .2f}', (text_offset_u, text_offset_v), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            text_offset_v += delta_v
            category = state.get('category', 'unknown')
            cv2.putText(rgb, f'category: {category}', (text_offset_u, text_offset_v), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            text_offset_v += delta_v
            if 0:
                tmp = [obj['scores'] for obj in affs[0]]
                cv2.putText(rgb, f'aff scores: {tmp}', (text_offset_u, text_offset_v), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if 1 and chosen_aff is None:
            text_offset_v += delta_v
            cv2.putText(
                rgb,
                'no valid afforandance',
                (text_offset_u, text_offset_v),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        text_offset_v += delta_v
        cv2.putText(
            rgb,
            f'mode: {self.mode}',
            (text_offset_u, text_offset_v),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # RGB + Depth 上下堆叠
        rgb = np.vstack((rgb, depth_colored))
        return rgb

    def choose(self, rgb, depth, affs, fts):
        """选择最优抓取点算法
        
        从检测到的多个可抓取点中，根据评分和约束条件选择最优的抓取位置。
        
        Args:
            rgb: RGB图像
            depth: 深度图
            affs: 检测到的所有可抓取点信息
            fts: 过滤条件（如指定目标框）
            
        Returns:
            dict: 包含最优抓取点信息，或None（无有效抓取点）
        """
        # 调试信息（简化）
        print(f"[CHOOSE] 开始选择最优抓取点 (检测到{len(affs[0]) if affs else 0}个候选对象)")
        
        result_bbox = []  # 候选抓取框列表
        result_img_id = []  # 对应的图像ID
        score_li = []  # 抓取评分列表
        tps_li = []  # 接触点列表
        category_li = []  # 类别列表
        img_h, img_w, _ = rgb.shape
        
        # 遍历所有检测到的物体
        for img_id in range(1):
            for obj in affs[img_id]:
                scores_obj = obj['scores']  # 该物体所有抓取点的评分
                affs_obj = obj['affs']  # 该物体所有抓取点的位置
                tps = obj['touching_points']  # 抓取接触点
                
                # mask = obj['dt_mask']
                # mask = np.asfortranarray(mask)
                # rle = coco_mask.encode(np.asfortranarray(mask))
                # rle['counts'] = rle['counts'].decode('utf-8')
                # obj['dt_mask'] = rle

                if 1:
                    tmp1 = []
                    tmp2 = []
                    tmp3 = []
                    for i in range(len(affs_obj)):
                        #if affs_obj[i][0] > img_w // 2 or affs_obj[i][1] > img_h:
                        if affs_obj[i][0] > img_w  or affs_obj[i][1] > img_h:
                            continue
                        else:
                            tmp1.append(affs_obj[i])
                            # breakpoint()
                            tmp2.append(scores_obj[i])
                            tmp3.append(tps[i])

                    if len(tmp1) > 1:
                        affs_obj = tmp1
                        scores_obj = tmp2
                        tps = tmp3
                    else:
                        continue

                chosen_category = None  # 记录选中物体的类别
                if 'targets' in fts:
                    targets = fts['targets']  # in XYXY
                    tmp1, tmp2, tmp3 = [], [], []
                    tmp_categories = []  # 记录对应的类别
                    for i in range(len(affs_obj)):  # for each affs of an object
                        bbox = affs_obj[i]
                        for j, target in enumerate(targets):
                            target_bbox = target['bbox']
                            target_category = target.get('category', 'unknown')
                            if target_bbox[0] <= bbox[0] <= target_bbox[2] and target_bbox[1] <= bbox[1] <= target_bbox[3]:
                                tmp1.append(bbox)
                                tmp2.append(scores_obj[i])
                                tmp3.append(tps[i])
                                tmp_categories.append(target_category)
                                break
                    
                    if len(tmp1) > 0:
                        affs_obj = tmp1
                        scores_obj = tmp2
                        tps = tmp3
                        # 选择最高分的抓取点，并记录对应的类别
                        k = np.argmax(scores_obj)
                        chosen_category = tmp_categories[k] if tmp_categories else 'unknown'
                    else:
                        continue  # ignore the object
                else:
                    chosen_category = 'unknown'  # 如果没有targets过滤，类别为unknown
                    print(f"[CHOOSE-DEBUG] 无targets过滤，类别设为unknown")
                
                k = np.argmax(scores_obj)
                result_bbox.append(affs_obj[k])
                score_li.append(scores_obj[k])
                tps_li.append(tps[k])
                result_img_id.append(img_id)
                category_li.append(chosen_category)  # 保存类别信息
        if len(score_li) == 0:
            # breakpoint()
            return None
        max_index, max_score = max(enumerate(score_li), key=lambda x: x[1])
        # breakpoint()
        # if max_score < 0.8:
        #     print(f'max_score is : {max_score}')
        #     return None
        result_bbox = result_bbox[max_index]
        tps = tps_li[max_index]
        category = category_li[max_index] if category_li else 'unknown'
        return dict(aff=result_bbox, score=max_score, touching_points=tps, index=max_index, category=category)

    def add_arm_ops(self, action, para):
        old_mode = self.mode
        self.mode = 'arm_todo'
        if 'start_mode' not in para:
            para['start_mode'] = old_mode
        self.arm_ops_q.put((action, para))
    
    def detect(self, timeout=3.0, target_text='pen.remote.bottle.eraser.battery.tape.rubik cube'):
        """检测物体，最多循环指定时间
        
        Args:
            timeout: 超时时间（秒）
            target_text: 要检测的目标物体描述
            
        Returns:
            dict: 包含检测结果的字典，如果未检测到则返回None
        """
        print(f"开始检测，超时时间: {timeout}秒")
        start_time = time.time()
        detection_result = None
        
        while (time.time() - start_time) < timeout:
            # 检查是否在初始位置
            if not self.is_at_init_position(tolerance=20):
                print("⚠ 机械臂不在初始位置，等待...")
                time.sleep(0.5)
                continue
                
            # 获取相机数据
            out = self.camera.get_image_bundle_enhanced(skip_frames=5, apply_filters=True)
            _, end_pos = self.arm.arm.get_position(return_gripper_center=True)
            end_pos = np.array(end_pos)
            end_pos[0:3] /= 1000.0  # 只转换前3个位置坐标为米，保留后3个旋转角度
            
            rgb = out['rgb']
            rgb_ori = rgb.copy()
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # 直接覆盖rgb变量，与start()函数保持一致
            depth = out['depth']
            
            # Grasp检测
            affs = None
            try:
                affs, _ = self.det.forward(rgb_ori, depth, bag=False)
                # 打印grasp检测结果
                if affs:
                    print(f"[GRASP] 检测到 {len(affs)} 个抓取候选:")
                    for i, aff in enumerate(affs):
                        if 'aff' in aff:
                            rect = aff['aff']
                            score = aff.get('score', 0)
                            category = aff.get('category', 'unknown')
                            print(f"  候选{i+1}: 类别={category}, 得分={score:.3f}, 位置=({rect[0]:.1f},{rect[1]:.1f})")
                else:
                    print("[GRASP] 未检测到抓取候选")
            except Exception as e:
                print(f"[GRASP] 检测失败: {e}")
                time.sleep(0.5)
                continue
                
            if affs is None:
                affs = [{}]
                

            # Refer检测
            targets = []
            try:
                task = self.ref.forward(rgb=rgb_ori, text=target_text)
                targets = task.result.get('objects', [])
            except Exception as e:
                print(f"[REFER] 检测失败: {e}")
            fts = {'targets': targets}

            
            # 选择最优抓取点
            chosen_aff = self.choose(rgb_ori, depth, affs, fts=fts)
            
            # 打印选择结果
            if chosen_aff is not None:
                rect = chosen_aff.get('aff', [])
                category = chosen_aff.get('category', 'unknown')
                score = chosen_aff.get('score', 0)
                if rect:
                    print(f"[CHOOSE] 选中抓取点: 类别={category}, 得分={score:.3f}, 位置=({rect[0]:.1f},{rect[1]:.1f}), 角度={rect[4]*180/math.pi:.1f}°")
                else:
                    print(f"[CHOOSE] 选中抓取点但无位置信息")
            else:
                print("[CHOOSE] 未找到合适的抓取点")
            
            if chosen_aff is not None:
                # 计算3D坐标和偏移
                data3d = self.camera.unprj(aff=chosen_aff, rgb=rgb, depth=depth)
                point3d = data3d['point3d']
                width3d = data3d['width3d']
                
                # 调试：打印3D投影计算详情
                rect = chosen_aff.get('aff', [])
                if rect:
                    cx, cy = rect[0], rect[1]
                    depth_value = depth[int(cy), int(cx)] if cy < depth.shape[0] and cx < depth.shape[1] else -1
                    print(f"[3D-DEBUG] 像素位置=({cx:.1f}, {cy:.1f}), 深度值={depth_value:.3f}m")
                    print(f"[3D-DEBUG] 相机坐标系3D点: ({point3d[0]:.3f}, {point3d[1]:.3f}, {point3d[2]:.3f})")
                    print(f"[3D-DEBUG] 夹爪宽度3D: {width3d:.3f}m")
                
                # 调试：打印完整的坐标变换过程
                print(f"[TRANSFORM-DEBUG] 末端坐标: ({end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f})")
                print(f"[TRANSFORM-DEBUG] 相机标定参数:")
                print(f"  T_cam2flan: ({self.arm.T_cam2flan[0]:.3f}, {self.arm.T_cam2flan[1]:.3f}, {self.arm.T_cam2flan[2]:.3f})")
                print(f"  T_cam2end: ({self.arm.T_cam2end[0]:.3f}, {self.arm.T_cam2end[1]:.3f}, {self.arm.T_cam2end[2]:.3f})")
                
                data = self.arm.offset_from_end(point3d=point3d, end_pos=end_pos, src_frame='cam')
                offset = data['offset']
                point_in_base = data['point_in_base']
                
                print(f"[TRANSFORM-DEBUG] 相机→基座变换后: ({point_in_base[0]:.3f}, {point_in_base[1]:.3f}, {point_in_base[2]:.3f})")
                print(f"[TRANSFORM-DEBUG] 偏移向量: ({offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f})")
                
                # 检查工作范围
                min_reach = 0.250
                max_reach = 0.600
                min_height = -0.270
                max_height = 0.300
                
                print("POINT_IN_BASE:",point_in_base)
                horizontal_distance = np.sqrt(point_in_base[0]**2 + point_in_base[1]**2)
                z_height = point_in_base[2]
                
                in_workspace = (min_reach <= horizontal_distance <= max_reach and 
                              min_height <= z_height <= max_height)
                
                # 打印工作空间检查结果
                print(f"[WORKSPACE] 目标点基座坐标: ({point_in_base[0]:.3f}, {point_in_base[1]:.3f}, {point_in_base[2]:.3f})")
                print(f"[WORKSPACE] 水平距离={horizontal_distance:.3f}m (范围: {min_reach:.3f}-{max_reach:.3f}), 高度={z_height:.3f}m (范围: {min_height:.3f}-{max_height:.3f})")
                print(f"[WORKSPACE] 在工作空间内: {'是' if in_workspace else '否'}")
                
                if in_workspace:
                    rect = chosen_aff['aff']
                    angle = rect[4] * 180 / math.pi
                    
                    detection_result = {
                        'success': True,
                        'chosen_aff': chosen_aff,
                        'offset': offset,
                        'angle': angle,
                        'width3d': width3d,
                        'category': chosen_aff.get('category', 'unknown'),
                        'point3d': point3d,
                        'point_in_base': point_in_base
                    }
                    
                    # 可视化
                    state = {
                        'affs': affs,
                        'chosen_aff': chosen_aff,
                        'fts': fts,
                        'cmd': target_text,
                        'point3d': point3d,
                        'offset': offset,
                        'point_in_base': point_in_base,
                        'eef_in_base': data['eef_in_base'],
                        'category': chosen_aff.get('category', 'unknown')
                    }

                    # 显示可视化
                    try:
                        vis_rgb = self.vis_frame(rgb=rgb, depth=depth, state=state)
                        cv2.imshow(self.cfg.window_name, vis_rgb)
                        cv2.waitKey(1)
                    except Exception as e:
                        print(f"可视化失败: {e}")

                    print(f"✓ 检测到物体: {detection_result['category']}")
                    break
                else:
                    print(f"⚠ 目标超出工作范围 (距离: {horizontal_distance:.3f}m, 高度: {z_height:.3f}m)")

            # 即使没有检测到也显示当前帧
            try:
                # 构建基础可视化state
                vis_state = {
                    'affs': affs if affs else [{}],
                    'chosen_aff': chosen_aff,
                    'fts': fts,
                    'cmd': target_text,
                    'point3d': None,
                    'offset': None,
                    'point_in_base': None,
                    'eef_in_base': None,
                    'category': None
                }
                vis_rgb = self.vis_frame(rgb=rgb, depth=depth, state=vis_state)

                # 添加检测状态文字（中文）
                elapsed = time.time() - start_time
                status_text = f"抓取检测中... {elapsed:.1f}s / {timeout:.1f}s"
                vis_rgb = self.draw_chinsese_text(vis_rgb, pos=(50, 50), text=status_text)

                cv2.imshow(self.cfg.window_name, vis_rgb)
            except:
                # 简单显示原始图像并添加状态
                elapsed = time.time() - start_time
                status_text = f"抓取检测中... {elapsed:.1f}s / {timeout:.1f}s"
                rgb = self.draw_chinsese_text(rgb, pos=(50, 50), text=status_text)
                cv2.imshow(self.cfg.window_name, rgb)

            # 短暂等待
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
                
        if detection_result is None:
            print(f"✗ {timeout}秒内未检测到有效目标")
            
        return detection_result
    
    def grasp(self, detection_result):
        """执行抓取动作
        
        Args:
            detection_result: detect()方法返回的检测结果
            
        Returns:
            bool: 抓取是否成功
        """
        if detection_result is None or not detection_result.get('success', False):
            print("✗ 无有效检测结果，跳过抓取")
            return False
            
        # 获取抓取参数
        offset = detection_result['offset']
        angle = detection_result['angle']
        gripper_width = detection_result['width3d']
        category = detection_result['category']
        point3d = detection_result['point3d']
        point_in_base = detection_result['point_in_base']
        
        # 打印抓取信息
        print("="*50)
        print(f"开始抓取物体")
        print(f"  类别: {category}")
        if point3d is not None:
            point3d_list = point3d if isinstance(point3d, list) else point3d.tolist()
            print(f"  相机坐标系3D位置: {[round(x*1000) for x in point3d_list]} mm")
        if point_in_base is not None:
            point_base_list = point_in_base if isinstance(point_in_base, list) else point_in_base.tolist()
            print(f"  基座坐标系3D位置: {[round(x*1000) for x in point_base_list]} mm")
        if offset is not None:
            offset_list = offset if isinstance(offset, list) else offset.tolist()
            print(f"  偏移量: {[round(x*1000) for x in offset_list]} mm")
        print(f"  旋转角度: {angle:.1f}°" if angle is not None else "  旋转角度: None")
        if gripper_width is not None:
            if isinstance(gripper_width, (int, float)):
                print(f"  夹爪宽度: {gripper_width:.4f}")
            else:
                print(f"  夹爪宽度: {gripper_width}")
        else:
            print(f"  夹爪宽度: default")
        print("="*50)
        
        try:
            # 直接调用机械臂的pick方法
            self.arm.pick(offset=offset, angle=angle, gripper_width=gripper_width)
            print("✓ 抓取完成")
            return True
        except Exception as e:
            print(f"✗ 抓取失败: {e}")
            return False
    
    def mobile_grasp_show(self):
        """执行完整的演示流程
        底盘移动到3个位置，每个位置尝试检测和抓取
        
        注意事项：
        1. 底盘移动会改变相机-世界坐标系关系
        2. 当前实现假设机械臂基座固定在底盘上
        3. 建议在每个位置放置易抓取的测试物体
        """
        print("="*60)
        print("开始 Great Show 演示")
        print("⚠ 警告：底盘移动后坐标系会改变，请确保物体在机械臂工作范围内")
        print("="*60)
        
        # 检查底盘
        if self.chassis is None:
            print("✗ 底盘未初始化，无法执行演示")
            return False
            
        # 初始化显示窗口
        cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.cfg.window_name, self.cfg.img_w, 2 * self.cfg.img_h)
        
        try:
            # 检查底盘状态
            print("\n检查底盘状态...")
            if not self.chassis.is_connected():
                print("✗ 底盘未连接")
                return False
                
            if not self.chassis.check_battery():
                print("✗ 电池电压异常")
                return False
                
            if not self.chassis.check_and_fix_fault():
                print("✗ 底盘故障")
                return False
                
            print("✓ 底盘状态正常")
            
            
            
            '''
            # 在位置0进行检测和抓取
            print("\n在位置0进行检测...")
            detection = self.detect(timeout=1000.0)
            
            if detection and detection.get('success', False):
                print(f"\n检测到物体: {detection.get('category', 'unknown')}")
                print("准备抓取...")
                success = self.grasp(detection)
                if success:
                    print("✓ 位置0抓取成功")
                    time.sleep(2)
                else:
                    print("✗ 位置0抓取失败")
            else:
                print("位置0未检测到可抓取物体")
            '''
                
            
            # 可选：启用底盘移动演示（现在注释掉，仅在位置0测试）
            # 步骤1：移动到位置1并抓取
            print("\n" + "="*40)
            print("步骤1: 移动到位置1")
            print("="*40)
            self.chassis.move_step1()
            self.chassis.stop()
            time.sleep(1)
            
            print("在位置1进行检测...")
            detection = self.detect(timeout=5.0)
            if detection and detection.get('success', False):
                print(f"检测到物体: {detection.get('category', 'unknown')}")
                self.grasp(detection)
                time.sleep(2)
                
            # 步骤2：移动到位置2并抓取
            print("\n" + "="*40)
            print("步骤2: 移动到位置2")
            print("="*40)
            self.chassis.move_step2()
            self.chassis.stop()
            time.sleep(1)
            
            print("在位置2进行检测...")
            detection = self.detect(timeout=5.0)
            if detection and detection.get('success', False):
                print(f"检测到物体: {detection.get('category', 'unknown')}")
                self.grasp(detection)
                time.sleep(2)
                
            # 步骤3：移动到位置3并抓取
            print("\n" + "="*40)
            print("步骤3: 移动到位置3")
            print("="*40)
            self.chassis.move_step3()
            self.chassis.stop()
            time.sleep(1)
            
            print("在位置3进行棉测...")
            detection = self.detect(timeout=5.0)
            if detection and detection.get('success', False):
                print(f"检测到物体: {detection.get('category', 'unknown')}")
                self.grasp(detection)
                time.sleep(2)
            
            
            print("\n" + "="*60)
            print("✓ Great Show 演示完成！")
            print("="*60)
            
            # 停止底盘
            if self.chassis:
                self.chassis.stop()
            
            cv2.destroyAllWindows()
            return True
            
        except Exception as e:
            print(f"✗ 演示过程中出错: {e}")
            # 确保资源正确释放
            try:
                if self.chassis:
                    self.chassis.stop()
                cv2.destroyAllWindows()
            except:
                pass
            return False

    def navi_grasp_show(self, target_file=None):
        """导航抓取演示：导航到目标位置后连续抓取两个物体

        流程：
        1. 初始化ROS节点和move_base客户端
        2. 读取目标位置并导航
        3. 到达后进行检测和抓取
        4. 连续成功抓取两个物体后结束

        Args:
            target_file: 目标位置文件路径，默认为 demo_slam_grasp_pipline/target.txt
        """
        print("="*60)
        print("开始 Navigation Grasp Show 演示")
        print("="*60)

        # 检查ROS导航模块
        if not HAS_ROS_NAV:
            print("✗ ROS导航模块未安装，无法执行演示")
            return False

        # 设置目标文件路径
        if target_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            target_file = os.path.join(script_dir, 'demo_slam_grasp_pipline', 'target.txt')

        # 读取目标位置
        if not os.path.exists(target_file):
            print(f"✗ 目标文件不存在: {target_file}")
            print("请先运行 _cc_get_current_pose.py 保存目标位置")
            return False

        try:
            with open(target_file, 'r') as f:
                line = f.readline().strip()
                values = [float(v) for v in line.split()]

            if len(values) != 7:
                print("✗ target.txt 格式错误，需要 7 个值: px py pz ox oy oz ow")
                return False

            px, py, pz, ox, oy, oz, ow = values
            print(f"✓ 读取目标位置: x={px:.3f}, y={py:.3f}")

        except Exception as e:
            print(f"✗ 读取目标文件失败: {e}")
            return False

        # 初始化显示窗口
        cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.cfg.window_name, self.cfg.img_w, 2 * self.cfg.img_h)

        grasp_count = 0  # 成功抓取计数
        target_grasp_count = 2  # 目标抓取数量

        try:
            # 初始化ROS节点（如果尚未初始化）
            try:
                rospy.init_node('navi_grasp_show', anonymous=True)
            except rospy.exceptions.ROSException:
                # 节点可能已经初始化
                pass

            # 连接move_base action server
            print("\n" + "="*40)
            print("步骤1: 导航到目标位置")
            print("="*40)
            print("等待 move_base action server...")

            client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
            if not client.wait_for_server(rospy.Duration(10.0)):
                print("✗ 无法连接到 move_base action server")
                return False
            print("✓ 已连接到 move_base")

            # 构建导航目标
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = px
            goal.target_pose.pose.position.y = py
            goal.target_pose.pose.position.z = pz
            goal.target_pose.pose.orientation.x = ox
            goal.target_pose.pose.orientation.y = oy
            goal.target_pose.pose.orientation.z = oz
            goal.target_pose.pose.orientation.w = ow

            # 发送导航目标
            print(f"发送导航目标: x={px:.3f}, y={py:.3f}")
            client.send_goal(goal)

            # 等待导航完成，同时更新相机画面
            print("等待导航完成...")
            nav_start_time = time.time()
            while not rospy.is_shutdown():
                # 检查导航状态（非阻塞）
                if client.wait_for_result(rospy.Duration(0.1)):
                    break

                # 更新相机画面显示
                try:
                    out = self.camera.get_image_bundle_enhanced(skip_frames=2, apply_filters=False)
                    rgb = out['rgb']
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    # 添加导航状态文字（中文）
                    elapsed = time.time() - nav_start_time
                    status_text = f"导航中... {elapsed:.1f}s  目标: ({px:.2f}, {py:.2f})"
                    rgb = self.draw_chinsese_text(rgb, pos=(50, 50), text=status_text)

                    cv2.imshow(self.cfg.window_name, rgb)
                except Exception as e:
                    pass

                # 检查用户是否按 'q' 取消
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    print("用户取消导航")
                    client.cancel_goal()
                    return False

            state = client.get_state()
            if state == actionlib.GoalStatus.SUCCEEDED:
                print("✓ 导航成功完成")
            else:
                print(f"⚠ 导航结束，状态码: {state}")
                # 即使导航不完全成功，也尝试继续抓取

            # 等待机器人稳定
            time.sleep(1.0)

            # 步骤2: 连续抓取
            print("\n" + "="*40)
            print("步骤2: 开始检测和抓取")
            print(f"目标: 连续抓取 {target_grasp_count} 个物体")
            print("="*40)

            max_attempts = 10  # 最大尝试次数
            attempt = 0

            while grasp_count < target_grasp_count and attempt < max_attempts:
                attempt += 1
                print(f"\n--- 抓取尝试 {attempt}/{max_attempts} (已成功: {grasp_count}/{target_grasp_count}) ---")

                # 检测物体
                detection = self.detect(timeout=10.0)

                if detection and detection.get('success', False):
                    category = detection.get('category', 'unknown')
                    print(f"检测到物体: {category}")
                    print("准备抓取...")

                    # 执行抓取
                    success = self.grasp(detection)

                    if success:
                        grasp_count += 1
                        print(f"✓ 抓取成功! ({grasp_count}/{target_grasp_count})")
                        # 等待机械臂复位
                        time.sleep(2.0)
                    else:
                        print("✗ 抓取失败，继续尝试下一个")
                        time.sleep(1.0)
                else:
                    print("未检测到可抓取物体，等待后重试...")
                    time.sleep(2.0)

                # 检查用户是否按下 'q' 退出
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    print("用户中断")
                    break

            # 结果汇报
            print("\n" + "="*60)
            if grasp_count >= target_grasp_count:
                print(f"✓ Navigation Grasp Show 演示完成！")
                print(f"  成功抓取 {grasp_count} 个物体")
            else:
                print(f"⚠ 演示结束，未达到目标")
                print(f"  成功抓取: {grasp_count}/{target_grasp_count}")
                print(f"  尝试次数: {attempt}")
            print("="*60)

            cv2.destroyAllWindows()
            return grasp_count >= target_grasp_count

        except Exception as e:
            print(f"✗ 演示过程中出错: {e}")
            import traceback
            traceback.print_exc()
            try:
                cv2.destroyAllWindows()
            except:
                pass
            return False

    def start(self):
        """主循环函数，系统的核心运行逻辑

        该函数实现了完整的感知-决策-执行循环：
        1. 采集RGB-D图像
        2. 调用检测服务获取抓取点
        3. 选择最优抓取方案
        4. 坐标变换和可视化
        5. 根据模式执行相应动作
        """
        # 初始化显示窗口
        cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.cfg.window_name, self.cfg.img_w, 2 * self.cfg.img_h)
        
        fts = dict()  # 过滤器，用于指定抓取目标
        cmd = None  # 用户的文本/语音命令
        just_change_fts = False
        
        # 主循环：持续运行直到用户退出
        while True:
            if self.is_at_init_position(tolerance=20):
                
                # 获取相机数据（RGB + 深度）
                #out = self.camera.get_image_bundle()
                out = self.camera.get_image_bundle_enhanced(skip_frames=5, apply_filters=True)
                # 获取机械臂当前位姿（单位：mm）
                _, end_pos = self.arm.arm.get_position(return_gripper_center=True)

                end_pos = np.array(end_pos)
                end_pos[0:3] /= 1000.0
                intrinsics = self.camera.intrinsics
                K = dict(coeffs=intrinsics.coeffs, fx=intrinsics.fx, fy=intrinsics.fy, height=intrinsics.height, width=intrinsics.width, ppx=intrinsics.ppx, ppy=intrinsics.ppy)
                scale = self.camera.scale

                rgb = out['rgb']
                rgb_ori = rgb.copy()
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                state = dict()
                # breakpoint()
                state['arm.end_pos'] = end_pos.tolist()
                state['camera.K'] = K
                state['camera.scale'] = scale
                state['mode'] = self.mode
                state['time_now'] = time.time()

                depth = out['depth']
                bag = self.mode == 'voice' and cmd is not None and '袋子' in cmd
                bag = False
                if bag:
                    print('bag!!!')
            else:
                None
            
            # 检查是否在初始位置，只有在初始位置才进行检测
            if self.is_at_init_position(tolerance=20):
                # Grasp检测，重试3次
                affs = None
                for retry in range(3):
                    try:
                        affs, _ = self.det.forward(rgb_ori, depth, bag=bag)  # quick, delay < 0.1s
                        break
                    except Exception as e:
                        print(f"Grasp检测失败 (尝试 {retry+1}/3): {e}")
                    if retry < 2:  # 不是最后一次尝试
                        time.sleep(0.5)
                
                if affs is None:
                    affs = [{}]  # 如果全部失败，使用空结果
                    print("⚠ Grasp检测失败3次，跳过")
                    continue  # 跳过本次循环，等待下一帧
                

                # Refer检测
                cmd = text = 'toy cubic.remote.bottle'
                targets = []
                try:
                    task = self.ref.forward(rgb=rgb_ori, text=text)
                    targets = task.result.get('objects', [])
                except Exception as e:
                    print(f"Refer检测失败: {e}")

                fts['targets'] = targets
                if not targets:
                    print("⚠ Refer未找到目标")
                    continue  # 跳过本次循环，等待下一帧



                chosen_aff = self.choose(rgb_ori, depth, affs, fts=fts)
            
            else:
                # 不在初始位置，跳过检测
                affs = [{}]  # 空的检测结果
                chosen_aff = None
                # 可选：打印提示信息
                if hasattr(self, '_last_position_warning_time'):
                    if time.time() - self._last_position_warning_time > 5:  # 每5秒最多提示一次
                        print("⚠ 机械臂不在初始位置，跳过检测")
                        self._last_position_warning_time = time.time()
                else:
                    print("⚠ 机械臂不在初始位置，跳过检测")
                    self._last_position_warning_time = time.time()
            point_in_base = None
            if chosen_aff is not None and (chosen_aff['aff'][0] >= rgb.shape[1] or chosen_aff['aff'][1] >= rgb.shape[0]):
                print(f'out of scene: {chosen_aff}')
                chosen_aff = rect = offset = point3d = angle = width3d = None

            if chosen_aff is not None:
                data3d = self.camera.unprj(aff=chosen_aff, rgb=rgb, depth=depth)  # point in camera frame
                point3d = data3d['point3d']
                width3d = data3d['width3d']
                # offset = self.arm.offset2tool_in_base(point3d_in_cam=point3d)  # 
                
                data = self.arm.offset_from_end(point3d=point3d, end_pos=end_pos, src_frame='cam')
                offset = data['offset']
                point_in_base = data['point_in_base']
                eef_in_base = data['eef_in_base']
                rect = chosen_aff['aff']
                rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4] * 180 / math.pi)
                angle = rect[2]
                
                
                ### 增加工作范围过滤 - Piper gripper 安全工作范围
                # 基于 URDF 文件计算的 Piper 机械臂实际工作空间（单位：米）
                # 连杆总长: base(0.123) + link2(0.285) + link3(0.273) + link5(0.091) + gripper(0.136) ≈ 0.908m
                min_reach = 0.250
                max_reach = 0.600
                min_height = -0.270
                max_height = 0.300
                
                # 计算目标点到基座的水平距离
                if point_in_base is not None:
                    horizontal_distance = np.sqrt(point_in_base[0]**2 + point_in_base[1]**2)
                    z_height = point_in_base[2]
                    
                    # 检查是否在安全工作范围内
                    in_workspace = (min_reach <= horizontal_distance <= max_reach and 
                                  min_height <= z_height <= max_height)
                    
                    if not in_workspace:
                        # 超出工作范围，清空所有抓取相关变量
                        print(f"⚠ 目标超出工作范围: 距离={horizontal_distance:.3f}m (安全范围: {min_reach}-{max_reach}m), "
                              f"高度={z_height:.3f}m (安全范围: {min_height}-{max_height}m)")
                        chosen_aff = rect = offset = point3d = angle = width3d = None
                      
            else:  # cannot find a valid aff for given fts
                chosen_aff = rect = offset = point3d = angle = width3d = None

            state['affs'] = affs
            state['chosen_aff'] = chosen_aff
            state['fts'] = fts
            state['cmd'] = cmd
            state['point3d'] = point3d
            state['offset'] = offset
            state['point_in_base'] = point_in_base
            state['eef_in_base'] = eef_in_base
            state['category'] = chosen_aff.get('category', 'unknown') if chosen_aff else None
            if chosen_aff != None:
                try:
                    vis_rgb = self.vis_frame(rgb=rgb, depth=depth, state=state)
                    cv2.imshow(self.cfg.window_name, vis_rgb)
                    cv2.waitKey(1)
                except Exception as e:
                    print(f"可视化失败: {e}")
                   


            
            
            key = cv2.waitKey(1)
            if self.mode.startswith('arm_'):  # arm is doing ops, mode is set in listen_on_arm_ops
                # when ops is done, mode is set to be not arm_, gurantee by listen_on_arm_ops
                continue
            
            '''
            if key & 0xFF == ord('q'):
                self.vis.stop()
                time.sleep(3.0)
                cv2.destroyAllWindows()
                break
            elif key & 0xFF == ord('t'):  # ref to an object
                self.mode = 'ref'
                cmd = text = input('\n\nprompt->')
                task = self.ref.forward(rgb=rgb_ori, text=text)  # time consuming, delay in 3s to 6s
                targets = task.result
                targets = targets['objects']
                fts['targets'] = targets
            '''

            action = 'pick'
            # 添加类别和3D信息到参数中
            category = chosen_aff.get('category', 'unknown') if chosen_aff else 'unknown'
            para = dict(offset=offset, angle=angle, gripper_width=width3d, 
                       category=category, point3d=point3d, point_in_base=point_in_base)    
            if key != -1:
                just_change_fts = True
            if key & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            elif key & 0xFF == ord('a'):
                self.mode = 'auto'
                fts.clear()
            elif key & 0xFF == ord('g'):
                self.mode = 'confirm'
                self.add_arm_ops(action=action, para=para)  # make sure, mode is starts with 'arm_'
            elif key & 0xFF == ord('w'):  # freeze
                self.mode = 'wait'
                key = cv2.waitKey()
                if key & 0xFF == ord('g'):
                    self.add_arm_ops(action=action, para=para)
            elif key & 0xFF == ord('t'):  # ref to an object
                self.mode = 'ref'
                cmd = text = 'toy cubic.remote.bottle'
                # 只有在初始位置才执行参考检测
                if self.is_at_init_position(tolerance=20):
                    targets = []
                    try:
                        task = self.ref.forward(rgb=rgb_ori, text=text)
                        targets = task.result.get('objects', [])
                    except Exception as e:
                        print(f"Refer检测失败: {e}")
                    fts['targets'] = targets
                    if not targets:
                        print("⚠ Refer未找到目标")
                else:
                    print("⚠ 机械臂不在初始位置，无法执行参考检测")
                    fts['targets'] = []


            elif key & 0xFF == ord('v'):  # sound
                self.mode = 'voice'
                if self.voc is None:
                    print('voc is not enabled')
                    sys.exit(1)
            elif key == ord('c'):
                self.mode = 'confirm'
                fts.clear()
            elif key == -1:
                if self.mode in ['auto', 'ref']:  # set fts once and use it for the following
                    self.add_arm_ops(action=action, para=para)
                    
                    fts.clear()    ####清理 fts
                    
                elif self.mode in ['voice']:  # two things to do: pick if previous loop has set condition; accept new condition by voice
                    if cmd is not None and 'targets' in fts and chosen_aff is not None:
                        self.add_arm_ops(action=action, para=para)
                    else:
                        # keep accepting new fts inputs via voice with non-blocking style
                        try:
                            cmd = text = self.voc.cmd_q.get(timeout=0.1)
                            print(f'demo gets voice command: {text}')
                            if self.is_at_init_position(tolerance=20):
                                targets = []
                                try:
                                    task = self.ref.forward(rgb=rgb_ori, text=text)
                                    targets = task.result.get('objects', [])
                                except Exception as e:
                                    print(f"语音Refer检测失败: {e}")

                                if len(targets) > 0:
                                    fts['targets'] = targets
                                    print(f'ref: {targets} via {cmd=}')
                                    just_change_fts = False
                                else:
                                    print(f'no ref via {cmd=}')
                                    cmd = None
                            else:
                                print("⚠ 机械臂不在初始位置，无法执行语音参考检测")
                                cmd = None
                        except queue.Empty:
                            pass
                        except Exception as e:
                            print(f'{e}')
                just_change_fts = False
            else:
                self.mode = 'wait'
            
   
def init_robot():
    img_width = 1280
    img_height = 720

    cfg = Config()
    cfg.server_list = r'config/server_grasp.json'
    cfg.model_name = 'full'
    det = GraspAnythingOnline(cfg)

   

    cfg = Config()
    cfg.can_name = 'can0'
    #cfg.init_pos = dict(x=270, y=0, z=307, roll=-180, pitch=0, yaw=0)   # xarm bev
    #cfg.init_pos = dict(x=270, y=0, z=307)
    #cfg.init_pos = dict(x=345, y=0, z=380, roll=180, pitch=40, yaw=180)  # piper bev flange
    cfg.init_pos =dict(x=400, y=0, z=150, roll=180, pitch=30, yaw=180)  # piper bev, gripper center
    


    ### 20260109
    cfg.T_cam2flan = [-0.07583874846136467,0.05292914229358449,0.05706895145407312]
    q = [0.1558761610824957,-0.16214154506130807,0.6975255175538586,-0.6803461575790205]

    cfg.R_cam2flan = R.from_quat(q).as_matrix().tolist()
    cfg.T_gripper2flan = [0.0, 0.0, 0.13503]
    #cfg.T_gripper2flan = [0.0, 0.0, 0.1000]
    arm = ArmRobot(cfg)
    arm.connect()



    cfg = Config()
    cfg.device_id = 337122071190  # hand camera id 
    cfg.width = img_width
    cfg.height = img_height
    cfg.fps = 6
    camera = RealSenseCamera(cfg=cfg)
    camera.connect()


    cfg = Config()
    ref = DinoXDetectorOnline(cfg)
    
    
    voc = None
    '''
    cfg = Config()
    cfg.hotword = 'bottle'
    cfg.keyword = 'bottle'
    voc = VoiceInput(cfg=cfg)
    asr_thread = threading.Thread(target=voc.asr_process)
    asr_thread.daemon = True
    asr_thread.start()
    listen_thread = threading.Thread(target=voc.listen)
    listen_thread.daemon = True
    listen_thread.start()
    '''
    
    tracker = None
    '''
    cfg = Config()
    cfg.aspect_ratio_thresh = 1.6
    cfg.min_box_area = 10
    cfg.track_buffer = 300
    cfg.track_thresh = 0.15
    cfg.match_thresh = 0.8  # only dist <= match_thresh are allowed
    cfg.mot20 = False
    # tracker = Tracker(cfg=cfg)
    '''

    # 底盘初始化
    chassis = None
    if IS_MOBILE:
        try:
            chassis = SimpleTracer()
            print("底盘初始化成功")
        except Exception as e:
            print(f"底盘初始化失败: {e}")
            print("将在无底盘模式下运行")

    cfg = Config()
    cfg.window_name = 'DINO-X Grasp'
    cfg.mode = 'ref'
    cfg.img_w = img_width
    cfg.img_h = img_height
    demo_robot = DemoRobot(cfg=cfg, camera=camera, det=det, arm=arm, ref=ref, voc=voc, tracker=tracker, chassis=chassis)

    return demo_robot

            


if __name__ == '__main__':
    dr = init_robot()
    
    robot_thread = threading.Thread(target=dr.listen_on_arm_ops)  # for demo.mode management, we keep it here
    robot_thread.daemon = True
    robot_thread.start()
    dr.start()
    '''
   
    dr.navi_grasp_show()
    '''
    
    
