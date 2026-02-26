import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from collections import UserDict
from typing import Optional, Union
import copy
import cv2
import dash
import h5py
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pyrealsense2 as rs
from camera import RealSenseCamera
from dash import Dash, Input, Output, callback, dcc, html
from mmengine.config import Config
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R
from robot import XArmRobot
import math
from pycocotools import mask as coco_mask
from mmcv.ops import box_iou_rotated
import numpy as np
import torch

def flatten(lst):
    """递归展开嵌套列表/元组为一维列表
    
    将任意嵌套的列表或元组结构展开为一维列表。
    支持任意层次的嵌套结构。
    
    Args:
        lst: 嵌套列表或元组
        
    Returns:
        list: 展开后的一维列表
        
    Example:
        >>> flatten([1, [2, 3], [[4, 5], 6]])
        [1, 2, 3, 4, 5, 6]
    """
    result = []
    for item in lst:
        if isinstance(item, list | tuple):
            result.extend(flatten(item))  # 递归展开嵌套结构
        else:
            result.append(item)           # 直接添加叶子节点
    return result

class SLAM:
    """物体级SLAM（同时定位与建图）系统
    
    与传统的点云SLAM不同，该类实现了面向物体的SLAM。
    将检测到的2D物体映射到3D空间，建立持久性的3D物体地图。
    
    主要功能：
    - 2D检测结果到3D映射
    - 跨帧物体关联匹配
    - 3D地图建立和更新
    - 实时可视化显示
    """
    def __init__(self, cfg):
        """初始化SLAM系统
        
        Args:
            cfg: 配置对象，包含：
                - plane_dpt: 桌面深度参考值（米）
                - web_port: Web可视化端口
        """
        self.cfg = cfg
        self.plane_dpt = cfg.get('plane_dpt', 0.48)  # 桌面深度参考值
        
        # === 数据存储结构 ===
        self.obj3ds = dict()     # 每帧的现时物体3D信息 {frame_id: [Object3D, ...]}
        self.obj_acc = dict()    # 累积的持久性物体3D地图 {frame_id: [Object3D, ...]}
        self.grippers = dict()   # 每帧的机械臂夹爪3D位置 {frame_id: gripper_3d_points}

    def forward(self, dt, depth=None):
        pass

    def demo_video(self, fp):
        pass

    def vis3d(self, sample_rate):
        """启动交互式3D可视化Web应用
        
        使用Dash和Plotly建立交互式的3D场景可视化。
        展示物体3D地图和机械臂位置的时序变化。
        
        Args:
            sample_rate: 采样率，控制显示的帧间隔
        """
        def create_cube_lines(center, size):
            """创建3D立方体的边框线条
            
            Args:
                center: 立方体中心 [x, y, z]
                size: 立方体尺寸 [length, width, height]
                
            Returns:
                tuple: (x_lines, y_lines, z_lines) 用于Plotly绘图
            """
            cx, cy, cz = center
            l, w, h = size
            # 8个顶点
            vertices = np.array(
                [
                    [cx - l / 2.0, cy - w / 2.0, cz - h / 2.0],
                    [cx - l / 2.0, cy - w / 2.0, cz + h / 2.0],
                    [cx - l / 2.0, cy + w / 2.0, cz - h / 2.0],
                    [cx - l / 2.0, cy + w / 2.0, cz + h / 2.0],
                    [cx + l / 2.0, cy - w / 2.0, cz - h / 2.0],
                    [cx + l / 2.0, cy - w / 2.0, cz + h / 2.0],
                    [cx + l / 2.0, cy + w / 2.0, cz - h / 2.0],
                    [cx + l / 2.0, cy + w / 2.0, cz + h / 2.0],
                ]
            )

            # 12条边
            edges = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]

            x_lines, y_lines, z_lines = [], [], []
            for edge in edges:
                x_lines.append([vertices[edge[0]][0], vertices[edge[1]][0]])
                y_lines.append([vertices[edge[0]][1], vertices[edge[1]][1]])
                z_lines.append([vertices[edge[0]][2], vertices[edge[1]][2]])

            return x_lines, y_lines, z_lines
        

        def create_gripper(p6):
            """创建机械臂夹爪的3D线框模型
            
            根据6个关键点创建夹爪的线框模型用于可视化。
            
            Args:
                p6: 6个关键点的3D坐标 [[x,y,z], ...]
                
            Returns:
                tuple: (x_lines, y_lines, z_lines) 用于Plotly绘图
                
            夹爪结构示意图：
                           5 (夹爪中心)
                1          2          3
                0 (基座)              4
            """
            vertices = p6  # 6个关键点坐标
            
            # 定义夹爪的连接关系
            edges = [[0,1], [1,2], [2,3], [3,4], [2,5]]  # 边的连接关系
            
            # 生成线段坐标供 Plotly 显示
            x_lines, y_lines, z_lines = [], [], []
            for edge in edges:
                start_idx, end_idx = edge[0], edge[1]
                x_lines.append([vertices[start_idx][0], vertices[end_idx][0]])
                y_lines.append([vertices[start_idx][1], vertices[end_idx][1]])
                z_lines.append([vertices[start_idx][2], vertices[end_idx][2]])

            return x_lines, y_lines, z_lines
        
        def create_lines(obj):
            vertices = obj.p8
        
            edges = [[0,1], [1,2],[2,3],[0,3],[4,5], [5,6],[6,7],[4,7], [0,4],[1,5],[2,6],[3,7]]
            x_lines, y_lines, z_lines = [], [], []
            for edge in edges:
                x_lines.append([vertices[edge[0]][0], vertices[edge[1]][0]])
                y_lines.append([vertices[edge[0]][1], vertices[edge[1]][1]])
                z_lines.append([vertices[edge[0]][2], vertices[edge[1]][2]])

            return x_lines, y_lines, z_lines
        app = dash.Dash(__name__)

        @app.callback(Output('3d-cube-plot', 'figure'), Input('range-slider', 'value'))
        def update_figure2(slider_range):
            fig = go.Figure()
            xlim = [0., 0.5]
            ylim = [-0.5, 0.5]
            zlim = [-0.025, 0.5]
            fig.update_xaxes(range=xlim)  # x-axis from 0 to 4
            fig.update_yaxes(range=ylim)
            # fig.update_zaxes(range=(-0.1, 0.3))
            if 0:
                # 示例：定义多个立方体（中心点, 尺寸）
                cubes = [
                    {'center': (0, 0, 0), 'size': (2, 1, 1), 'color': 'blue'},
                    {'center': (3, 1, 0.5), 'size': (1, 1, 1), 'color': 'red'},
                    {'center': (-2, 1, 1), 'size': (0.8, 0.8, 0.8), 'color': 'green'},
                    {'center': (1, -2, 0), 'size': (1.5, 0.5, 2), 'color': 'purple'},
                ]
            else:
                low, high = slider_range
                # frame_id = list(self.obj3ds.keys())[-1]
                frame_id = int(low)
                frame_id = int(frame_id / sample_rate) * sample_rate
                cubes = []
                if 0:
                    for frame_id in range(low, high):
                        objs = self.obj3ds[frame_id]
                        for obj in objs:
                            obj['color'] = 'blue'
                            cubes.append(obj)
                else:
                    objs = self.obj_acc[frame_id]
                    for obj in objs:
                        obj['color'] = 'blue'
                        cubes.append(obj)
                print(f'{frame_id=} {low=} {high=} obj#={len(cubes)}')
            # breakpoint()
            for cube in cubes:
                x_lines, y_lines, z_lines = create_lines(cube)
                for x, y, z in zip(x_lines, y_lines, z_lines):
                    fig.add_trace(
                        go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=cube['color'], width=6), showlegend=False, hoverinfo='none')
                    )

            x_lines, y_lines, z_lines = create_gripper(self.grippers[frame_id])
            for x, y, z in zip(x_lines, y_lines, z_lines):
                    fig.add_trace(
                        go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='red', width=6), showlegend=False, hoverinfo='none')
                    )
            # 设置布局
            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    xaxis=dict(range=xlim),
                    yaxis=dict(range=ylim),
                    zaxis=dict(
                        range=zlim,
                        backgroundcolor="lightgray",
                        gridcolor="white"
                    ),
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=2, z=1),  # 保持真实比例
                    # autosize=False,
                ),
                margin=dict(l=0, r=0, b=0, t=50),
                hovermode=False,
            )

            return fig
        
        
        N = max(self.obj3ds.keys())
        app.layout = html.Div(
            [
                html.H1('3D Cubes Visualization with Dash', style={'textAlign': 'center'}),
                # 3D 图形区域
                dcc.Graph(id='3d-cube-plot', style={'width': '100%', 'height': '80vh'}),
                # 控制按钮（可扩展）
                html.Div(
                    [
                        html.Button('Reset View', id='btn-reset', n_clicks=0),
                        html.P('Petal Width:'),
                        dcc.RangeSlider(id='range-slider', min=0, max=N, step=1, marks={0: '0', N: f'{N}'}, value=[0, N]),
                    ],
                    style={'textAlign': 'center', 'margin': '20px'},
                ),
            ]
        )


        # breakpoint()
        port = self.cfg.get('web_port', 8050)
        print(f'start web on {port}. {self.obj3ds.keys()}')
        app.run_server(host='0.0.0.0', debug=True, port=port)



    def update_objs(self, objs, frame_id):
        """更新物体地图并执行数据关联
        
        将当前帧的3D物体与已有地图进行关联，更新持久性物体地图。
        使用旋转IoU进行相似度计算和数据关联。
        
        Args:
            objs: 当前帧的3D物体列表
            frame_id: 帧ID
        """
        # === 为每个物体计算2D投影和旋转边界框 ===
        for obj in objs:
            # 提取3D物体的底面4个角点在XY平面的投影
            p4 = obj.p8[0:4, 0:2]  # 取前4个点的x,y坐标 (4,2)
            obj.p4 = p4
            
            # 使用OpenCV拟合最小外接矩形
            rect = cv2.minAreaRect(p4.astype(np.float32))  # 返回 (center, (w, h), angle)
            cx, cy = rect[0]  # 中心点坐标
            w, h = rect[1]    # 宽度和高度
            angle = rect[2]   # 旋转角度（OpenCV范围：[-90, 0)）
            
            # === 角度标准化处理 ===
            if w < h:
                w, h = h, w    # 交换宽高，确保宽度≥高度
                angle += 90    # 相应调整角度
            angle = angle % 180  # 角度标准化到[0, 180)
            angle = angle * math.pi / 180.0  # 度转弧度
            
            # 生成旋转边界框参数 [cx, cy, w, h, angle]
            rb = [cx, cy, w, h, angle]
            obj.rb = rb  # 保存到物体属性中

        # === 更新现时物体地图 ===
        self.obj3ds[frame_id] = objs
        
        # === 初始化情况：第一帧 ===
        if len(self.obj_acc) == 0:
            self.obj_acc[frame_id] = objs
            return
        
        # === 获取上一帧的累积物体 ===
        frame_id0 = max(self.obj_acc.keys())  # 最新帧ID
        olds = self.obj_acc[frame_id0]        # 上一帧的累积物体
        
        # === 边界情况处理 ===
        if len(objs) == 0:  # 当前帧无物体，保持原有物体
            self.obj_acc[frame_id] = olds
            return
        if len(olds) == 0:  # 上一帧无物体，直接使用当前帧
            self.obj_acc[frame_id] = objs
            return

        # === 数据关联算法 ===
        # 构造旋转边界框张量用于IoU计算
        old_boxes = torch.tensor([o.rb for o in olds], dtype=torch.float32)  # 上一帧物体
        new_boxes = torch.tensor([o.rb for o in objs], dtype=torch.float32)  # 当前帧物体
        
        # 计算旋转IoU矩阵 (N_old, N_new)
        ious = box_iou_rotated(old_boxes, new_boxes)
        ious = ious.numpy()
        
        # 对每个新物体，找到与它最相似的旧物体的IoU
        max_iou_per_obj = ious.max(axis=0)  # 形状: (N_new,)
        
        # 判定新物体：IoU < 0.3认为是新出现的物体
        new_obj_mask = (max_iou_per_obj < 0.3)
        
        if new_obj_mask.sum() > 0:
            # 提取新物体并加入地图
            new_objs = [objs[i] for i in range(len(objs)) if new_obj_mask[i]]
            self.obj_acc[frame_id] = olds + new_objs  # 合并旧物体和新物体
        else:
            # 无新物体，保持原有物体地图
            self.obj_acc[frame_id] = olds


        
        

        


    def fit_shape(self, mask):
        """从2D掉码拟匈3D物体形状
        
        使用轮廓检测和最小外接矩形拟合算法，
        从物体的分割掉码中提取2D形状参数。
        
        Args:
            mask: 二值化的物体掉码 (H, W)
            
        Returns:
            dict: 包含旋转边界框、角点和掉码的字典
        """
        # === 数据类型和内存布局处理 ===
        mask = mask.astype(np.uint8)  # 确保为uint8类型
        guess_mask = np.zeros_like(mask)
        guess_mask = np.ascontiguousarray(guess_mask)  # OpenCV要求内存连续
        
        # === 轮廓检测 ===
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL,        # 只检测外轮廓
            cv2.CHAIN_APPROX_SIMPLE   # 压缩轮廓，去除冗余点
        )
        
        # 选择最大轮廓（面积最大的连通区域）
        largest_contour = max(contours, key=cv2.contourArea)

        # === 最小外接矩形拟合 ===
        rect = cv2.minAreaRect(largest_contour)
        (cx, cy), (w, h), angle = rect  # 解析矩形参数
        
        # === 角度和尺寸标准化 ===
        if w < h:
            w, h = h, w    # 保证宽度 ≥ 高度
            angle += 90    # 相应调整角度
            angle = angle % 180  # 标准化到[0, 180)

        # 重组矩形参数
        rect_params = [(cx, cy), (w, h), angle]

        # === 获取矩形研4个角点 ===
        p4 = cv2.boxPoints(rect_params).astype(int)
        # 点的顺序：右下 → 左下 → 左上 → 右上
        
        # === 参数格式转换 ===
        rb_params = flatten(rect_params)  # [cx, cy, w, h, angle(degree)]
        rb_params[4] = rb_params[4] / 180 * math.pi  # 度转弧度
        
        # === 绘制拟合的矩形掉码 ===
        cv2.drawContours(guess_mask, [p4], -1, 255, -1)  # 填充矩形区域
        
        return dict(
            rb=rb_params,      # 旋转边界框参数 [cx, cy, w, h, angle_rad]
            points=p4,         # 4个角点坐标 (4, 2)
            mask=guess_mask    # 拟合的矩形掉码
        )

    def demo_exp_log(self, fp, cam, arm, sample_rate):
        video_fp = fp + '.mp4'
        state_fp = fp + '.log'
        dpt_fp = fp + '.h5'

        dpt_fi = h5py.File(dpt_fp)
        # breakpoint()
        dpts = dpt_fi.get('frames')  # (N, h, w)
        N, h, w = dpts.shape
        max_dim = max(h, w)

        
        state_fi = open(state_fp)
        # self.obj3ds = o3ds = dict()  # keyed by frame
        # breakpoint()
        for i in range(N):  # for each frame
            if i % 10 == 0:
                print(f'process frame {i}/{N-1}')
            
            # if i > 2: break
            
            if i % sample_rate != 0:
                next(state_fi)
                continue
            # print(f'process {i}')
            line = state_fi.readline()

            dt = line.split(':')
            frame_id = int(dt[0])
            assert frame_id == i, f'{frame_id=} {i=}'
            state = ':'.join(dt[1:])
            state = json.loads(state)
            
            # breakpoint()
            if cam.scale is None:
                cam.scale = state['camera.scale']
                K = state['camera.K']
                cam.intrinsics = intrinsics = rs.intrinsics()
                intrinsics.width = K['width']  # 图像宽度
                intrinsics.height = K['height']  # 图像高度
                intrinsics.ppx = K['ppx']  # 主点 x
                intrinsics.ppy = K['ppy']  # 主点 y
                intrinsics.fx = K['fx']  # 焦距 fx
                intrinsics.fy = K['fy']  # 焦距 fy
                intrinsics.model = rs.distortion.none  # 无畸变
                intrinsics.coeffs = K['coeffs']  # 畸变系数（全零）
            
            if 'arm.end_pos' in state:
                end_pos = state['arm.end_pos']
            elif 'arm.flan_pos' in state:
                flan_pos = state['arm.flan_pos']
                for j in range(3):
                    flan_pos[j] /= 1000.0
                end_pos = flan_pos
                # print(f'use arm.flan_pos as end_pos due to old log file')
            else:
                raise NotImplemented(f'not found: end_pos. {state.keys()}')
            
            # breakpoint()
            ps2 = []
            p6 = [[0, 0.04, 0], [0, 0.04, -0.05], [0, 0, -0.05], [0,-0.04,-0.05], [0,-0.04,0],[0,0,-0.1]]
            for point_in_end in p6:
                point_in_base = arm.point2base(point3d=point_in_end, src_frame='end', end_pos=end_pos)
                ps2.append(point_in_base)
            self.grippers[i] = ps2
            
            
            dpt = dpts[i]  # in m
            # dpt[dpt > 0.5] = 0
            # dpt[dpt<0.3] = 0
            o3d_per_frame = []
            if state['mode'] == 'arm_picking':
                self.update_objs(o3d_per_frame, frame_id=frame_id)
                continue

            for j in range(len(state['affs'][0])):  # for each obj
                obj = state['affs'][0][j]  # dict_keys(['scores', 'affs', 'dt_score', 'dt_bbox', 'dt_mask', 'reid_fea', 'touching_points'])
                bbox = obj['dt_bbox']  # (x1,y1,x2,y2)

                score = obj['dt_score']
                if score < 0.25:
                    continue
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > 300 * 300:
                    continue
            
                mask = obj['dt_rle']


                mask = coco_mask.decode(mask)
                mask = cv2.resize(mask, dsize=(max_dim, max_dim))  # , interpolation=cv2.INTERPOLATION_BILINEAR)
                mask = mask[:h, :w].astype(bool)
                mask = mask.astype(bool)
                if mask[0].sum() > 0 or mask[:, 0].sum() > 0 or mask[-2].sum() > 0 or mask[:, -2].sum() > 0:
                    continue
                # breakpoint()
                # mask = np.array(mask, dtype=bool)

                valid = mask & (dpt > 0.11) & (dpt < 10)

                valid = dpt[valid]
                if len(valid) == 0:
                    continue
                try:
                    p99 = np.percentile(valid, 99)
                    p1 = np.percentile(valid, 1)
                    p99 = np.percentile(valid, 10)
                    p99 = valid.min()
                except:
                    breakpoint()
                if 0:
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                    ps = [[bbox[0], bbox[1]], [bbox[1], bbox[3]], [cx, cy]]
                    # point3d = cam.unprj_point(dpt=p99, cx=cx, cy=cy)
                    points = cam.unprj_points(dpts=[p99] * 3, ps=ps)
                else:
                    fit = self.fit_shape(mask)
                    ps = fit['points'] #
                    # rb = fit['rb'] #(x, y, w, h, angle)
                    # breakpoint()
                    points = cam.unprj_points(dpts=[p99] * len(ps), ps=ps)

                # breakpoint()
                obj3d = Config()
                p1, p2, p3, p4 = points
                ps2 = []
                for point_in_cam in points:
                    point_in_base = arm.point2base(point3d=point_in_cam, src_frame='cam', end_pos=end_pos)
                    ps2.append(point_in_base)
                ps2 = np.array(ps2) #(4, 3)
                max_dpt = ps2[:,2].max()
                if max_dpt < -0.025:
                    # breakpoint()
                    continue
                ps2[:, 2] = max_dpt

                q4 = copy.deepcopy(ps2)
                q4[:, 2] = -0.025
                p8 = np.vstack([ps2, q4]) #(8, 3)
                # breakpoint()

                obj3d.p8 = p8.astype(np.float32)
                obj3d.score = score
                # obj3d.mask = mask
                # obj3d.points = points
                obj3d.bbox = bbox
                o3d_per_frame.append(obj3d)
            
            self.update_objs(o3d_per_frame, frame_id=frame_id)
            # if 22 <= i <= 24:
            #     breakpoint()
        # breakpoint()

if __name__ == '__main__':
    cfg = Config()
    cfg.window_name = 'Demo'
    cfg.q_size = 10
    cfg.img_h = 720
    cfg.img_w = 1280
    slam = SLAM(cfg)

    cfg = Config()
    cfg.device_id = 344322074267  # Example device ID
    cfg.width = 1280
    cfg.height = 720
    cfg.fps = 30
    cam = RealSenseCamera(cfg)

    cfg = Config()
    cfg.ip = '192.168.1.236'
    cfg.init_pos = dict(x=270, y=0, z=307, roll=-180, pitch=0, yaw=0)
    cfg.T_cam2flan = [0.06644488210075013, -0.034881058367930894, 0.023248884204089854]
    q = [
        0.000849727426002991,
        -0.002441518543559673,
        0.7125339065369926,
        0.7016329161218389,
    ]
    cfg.R_cam2flan = R.from_quat(q).as_matrix().tolist()
    cfg.T_gripper2flan = [0, 0, 0.172]
    arm = XArmRobot(cfg)

    sample_rate = 1
    fp = r'/comp_robot/dino-3D/grasp/Ours/xarm/0804/grasp'
    fp = r'/comp_robot/dino-3D/grasp/Ours/xarm/0811/pick4/grasp'
    # fp = r'/comp_robot/dino-3D/grasp/Ours/xarm/0811/movingarm/grasp'
    # fp = r'/comp_robot/dino-3D/grasp/Ours/xarm/0811/stable/grasp'
    # fp = r'/comp_robot/dino-3D/grasp/Ours/xarm/0811/stable2/grasp'
    slam.demo_exp_log(fp=fp, cam=cam, arm=arm, sample_rate=sample_rate)
    slam.vis3d(sample_rate=sample_rate)