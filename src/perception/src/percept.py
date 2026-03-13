#!/usr/bin/env python3
"""
在线检测服务客户端

包含多个在线 AI 服务的客户端实现：
- DinoXDetectorCloud: 云端 DINO-X 检测服务
- DinoXDetectorOnline: 本地 DINO-X 检测服务
- GraspAnythingOnline: 抓取检测服务
- SAM2TrackerOnline: SAM2 目标跟踪服务
- DepthOptimizerOnline: CDM 深度优化服务

这是一个纯 Python 模块，不依赖 ROS。
"""

import base64
import json
import math
import os
import sys
import time
from io import BytesIO
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import requests
import yaml
from PIL import Image
from pycocotools import mask as coco_mask

# 可选依赖：dds_cloudapi_sdk (仅 DinoXDetectorCloud 使用)
try:
    from dds_cloudapi_sdk import Client
    from dds_cloudapi_sdk import Config as DDSConfig
    from dds_cloudapi_sdk.tasks.v2_task import V2Task
    HAS_DDS_SDK = True
except ImportError:
    HAS_DDS_SDK = False


def flatten(lst):
    """递归展开嵌套列表"""
    result = []
    for item in lst:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


class LocalTaskResult:
    """模拟 V2Task 的结果对象，用于本地服务返回"""
    def __init__(self, result_dict):
        self.result = result_dict


class DinoXDetectorCloud:
    """云端 DINO-X 语义检测服务客户端

    基于 DINO-X 模型的语义理解服务，将文本描述转换为目标定位。
    支持中英文自然语言输入，返回目标的边界框坐标。

    需要安装 dds_cloudapi_sdk。
    """
    def __init__(self, cfg):
        """初始化云端检测服务

        Args:
            cfg: 配置对象，包含：
                - uri: API路径
                - status_uri: 状态查询路径
                - token: API访问令牌
        """
        if not HAS_DDS_SDK:
            raise ImportError("dds_cloudapi_sdk not installed. Run: pip install dds-cloudapi-sdk")

        self.uri = cfg.uri
        self.status_uri = cfg.status_uri
        self.token = cfg.token
        config = DDSConfig(self.token)
        self.client = Client(config)

    def forward(self, text: str, rgb: np.ndarray, **kwargs) -> Any:
        """根据文本描述定位目标

        Args:
            text: 目标描述文本，支持中英文
            rgb: RGB 图像 (numpy 数组)

        Returns:
            V2Task: 任务对象，包含检测结果
        """
        body = {
            'model': 'DINO-X-1.0',
            'image': None,
            'prompt': {'type': 'text', 'text': None},
            'targets': ['bbox']
        }

        pil_img = Image.fromarray(rgb)
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        buffer.seek(0)

        img_bytes = buffer.getvalue()
        b64 = base64.b64encode(img_bytes).decode('utf-8')
        b64 = f'data:image/jpg;base64,{b64}'

        body['image'] = b64
        body['prompt']['text'] = text

        task = V2Task(api_path=self.uri, api_body=body)
        self.client.run_task(task)
        return task


class DinoXDetectorOnline:
    """本地 DINO-X 语义检测服务客户端

    通过 HTTP API 访问本地部署的 DINO-X 服务。
    提供与 DinoXDetectorCloud 相同的接口。
    """
    def __init__(self, cfg):
        """初始化本地检测服务

        Args:
            cfg: 配置对象，包含：
                - url: 服务地址，默认 http://192.168.112.14:10086
                - min_score: 最小检测分数阈值，默认 0.25
                - iou_threshold: IoU 阈值，默认 0.5
                - jpeg_quality: JPEG 压缩质量，默认 50
                - resize: 图片缩放尺寸 (width, height)，默认 (1280, 720)
                - warmup: 预热次数，默认 0
        """
        self.base_url = getattr(cfg, 'url', 'http://192.168.112.14:10086')
        self.api_url = f"{self.base_url}/api/predict"
        self.min_score = getattr(cfg, 'min_score', 0.25)
        self.iou_threshold = getattr(cfg, 'iou_threshold', 0.5)
        self.jpeg_quality = getattr(cfg, 'jpeg_quality', 50)
        self.resize = getattr(cfg, 'resize', (1280, 720))

        # 创建不使用代理的 Session
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies = {'http': None, 'https': None}

        # 预热
        warmup_runs = getattr(cfg, 'warmup', 0)
        if warmup_runs > 0:
            self._warmup(warmup_runs)

    def _warmup(self, num_runs: int):
        """预热服务"""
        print(f"[DinoXDetectorOnline] Warmup ({num_runs} runs)...")
        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in range(num_runs):
            self.forward('object', dummy_img)
            print(f"  warmup {i+1}/{num_runs}")
        print("[DinoXDetectorOnline] Warmup complete")

    def forward(self, text: str, rgb: np.ndarray, _timing: Optional[Dict] = None, **kwargs) -> LocalTaskResult:
        """根据文本描述定位目标

        Args:
            text: 目标描述文本，多个类别用 '.' 分隔
            rgb: RGB/BGR 图像 (numpy 数组)
            _timing: 可选，用于记录内部耗时
            **kwargs: 额外参数 (min_score, iou_threshold)

        Returns:
            LocalTaskResult: 包含检测结果
        """
        t0 = time.time()

        min_score = kwargs.get('min_score', self.min_score)
        iou_threshold = kwargs.get('iou_threshold', self.iou_threshold)

        # 记录原始尺寸
        orig_h, orig_w = rgb.shape[:2]
        scale_x, scale_y = 1.0, 1.0

        # Resize
        if self.resize is not None:
            target_w, target_h = self.resize
            rgb_resized = cv2.resize(rgb, (target_w, target_h))
            scale_x = orig_w / target_w
            scale_y = orig_h / target_h
        else:
            rgb_resized = rgb

        t1 = time.time()

        # JPEG 编码
        _, img_encoded = cv2.imencode('.jpg', rgb_resized, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        img_bytes = img_encoded.tobytes()

        t2 = time.time()

        # 发送请求
        files = {'images': ('image.jpg', img_bytes, 'image/jpeg')}
        data = {
            'text_prompt': text,
            'min_score': min_score,
            'iou_threshold': iou_threshold,
            'chosen_policy': 'det'
        }

        categories = [c.strip() for c in text.split('.')]

        try:
            response = self.session.post(self.api_url, files=files, data=data, timeout=30)
            t3 = time.time()

            if _timing is not None:
                _timing['dinox_preprocess'] = (t1 - t0) * 1000
                _timing['dinox_encode'] = (t2 - t1) * 1000
                _timing['dinox_http'] = (t3 - t2) * 1000

            response.raise_for_status()
            result_json = response.json()

            objects = []
            if 'results' in result_json and len(result_json['results']) > 0:
                result = result_json['results'][0]
                for obj in result.get('objects', []):
                    cat_idx = obj.get('category')
                    if isinstance(cat_idx, int) and 0 <= cat_idx < len(categories):
                        category_name = categories[cat_idx]
                    else:
                        category_name = str(cat_idx)

                    bbox = obj.get('bbox')
                    if bbox and (scale_x != 1.0 or scale_y != 1.0):
                        bbox = [
                            bbox[0] * scale_x, bbox[1] * scale_y,
                            bbox[2] * scale_x, bbox[3] * scale_y
                        ]

                    obj_data = {
                        'bbox': bbox,
                        'score': obj.get('score'),
                        'category': category_name,
                    }
                    if 'mask' in obj:
                        obj_data['mask'] = obj['mask']
                    objects.append(obj_data)

            return LocalTaskResult({'objects': objects})

        except requests.exceptions.RequestException as e:
            print(f"[DinoXDetectorOnline] Request failed: {e}")
            return LocalTaskResult({'objects': [], 'error': str(e)})
        except json.JSONDecodeError as e:
            print(f"[DinoXDetectorOnline] JSON decode failed: {e}")
            return LocalTaskResult({'objects': [], 'error': str(e)})


class GraspAnythingOnline:
    """抓取检测服务客户端

    调用抓取检测模型进行目标检测和抓取点生成。
    支持多目标检测、实例分割、抓取姿态估计。
    """
    def __init__(self, cfg):
        """初始化抓取检测服务

        Args:
            cfg: 配置对象，包含：
                - server_list: 服务器列表文件路径
                - model_name: 使用的模型名称
                - resize: 图片缩放尺寸，默认 (1280, 720)
                - warmup: 预热次数，默认 0
        """
        self.resize = getattr(cfg, 'resize', (1280, 720))
        self.server_list_fp = cfg.server_list

        if not os.path.isfile(self.server_list_fp):
            raise FileNotFoundError(f"Server list not found: {self.server_list_fp}")

        with open(self.server_list_fp) as f:
            api_dict = json.load(f).get('backends', {})
            self.server_list = api_dict

        self.model_name = cfg.model_name
        if self.model_name not in self.server_list:
            raise ValueError(f"{self.model_name} not in {list(self.server_list.keys())}")

        self.url = self.server_list[self.model_name] + '/generate'

        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies = {'http': None, 'https': None}

        warmup_runs = getattr(cfg, 'warmup', 0)
        if warmup_runs > 0:
            self._warmup(warmup_runs)

    def _warmup(self, num_runs: int):
        """预热服务"""
        print(f"[GraspAnythingOnline] Warmup ({num_runs} runs)...")
        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in range(num_runs):
            self.forward(dummy_img)
            print(f"  warmup {i+1}/{num_runs}")
        print("[GraspAnythingOnline] Warmup complete")

    def forward(self, rgb, depth=None, _timing: Optional[Dict] = None, **kwargs):
        """执行抓取检测

        Args:
            rgb: RGB/BGR 图像
            depth: 深度图（可选）
            _timing: 可选，用于记录内部耗时
            **kwargs: 额外参数

        Returns:
            tuple: (检测结果, 填充后的图像)
        """
        t0 = time.time()

        # 处理图像输入
        if isinstance(rgb, str):
            img = np.array(Image.open(rgb).convert('RGB'))[:, :, ::-1]
        elif isinstance(rgb, np.ndarray):
            img = rgb
        elif isinstance(rgb, bytes):
            img = cv2.imdecode(np.frombuffer(rgb, dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            raise TypeError(f"Unsupported image type: {type(rgb)}")

        orig_h, orig_w = img.shape[:2]

        # 计算坐标转换比例
        scale_x, scale_y = 1.0, 1.0
        if self.resize is not None:
            target_w, target_h = self.resize
            scale_x = orig_w / target_w
            scale_y = orig_h / target_h
            img = cv2.resize(img, (target_w, target_h))

        h, w = img.shape[:2]
        max_dim = max(h, w)

        # Padding
        top, left = 0, 0
        bottom = max_dim - h
        right = max_dim - w
        padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)

        t1 = time.time()

        _, img_encoded = cv2.imencode('.jpg', padded_img, [cv2.IMWRITE_JPEG_QUALITY, 30])
        img_bytes = img_encoded.tobytes()

        t2 = time.time()

        response = self.session.post(self.url, files={'img_source': img_bytes})

        t3 = time.time()

        if _timing is not None:
            _timing['grasp_preprocess'] = (t1 - t0) * 1000
            _timing['grasp_encode'] = (t2 - t1) * 1000
            _timing['grasp_http'] = (t3 - t2) * 1000

        objs = response.json()

        if len(objs) != 1:
            return objs, padded_img

        if len(objs[0]) == 0:
            print("[GraspAPI] No objects detected!")
            return objs, padded_img

        obj0 = objs[0][0]

        t6 = time.time()
        if 'dt_mask' in obj0 and obj0['dt_mask'] is not None:
            rle_list = [obj['dt_mask'] for obj in objs[0]]
            for obj in objs[0]:
                obj['dt_rle'] = obj['dt_mask']

            masks_decoded = coco_mask.decode(rle_list)

            kernel = np.ones((5, 5), np.uint8)
            size = (max_dim, max_dim)

            for i, obj in enumerate(objs[0]):
                m = masks_decoded[:, :, i] if masks_decoded.ndim == 3 else masks_decoded
                m = cv2.resize(m, dsize=size)
                m = cv2.dilate(m, kernel, iterations=1)
                m = m[:h, :w].astype(bool)
                # 将 mask resize 回原始尺寸
                if scale_x != 1.0 or scale_y != 1.0:
                    m = cv2.resize(m.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                obj['dt_mask'] = m

        t7 = time.time()

        ret = objs
        if kwargs.get('bag', False):
            ret = self._extract_bag(ret, rgb, depth, **kwargs)
        if kwargs.get('use_touching_points', True):
            ret = self._post_process(ret, rgb, depth, **kwargs)

        # 将坐标转换回原始图像尺寸
        if scale_x != 1.0 or scale_y != 1.0:
            for obj in ret[0]:
                # 转换 affs: [cx, cy, w, h, angle]
                if 'affs' in obj:
                    for aff in obj['affs']:
                        aff[0] *= scale_x  # cx
                        aff[1] *= scale_y  # cy
                        aff[2] *= scale_x  # width
                        aff[3] *= scale_y  # height
                        # angle 不需要转换

                # 转换 dt_bbox: [x1, y1, x2, y2]
                if 'dt_bbox' in obj:
                    obj['dt_bbox'] = [
                        obj['dt_bbox'][0] * scale_x,
                        obj['dt_bbox'][1] * scale_y,
                        obj['dt_bbox'][2] * scale_x,
                        obj['dt_bbox'][3] * scale_y,
                    ]

                # 转换 touching_points: [[x1, y1], [x2, y2], ...]
                if 'touching_points' in obj:
                    for tp in obj['touching_points']:
                        for pt in tp:
                            pt[0] *= scale_x
                            pt[1] *= scale_y

        t8 = time.time()

        if _timing is not None:
            _timing['grasp_mask_decode'] = (t7 - t6) * 1000
            _timing['grasp_postproc'] = (t8 - t7) * 1000

        return ret, padded_img

    def _post_process(self, ret, rgb, depth, **kwargs):
        """后处理：计算抓取接触点"""
        for obj in ret[0]:
            if 'dt_mask' not in obj:
                continue
            m = obj['dt_mask']
            affs = obj['affs']
            touching_points = []
            mask_h, mask_w = m.shape

            for rb in affs:
                xc, yc, w, h, angle2 = rb
                p1, p2, p3, p4 = cv2.boxPoints(((xc, yc), (w, h), angle2 * 180 / math.pi))
                c1 = [int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)]
                c2 = [int((p4[0] + p3[0]) / 2), int((p4[1] + p3[1]) / 2)]

                line_pts = self._bresenham_line(c1[0], c1[1], c2[0], c2[1])

                valid_pts = []
                for px, py in line_pts:
                    if 0 <= px < mask_w and 0 <= py < mask_h and m[py, px]:
                        valid_pts.append((px, py))

                if len(valid_pts) < 2:
                    ps = [c1, c2]
                else:
                    pt1 = list(valid_pts[0])
                    pt2 = list(valid_pts[-1])
                    ps = [pt1, pt2]
                    rb[0] = (pt1[0] + pt2[0]) / 2.0
                    rb[1] = (pt1[1] + pt2[1]) / 2.0
                    rb[2] = math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
                    rb[3] = 30
                touching_points.append(ps)
            obj['touching_points'] = touching_points
        return ret

    @staticmethod
    def _bresenham_line(x0, y0, x1, y1):
        """Bresenham 线段算法"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    def _extract_bag(self, ret, rgb, depth, **kwargs):
        """提取袋子抓取点"""
        ret0 = ret[0]
        img_h, img_w = rgb.shape[:2]
        for obj in ret0:
            mask = obj['dt_mask']
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            largest_contour = max(contours, key=cv2.contourArea)
            guess = cv2.minAreaRect(largest_contour)
            (cx, cy), (w, h), angle = guess
            if w < h:
                w, h = h, w
                angle += 90
                angle = angle % 180

            guess = [(cx, cy), (w, h), angle]
            p1, p2, p3, p4 = cv2.boxPoints(guess).astype(int)
            guess = flatten(guess)
            guess[4] = guess[4] / 180 * math.pi

            c1 = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
            c2 = [(p4[0] + p3[0]) / 2, (p4[1] + p3[1]) / 2]
            c1 = [int(x) for x in c1]
            c2 = [int(x) for x in c2]
            c3 = [0.2 * c2[0] + 0.8 * c1[0], 0.2 * c2[1] + 0.8 * c1[1]]
            c4 = [0.2 * c1[0] + 0.8 * c2[0], 0.2 * c1[1] + 0.8 * c2[1]]
            c3 = [int(x) for x in c3]
            c4 = [int(x) for x in c4]

            if c3[0] >= img_w or c3[1] >= img_h or c4[0] >= img_w or c4[1] >= img_h:
                aff = None
            else:
                if mask[c3[1], c3[0]] and (not mask[c4[1], c4[0]]):
                    aff = [*c2, max(abs(c4[0] - c2[0]), abs(c4[1] - c2[1])), 30, guess[4]]
                elif (not mask[c3[1], c3[0]]) and mask[c4[1], c4[0]]:
                    aff = [*c1, max(abs(c3[0] - c1[0]), abs(c3[1] - c1[1])), 30, guess[4]]
                else:
                    aff = None

            if aff is not None:
                obj['scores'][-1] = 1.0
                obj['affs'][-1] = aff

        return ret


class SAM2TrackerOnline:
    """SAM2 目标跟踪服务客户端

    基于 SAM2 的目标跟踪服务，支持多目标跟踪和实例分割。
    """
    def __init__(self, cfg):
        """初始化跟踪服务

        Args:
            cfg: 配置对象，包含：
                - url: 服务地址，默认 http://192.168.112.14:11086
                - resize: 图像缩放尺寸，默认 (512, 512)
                - jpeg_quality: JPEG 压缩质量，默认 30
                - warmup: 预热次数，默认 0
        """
        self.base_url = getattr(cfg, 'url', 'http://192.168.112.14:11086')
        self.api_url = f'{self.base_url}/api/predict'
        self.resize = getattr(cfg, 'resize', (512, 512))
        self.jpeg_quality = getattr(cfg, 'jpeg_quality', 30)

        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies = {'http': None, 'https': None}

        self._scale_x = 1.0
        self._scale_y = 1.0
        self._orig_size = None

        warmup_runs = getattr(cfg, 'warmup', 0)
        if warmup_runs > 0:
            self._warmup(warmup_runs)

    def _warmup(self, num_runs: int):
        """预热服务"""
        print(f"[SAM2TrackerOnline] Warmup ({num_runs} runs)...")
        dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
        dummy_dets = [{'dt_bbox': [100, 100, 200, 200], 'dt_score': 0.9, 'name': 'object'}]
        for i in range(num_runs):
            self.forward(dummy_img, dets=dummy_dets, frame_idx=0)
            print(f"  warmup {i+1}/{num_runs}")
        print("[SAM2TrackerOnline] Warmup complete")

    def forward(self, rgb, dets=None, frame_idx: int = 0, **kwargs) -> Dict:
        """逐帧跟踪

        Args:
            rgb: RGB/BGR 图像
            dets: 检测结果列表（首帧必须提供）
            frame_idx: 帧索引，0 表示首帧

        Returns:
            dict: 跟踪结果
        """
        # 处理图像输入
        if isinstance(rgb, str):
            img = cv2.imread(rgb)
            if img is None:
                return {'success': False, 'error': f'Cannot read image: {rgb}'}
        elif isinstance(rgb, np.ndarray):
            img = rgb
        elif isinstance(rgb, bytes):
            img = cv2.imdecode(np.frombuffer(rgb, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return {'success': False, 'error': 'Cannot decode image'}
        else:
            return {'success': False, 'error': f'Unsupported image type: {type(rgb)}'}

        orig_h, orig_w = img.shape[:2]
        self._orig_size = (orig_w, orig_h)

        target_w, target_h = self.resize
        self._scale_x = orig_w / target_w
        self._scale_y = orig_h / target_h

        img_resized = cv2.resize(img, (target_w, target_h))

        # 缩放检测框
        dets_scaled = None
        if dets is not None:
            dets_scaled = []
            for det in dets:
                scaled_det = {
                    'dt_bbox': [
                        det['dt_bbox'][0] / self._scale_x,
                        det['dt_bbox'][1] / self._scale_y,
                        det['dt_bbox'][2] / self._scale_x,
                        det['dt_bbox'][3] / self._scale_y,
                    ],
                    'dt_score': det.get('dt_score', 0.9),
                    'name': det.get('name', 'object'),
                }
                dets_scaled.append(scaled_det)

        _, img_encoded = cv2.imencode('.jpg', img_resized, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        img_bytes = img_encoded.tobytes()

        files = {'img_source': ('frame.jpg', img_bytes, 'image/jpeg')}
        data = {'frame_idx': str(frame_idx)}
        if dets_scaled is not None:
            data['dets'] = json.dumps(dets_scaled)

        try:
            response = self.session.post(self.api_url, files=files, data=data, timeout=30)
            response.raise_for_status()
            result = response.json()

            if 'error' in result:
                return {'success': False, 'error': result['error']}

            if not result or len(result) == 0:
                return {'success': True, 'ids': [], 'boxes': [], 'scores': [],
                        'track_scores': [], 'cats': [], 'masks': []}

            # 缩放回原始尺寸
            if 'boxes' in result and len(result['boxes']) > 0:
                scaled_boxes = []
                for box in result['boxes']:
                    scaled_boxes.append([
                        box[0] * self._scale_x, box[1] * self._scale_y,
                        box[2] * self._scale_x, box[3] * self._scale_y,
                    ])
                result['boxes'] = scaled_boxes

            result['success'] = True
            return result

        except requests.exceptions.RequestException as e:
            print(f"[SAM2TrackerOnline] Request failed: {e}")
            return {'success': False, 'error': str(e)}
        except json.JSONDecodeError as e:
            print(f"[SAM2TrackerOnline] JSON decode failed: {e}")
            return {'success': False, 'error': str(e)}


class DepthOptimizerOnline:
    """CDM 深度图优化服务客户端

    基于 CDM (Depth Map Denoising with RGB-D Fusion) 的深度图优化服务。
    通过 RGB 图像引导对深度图进行去噪和修复。
    """
    def __init__(self, cfg):
        """初始化深度优化服务

        Args:
            cfg: 配置对象，包含：
                - url: 服务地址，默认 http://192.168.112.14:8082
                - chosen_policy: 输出策略 ('dn,vis', 'dn', 'vis')
                - target_long: letterbox 长边目标尺寸，默认 798
                - target_short: letterbox 短边目标尺寸，默认 448
                - warmup: 预热次数，默认 0
        """
        self.base_url = getattr(cfg, 'url', 'http://192.168.112.14:8082')
        self.api_url = f'{self.base_url}/api/predict'
        self.chosen_policy = getattr(cfg, 'chosen_policy', 'dn,vis')
        self.target_long = getattr(cfg, 'target_long', 798)
        self.target_short = getattr(cfg, 'target_short', 448)

        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies = {'http': None, 'https': None}

        # FoundationStereo 配置
        self.fs_url = getattr(cfg, 'fs_url', 'http://192.168.112.14:8084')
        self.fs_api_url = f'{self.fs_url}/api/predict'
        self.fs_vis_url = f'{self.fs_url}/api/predict_vis'

        warmup_runs = getattr(cfg, 'warmup', 0)
        if warmup_runs > 0:
            self._warmup(warmup_runs)

    def _warmup(self, num_runs: int):
        """预热服务"""
        print(f"[DepthOptimizerOnline] Warmup ({num_runs} runs)...")
        dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_depth = np.zeros((480, 640), dtype=np.uint16)
        for i in range(num_runs):
            self.forward(dummy_rgb, dummy_depth)
            print(f"  warmup {i+1}/{num_runs}")
        print("[DepthOptimizerOnline] Warmup complete")

    def _letterbox(self, img, target_long, target_short, fill_value=0):
        """Letterbox padding，保持宽高比缩放到目标矩形尺寸

        Args:
            img: 输入图像 (H, W) 或 (H, W, C)
            target_long: 长边目标尺寸
            target_short: 短边目标尺寸
            fill_value: 填充值

        Returns:
            tuple: (padded_img, scale, pad_top, pad_left, target_h, target_w)
        """
        h, w = img.shape[:2]

        # 确定目标尺寸（根据原图方向）
        if w >= h:
            target_w, target_h = target_long, target_short
        else:
            target_w, target_h = target_short, target_long

        # 计算缩放比例（保持宽高比，取较小值确保不超出目标框）
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize
        if len(img.shape) == 3:
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Padding
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left

        if len(img.shape) == 3:
            padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                        cv2.BORDER_CONSTANT, value=(fill_value, fill_value, fill_value))
        else:
            padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                        cv2.BORDER_CONSTANT, value=fill_value)

        return padded, scale, pad_top, pad_left, target_h, target_w

    def _unletterbox(self, img, orig_h, orig_w, scale, pad_top, pad_left):
        """去除 letterbox padding，恢复原始尺寸

        Args:
            img: letterbox 后的图像
            orig_h, orig_w: 原始尺寸
            scale: 缩放比例
            pad_top, pad_left: padding 偏移

        Returns:
            恢复到原始尺寸的图像
        """
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)

        # 去除 padding
        cropped = img[pad_top:pad_top + new_h, pad_left:pad_left + new_w]

        # Resize 回原始尺寸
        if len(img.shape) == 3:
            restored = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            restored = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return restored

    def forward(self, rgb, depth, chosen_policy: Optional[str] = None,
                _timing: Optional[Dict] = None, **kwargs) -> Dict:
        """优化深度图

        Args:
            rgb: RGB/BGR 图像
            depth: 深度图 (uint16, 单位 mm)
            chosen_policy: 输出策略
            _timing: 可选，用于记录内部耗时

        Returns:
            dict: 包含优化后的深度图和可视化结果
        """
        t0 = time.time()
        policy = chosen_policy or self.chosen_policy

        # 读取图像（如果是文件路径或 bytes）
        if isinstance(rgb, str):
            rgb = cv2.imread(rgb)
        elif isinstance(rgb, bytes):
            rgb = cv2.imdecode(np.frombuffer(rgb, dtype=np.uint8), cv2.IMREAD_COLOR)

        if isinstance(depth, str):
            depth = cv2.imread(depth, cv2.IMREAD_UNCHANGED)
        elif isinstance(depth, bytes):
            depth = cv2.imdecode(np.frombuffer(depth, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        # 确保深度图是 uint16 格式
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16)

        # 记录原始尺寸
        orig_h, orig_w = rgb.shape[:2]

        # Letterbox padding 到 target_long x target_short
        rgb_padded, scale, pad_top, pad_left, _, _ = self._letterbox(
            rgb, self.target_long, self.target_short, fill_value=0)
        depth_padded, _, _, _, _, _ = self._letterbox(
            depth, self.target_long, self.target_short, fill_value=0)

        t1 = time.time()

        # 编码 RGB
        _, rgb_encoded = cv2.imencode('.jpg', rgb_padded, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rgb_bytes = rgb_encoded.tobytes()
        rgb_filename = 'rgb.jpg'

        # 编码深度图
        _, depth_encoded = cv2.imencode('.png', depth_padded)
        depth_bytes = depth_encoded.tobytes()
        depth_filename = 'depth.png'

        t2 = time.time()

        files = {
            'rgb': (rgb_filename, rgb_bytes, 'image/jpeg'),
            'dpt': (depth_filename, depth_bytes, 'image/png'),
        }
        data = {'chosen_policy': policy}

        # 保存 unletterbox 参数供后处理使用
        self._last_letterbox_params = (orig_h, orig_w, scale, pad_top, pad_left)

        try:
            response = self.session.post(self.api_url, files=files, data=data, timeout=60)
            t3 = time.time()

            # 检查 Content-Type，兼容两种 API
            content_type = response.headers.get('Content-Type', '')

            if 'image/png' in content_type:
                # 新 API（8081端口）：直接返回 PNG 二进制
                if not response.ok:
                    return {'success': False, 'error': f'HTTP {response.status_code}'}

                depth_arr = cv2.imdecode(
                    np.frombuffer(response.content, dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED  # 保持 16-bit
                )

                # Unletterbox: 恢复到原始尺寸
                depth_arr = self._unletterbox(depth_arr, orig_h, orig_w, scale, pad_top, pad_left)

                t4 = time.time()
                if _timing is not None:
                    _timing['cdm_preprocess'] = (t1 - t0) * 1000
                    _timing['cdm_encode'] = (t2 - t1) * 1000
                    _timing['cdm_http'] = (t3 - t2) * 1000
                    _timing['cdm_postprocess'] = (t4 - t3) * 1000

                return {
                    'success': True,
                    'depth': depth_arr,
                    'chosen_policy': policy,
                }
            else:
                # 旧 API（8086端口）：返回 JSON + base64
                result = response.json()
                t4 = time.time()

                if not response.ok or result.get('error'):
                    error_msg = result.get('error', f'HTTP {response.status_code}')
                    print(f"[DepthOptimizerOnline] Server error: {error_msg}")
                    return {'success': False, 'error': error_msg}

                if not result.get('success'):
                    return {'success': False, 'error': result.get('error', 'Unknown error')}

                output = {
                    'success': True,
                    'device': result.get('device'),
                    'original_resolution': result.get('original_resolution'),
                    'depth_resolution': result.get('depth_resolution'),
                    'chosen_policy': result.get('chosen_policy'),
                }

                # 解码深度图并还原尺寸
                if result.get('depth'):
                    depth_b64 = result['depth']
                    depth_data = base64.b64decode(depth_b64)
                    depth_arr = cv2.imdecode(
                        np.frombuffer(depth_data, dtype=np.uint8),
                        cv2.IMREAD_UNCHANGED
                    )
                    # Unletterbox 还原原始尺寸
                    depth_arr = self._unletterbox(depth_arr, orig_h, orig_w, scale, pad_top, pad_left)
                    output['depth'] = depth_arr

                # 解码可视化图像并还原尺寸
                if result.get('vis_images') and len(result['vis_images']) > 0:
                    vis_b64 = result['vis_images'][0]
                    vis_data = base64.b64decode(vis_b64)
                    vis_arr = cv2.imdecode(
                        np.frombuffer(vis_data, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    # Unletterbox 还原原始尺寸
                    vis_arr = self._unletterbox(vis_arr, orig_h, orig_w, scale, pad_top, pad_left)
                    output['vis_image'] = vis_arr

                t5 = time.time()

                if _timing is not None:
                    _timing['cdm_preprocess'] = (t1 - t0) * 1000
                    _timing['cdm_encode'] = (t2 - t1) * 1000
                    _timing['cdm_http'] = (t3 - t2) * 1000
                    _timing['cdm_postprocess'] = (t5 - t4) * 1000

                return output

        except requests.exceptions.RequestException as e:
            print(f"[DepthOptimizerOnline] Request failed: {e}")
            return {'success': False, 'error': str(e)}
        except json.JSONDecodeError as e:
            print(f"[DepthOptimizerOnline] JSON decode failed: {e}")
            return {'success': False, 'error': str(e)}

    def optimize_depth(self, rgb, depth, **kwargs) -> Optional[np.ndarray]:
        """便捷接口：仅返回优化后的深度图"""
        result = self.forward(rgb, depth, chosen_policy='dn', **kwargs)
        if result.get('success') and 'depth' in result:
            return result['depth']
        return None

    def get_visualization(self, rgb, depth, **kwargs) -> Optional[np.ndarray]:
        """便捷接口：仅返回可视化图像"""
        result = self.forward(rgb, depth, chosen_policy='vis', **kwargs)
        if result.get('success') and 'vis_image' in result:
            return result['vis_image']
        return None

    def forward_stereo(self, ir_left, ir_right, intrinsics,
                       rgb=None, raw_depth=None,
                       get_vis=False, _timing=None) -> Dict:
        """被动双目深度估计（FoundationStereo）

        Args:
            ir_left: 左目 IR 灰度图（numpy array / 文件路径 str / bytes）
            ir_right: 右目 IR 灰度图（numpy array / 文件路径 str / bytes）
            intrinsics: 相机内参（文件路径 str / dict / bytes）
            rgb: 可选 RGB 图像（numpy array / 文件路径 str / bytes）
            raw_depth: 可选原始深度图（numpy array / 文件路径 str / bytes）
            get_vis: 是否同时获取可视化图像
            _timing: 可选，用于记录内部耗时

        Returns:
            dict: {'success': True, 'depth': np.ndarray(uint16), 'mode': 'foundation_stereo', ...}
                  或 {'success': False, 'error': str}
        """
        t0 = time.time()

        def _encode_ir(img):
            if isinstance(img, str):
                img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            elif isinstance(img, bytes):
                img = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            _, buf = cv2.imencode('.png', img)
            return buf.tobytes()

        def _encode_color(img):
            if isinstance(img, str):
                img = cv2.imread(img, cv2.IMREAD_COLOR)
            elif isinstance(img, bytes):
                img = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)
            _, buf = cv2.imencode('.png', img)
            return buf.tobytes()

        def _encode_depth(img):
            if isinstance(img, str):
                img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            elif isinstance(img, bytes):
                img = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            _, buf = cv2.imencode('.png', img)
            return buf.tobytes()

        def _encode_intrinsics(intr):
            if isinstance(intr, str):
                with open(intr, 'rb') as f:
                    return f.read()
            elif isinstance(intr, dict):
                return yaml.dump(intr).encode()
            elif isinstance(intr, bytes):
                return intr
            raise TypeError(f'Unsupported intrinsics type: {type(intr)}')

        try:
            ir_left_bytes = _encode_ir(ir_left)
            ir_right_bytes = _encode_ir(ir_right)
            intrinsics_bytes = _encode_intrinsics(intrinsics)

            files = {
                'left_ir':    ('left_ir.png',     ir_left_bytes,     'image/png'),
                'right_ir':   ('right_ir.png',    ir_right_bytes,    'image/png'),
                'intrinsics': ('intrinsics.yaml', intrinsics_bytes, 'application/octet-stream'),
            }
            if rgb is not None:
                files['rgb'] = ('rgb.png', _encode_color(rgb), 'image/png')
            if raw_depth is not None:
                files['raw_depth'] = ('depth.png', _encode_depth(raw_depth), 'image/png')

            t1 = time.time()

            resp = self.session.post(self.fs_api_url, files=files, timeout=60)
            resp.raise_for_status()

            t2 = time.time()

            depth_arr = cv2.imdecode(
                np.frombuffer(resp.content, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED
            )
            if depth_arr is None:
                return {'success': False, 'error': 'Failed to decode depth PNG from response'}

            result = {
                'success': True,
                'depth': depth_arr,
                'mode': 'foundation_stereo',
                'elapsed_ms': (t2 - t1) * 1000,
            }

            if get_vis:
                resp_vis = self.session.post(self.fs_vis_url, files=files, timeout=60)
                resp_vis.raise_for_status()
                vis_arr = cv2.imdecode(
                    np.frombuffer(resp_vis.content, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )
                if vis_arr is not None:
                    result['vis_image'] = vis_arr

            t3 = time.time()
            if _timing is not None:
                _timing['fs_encode'] = (t1 - t0) * 1000
                _timing['fs_http'] = (t2 - t1) * 1000
                _timing['fs_total'] = (t3 - t0) * 1000

            return result

        except requests.exceptions.RequestException as e:
            # DEBUG: capture response body for 500 errors
            _resp_body = ''
            if hasattr(e, 'response') and e.response is not None:
                try:
                    _resp_body = e.response.text[:500]
                except Exception:
                    pass
            print(f'[DepthOptimizerOnline/FS] Request failed: {e}')
            if _resp_body:
                print(f'[DepthOptimizerOnline/FS] Response body: {_resp_body}')
            return {'success': False, 'error': str(e)}
        except Exception as e:
            print(f'[DepthOptimizerOnline/FS] Error: {e}')
            return {'success': False, 'error': str(e)}

    def optimize_stereo_depth(self, ir_left, ir_right, intrinsics, **kwargs) -> Optional[np.ndarray]:
        """便捷接口：仅返回立体匹配深度图，失败返回 None"""
        result = self.forward_stereo(ir_left, ir_right, intrinsics, **kwargs)
        if result.get('success') and 'depth' in result:
            return result['depth']
        return None

    def get_stereo_visualization(self, ir_left, ir_right, intrinsics, **kwargs) -> Optional[np.ndarray]:
        """便捷接口：仅返回可视化 BGR 图，失败返回 None"""
        result = self.forward_stereo(ir_left, ir_right, intrinsics, get_vis=True, **kwargs)
        if result.get('success') and 'vis_image' in result:
            return result['vis_image']
        return None


class SAM3Online:
    """SAM3 分割检测服务客户端

    基于 SAM3 (Segment Anything Model 3) 的图像分割服务。
    通过 TensorRT 加速，支持文本提示的图像分割。
    提供与 DinoXDetectorOnline 兼容的接口。
    """
    def __init__(self, cfg):
        """初始化 SAM3 服务

        Args:
            cfg: 配置对象，包含：
                - url: 服务地址，默认 http://192.168.112.14:8081
                - confidence: 置信度阈值，默认 0.30
                - return_mask: 是否返回 mask，默认 True
                - tiled: 是否使用平铺模式，默认 False
                - jpeg_quality: JPEG 压缩质量，默认 85
                - resize: 图片缩放尺寸 (width, height)，默认 None
                - warmup: 预热次数，默认 0
        """
        self.base_url = getattr(cfg, 'url', 'http://192.168.112.14:8081')
        self.api_url = f'{self.base_url}/api/predict'
        self.health_url = f'{self.base_url}/api/health'
        self.confidence = getattr(cfg, 'confidence', 0.30)
        # 兼容 DinoX 参数名
        if hasattr(cfg, 'min_score'):
            self.confidence = cfg.min_score
        self.return_mask = getattr(cfg, 'return_mask', True)
        self.tiled = getattr(cfg, 'tiled', False)
        self.jpeg_quality = getattr(cfg, 'jpeg_quality', 85)
        self.resize = getattr(cfg, 'resize', None)

        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies = {'http': None, 'https': None}

        warmup_runs = getattr(cfg, 'warmup', 0)
        if warmup_runs > 0:
            self._warmup(warmup_runs)

    def _warmup(self, num_runs: int):
        """预热服务"""
        print(f"[SAM3Online] Warmup ({num_runs} runs)...")
        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in range(num_runs):
            self.forward('object', dummy_img)
            print(f"  warmup {i+1}/{num_runs}")
        print("[SAM3Online] Warmup complete")

    def check_health(self) -> Dict:
        """检查服务健康状态"""
        try:
            response = self.session.get(self.health_url, timeout=5)
            return response.json()
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def forward(self, text: str, rgb: np.ndarray, _timing: Optional[Dict] = None, **kwargs) -> LocalTaskResult:
        """根据文本描述进行图像分割检测

        提供与 DinoXDetectorOnline 兼容的接口。

        Args:
            text: 目标描述文本，多个类别用逗号或句点分隔
            rgb: RGB/BGR 图像 (numpy 数组)
            _timing: 可选，用于记录内部耗时
            **kwargs: 额外参数 (confidence, return_mask, tiled, min_score)

        Returns:
            LocalTaskResult: 包含检测结果
                {'objects': [{'bbox': [x1, y1, x2, y2], 'score': float, 'category': str, 'mask': dict}, ...]}
        """
        t0 = time.time()

        # 获取参数（兼容 DinoX 参数名）
        confidence = kwargs.get('confidence', kwargs.get('min_score', self.confidence))
        return_mask = kwargs.get('return_mask', self.return_mask)
        tiled = kwargs.get('tiled', self.tiled)

        # 记录原始尺寸
        orig_h, orig_w = rgb.shape[:2]
        scale_x, scale_y = 1.0, 1.0

        # Resize
        if self.resize is not None:
            target_w, target_h = self.resize
            rgb_resized = cv2.resize(rgb, (target_w, target_h))
            scale_x = orig_w / target_w
            scale_y = orig_h / target_h
        else:
            rgb_resized = rgb

        t1 = time.time()

        # JPEG 编码
        _, img_encoded = cv2.imencode('.jpg', rgb_resized, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        img_bytes = img_encoded.tobytes()

        t2 = time.time()

        # 发送请求
        files = {'images': ('image.jpg', img_bytes, 'image/jpeg')}
        data = {
            'text_prompt': text,
            'confidence': str(confidence),
            'return_mask': 'true' if return_mask else 'false',
            'tiled': 'true' if tiled else 'false',
        }

        try:
            response = self.session.post(self.api_url, files=files, data=data, timeout=30)
            t3 = time.time()

            if _timing is not None:
                _timing['sam3_preprocess'] = (t1 - t0) * 1000
                _timing['sam3_encode'] = (t2 - t1) * 1000
                _timing['sam3_http'] = (t3 - t2) * 1000

            response.raise_for_status()
            result_json = response.json()

            # 检查服务端错误
            if not result_json.get('success'):
                error_msg = result_json.get('error', 'Unknown error')
                print(f"[SAM3Online] Server error: {error_msg}")
                return LocalTaskResult({'objects': [], 'error': error_msg})

            # 解析返回结果，转换为与 DinoXDetectorOnline 兼容的格式
            objects = []
            if 'results' in result_json and len(result_json['results']) > 0:
                result = result_json['results'][0]
                for obj in result.get('objects', []):
                    # 获取 bbox 并映射回原始尺寸
                    bbox = obj.get('bbox')
                    if bbox and (scale_x != 1.0 or scale_y != 1.0):
                        bbox = [
                            bbox[0] * scale_x, bbox[1] * scale_y,
                            bbox[2] * scale_x, bbox[3] * scale_y
                        ]

                    obj_data = {
                        'bbox': bbox,
                        'score': obj.get('score'),
                        'category': obj.get('class_name', 'object'),
                    }
                    # 如果有 mask 信息也保留
                    if 'mask' in obj:
                        obj_data['mask'] = obj['mask']
                    objects.append(obj_data)

            return LocalTaskResult({'objects': objects})

        except requests.exceptions.RequestException as e:
            print(f"[SAM3Online] Request failed: {e}")
            return LocalTaskResult({'objects': [], 'error': str(e)})
        except json.JSONDecodeError as e:
            print(f"[SAM3Online] JSON decode failed: {e}")
            return LocalTaskResult({'objects': [], 'error': str(e)})
