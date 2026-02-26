import base64
import copy
import json
import math
import os
import sys
import time
from io import BytesIO

import cv2
import numpy as np
import requests
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk import Config as DDSConfig
from dds_cloudapi_sdk.tasks.v2_task import V2Task, create_task_with_local_image_auto_resize
from mmengine.config import Config
from PIL import Image
from pycocotools import mask as coco_mask


def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list | tuple):
            result.extend(flatten(item))  # 递归展开
        else:
            result.append(item)
    return result


class DinoXDetectorCloud:
    """在线语义参考定位服务类
    
    基于DINO-XSeek模型的语义理解服务，将文本/语音描述转换为目标定位。
    支持中英文自然语言输入，返回目标的边界框坐标。
    """
    def __init__(self, cfg):
        """初始化语义参考定位服务
        
        Args:
            cfg: 配置对象，包含：
                - uri: API路径
                - status_uri: 状态查询路径
                - token: API访问令牌
        """
        self.uri = cfg.uri  # API端点路径
        self.status_uri = cfg.status_uri  # 任务状态查询端点
        self.token = cfg.token  # 访问令牌
        # 初始化DDS云端API客户端
        config = DDSConfig(self.token)
        self.client = Client(config)

    def forward(self, text, rgb, **kwargs):
        """前向推理函数，根据文本描述定位目标
        
        Args:
            text: 目标描述文本，支持中英文
            rgb: RGB图像
            **kwargs: 额外参数
            
        Returns:
            V2Task: 任务对象，包含检测结果
        """
        # 构造API请求体
        body = {
            #'model': 'DINO-XSeek-1.0',  # 使用的模型
            'model': 'DINO-X-1.0',  # 使用的模型
            'image': None,  # 图像数据（将被Base64编码填入）
            'prompt': {'type': 'text', 'text': None},  # 文本提示
            'targets': ['bbox']  # 返回目标边界框
        }
        
        # 图像编码处理
        pil_img = Image.fromarray(rgb)

        # 创建BytesIO对象并保存图像数据
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        buffer.seek(0)  # 将指针移回起始位置

        # Base64编码
        img_bytes = buffer.getvalue()

        b64 = base64.b64encode(img_bytes).decode('utf-8')
        b64 = f'data:image/jpg;base64,{b64}'
        body['image'] = b64
        body['prompt']['text'] = text
        task = V2Task(api_path=self.uri, api_body=body)
        self.client.run_task(task)
        # print(task.result)  # {'objects': [{'bbox': [507.8738098144531, 602.8898315429688, 634.7514038085938, 817.6101684570312]}]}
        return task


class LocalTaskResult:
    """模拟 V2Task 的结果对象，用于本地服务返回"""
    def __init__(self, result_dict):
        self.result = result_dict


class DinoXDetectorOnline:
    """本地语义参考定位服务类

    基于本地 DINO-X 服务的语义理解，将文本描述转换为目标定位。
    通过 HTTP API 访问 http://192.168.112.14:10086/ 获取结果。
    提供与 DinoXDetectorCloud 相同的接口。
    """
    def __init__(self, cfg):
        """初始化本地语义参考定位服务

        Args:
            cfg: 配置对象，包含：
                - url: 服务地址，默认为 http://192.168.112.14:10086
                - min_score: 最小检测分数阈值，默认 0.25
                - iou_threshold: IoU 阈值，默认 0.5
                - jpeg_quality: JPEG压缩质量，默认 50（降低可减少网络传输时间）
                - resize: 图片缩放尺寸 (width, height)，默认 (1280, 720)，设为 None 禁用
                - warmup: 预热次数，默认 0（不预热）
        """
        self.base_url = getattr(cfg, 'url', 'http://192.168.112.14:10086')
        self.api_url = f"{self.base_url}/api/predict"
        self.min_score = getattr(cfg, 'min_score', 0.25)
        self.iou_threshold = getattr(cfg, 'iou_threshold', 0.5)
        self.jpeg_quality = getattr(cfg, 'jpeg_quality', 50)
        self.resize = getattr(cfg, 'resize', (1280, 720))  # (width, height)

        # 创建不使用代理的 Session，用于内网服务器访问
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies = {'http': None, 'https': None}

        # Warmup
        warmup_runs = getattr(cfg, 'warmup', 0)
        if warmup_runs > 0:
            self._warmup(warmup_runs)

    def _warmup(self, num_runs):
        """预热，排除冷启动影响"""
        print(f"[DinoXDetectorOnline] Warmup ({num_runs} 次)...")
        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        dummy_text = 'object'
        for i in range(num_runs):
            self.forward(dummy_text, dummy_img)
            print(f"  warmup {i+1}/{num_runs}")
        print("[DinoXDetectorOnline] Warmup 完成")

    def forward(self, text, rgb, _timing=None, **kwargs):
        """前向推理函数，根据文本描述定位目标

        Args:
            text: 目标描述文本，多个类别用 '.' 分隔
            rgb: RGB 图像 (numpy 数组)
            _timing: 可选的 dict，用于记录内部耗时分布
            **kwargs: 额外参数
                - min_score: 覆盖默认的最小分数阈值
                - iou_threshold: 覆盖默认的 IoU 阈值

        Returns:
            LocalTaskResult: 包含 result 属性的对象，格式为:
                {'objects': [{'bbox': [x1, y1, x2, y2], 'score': float, 'category': str, 'mask': dict}, ...]}
        """
        import time as _time
        _t0 = _time.time()

        # 获取参数
        min_score = kwargs.get('min_score', self.min_score)
        iou_threshold = kwargs.get('iou_threshold', self.iou_threshold)

        # 记录原始尺寸，用于坐标映射
        orig_h, orig_w = rgb.shape[:2]
        scale_x, scale_y = 1.0, 1.0

        # Resize 图像（减少传输时间）
        if self.resize is not None:
            target_w, target_h = self.resize
            rgb_resized = cv2.resize(rgb, (target_w, target_h))
            scale_x = orig_w / target_w
            scale_y = orig_h / target_h
        else:
            rgb_resized = rgb

        _t1 = _time.time()  # 预处理完成

        # 将图像编码为 JPEG 格式（使用配置的压缩质量）
        _, img_encoded = cv2.imencode('.jpg', rgb_resized, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        img_bytes = img_encoded.tobytes()

        _t2 = _time.time()  # JPEG编码完成

        # 准备 multipart form data
        files = {
            'images': ('image.jpg', img_bytes, 'image/jpeg')
        }
        data = {
            'text_prompt': text,
            'min_score': min_score,
            'iou_threshold': iou_threshold,
            'chosen_policy': 'det'  # 只需要检测结果，不需要可视化
        }

        # 解析 text prompt 中的类别列表，用于将索引转换为名称
        categories = [c.strip() for c in text.split('.')]

        try:
            response = self.session.post(self.api_url, files=files, data=data, timeout=30)
            _t3 = _time.time()  # HTTP请求完成

            # 记录时间分布
            if _timing is not None:
                _timing['dinox_preprocess'] = (_t1 - _t0) * 1000
                _timing['dinox_encode'] = (_t2 - _t1) * 1000
                _timing['dinox_http'] = (_t3 - _t2) * 1000

            response.raise_for_status()
            result_json = response.json()

            # 解析返回结果，转换为与 DinoXDetectorCloud 兼容的格式
            objects = []
            if 'results' in result_json and len(result_json['results']) > 0:
                result = result_json['results'][0]
                for obj in result.get('objects', []):
                    # 将 category 索引转换为类别名称
                    cat_idx = obj.get('category')
                    if isinstance(cat_idx, int) and 0 <= cat_idx < len(categories):
                        category_name = categories[cat_idx]
                    else:
                        category_name = str(cat_idx)  # 保留原值

                    # 将 bbox 坐标映射回原始尺寸
                    bbox = obj.get('bbox')
                    if bbox and (scale_x != 1.0 or scale_y != 1.0):
                        bbox = [
                            bbox[0] * scale_x,
                            bbox[1] * scale_y,
                            bbox[2] * scale_x,
                            bbox[3] * scale_y
                        ]

                    obj_data = {
                        'bbox': bbox,
                        'score': obj.get('score'),
                        'category': category_name,
                    }
                    # 如果有 mask 信息也保留
                    if 'mask' in obj:
                        obj_data['mask'] = obj['mask']
                    objects.append(obj_data)

            return LocalTaskResult({'objects': objects})

        except requests.exceptions.RequestException as e:
            print(f"[DinoXDetectorOnline] 请求失败: {e}")
            return LocalTaskResult({'objects': [], 'error': str(e)})
        except json.JSONDecodeError as e:
            print(f"[DinoXDetectorOnline] JSON 解析失败: {e}")
            return LocalTaskResult({'objects': [], 'error': str(e)})


class GraspAnythingOnline:
    """在线抓取检测服务类
    
    调用云端深度学习模型进行目标棄测和抓取点生成。
    支持多目标检测、实例分割、抓取姿态估计等功能。
    """
    def __init__(self, cfg):
        """初始化在线检测服务

        Args:
            cfg: 配置对象，包含：
                - server_list: 服务器列表文件路径
                - model_name: 使用的模型名称
                - resize: 图片缩放尺寸 (width, height)，默认 (1280, 720)，设为 None 禁用
                - warmup: 预热次数，默认 0（不预热）
        """
        self.resize = getattr(cfg, 'resize', (1280, 720))  # (width, height)
        self.server_list_fp = cfg.server_list  # 服务器配置文件路径

        # 检查配置文件存在性
        if not os.path.isfile(self.server_list_fp):
            if self.server_list_fp.startswith('/'):
                tmp = './' + os.path.basename(self.server_list_fp)
                if not os.path.isfile(tmp):
                    raise Exception(f'file not found: {self.server_list_fp} or {tmp}. {cfg=}')
                else:
                    print(f'load cfg from  {tmp}')
                    self.server_list_fp = tmp

        # 加载服务器列表配置
        with open(self.server_list_fp) as f:
            api_dict = json.load(f).get('backends', {})
            self.server_list = api_dict

        self.model_name = cfg.model_name
        assert self.model_name in self.server_list, f'{self.model_name} not in {self.server_list}. {cfg=}'
        self.url = self.server_list[self.model_name] + '/generate'  # API端点

        # 创建不使用代理的 Session，用于内网服务器访问
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies = {'http': None, 'https': None}  # 彻底禁用代理

        # Warmup
        warmup_runs = getattr(cfg, 'warmup', 0)
        if warmup_runs > 0:
            self._warmup(warmup_runs)

    def _warmup(self, num_runs):
        """预热，排除冷启动影响"""
        print(f"[GraspAnythingOnline] Warmup ({num_runs} 次)...")
        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in range(num_runs):
            self.forward(dummy_img)
            print(f"  warmup {i+1}/{num_runs}")
        print("[GraspAnythingOnline] Warmup 完成")

    def forward(self, rgb, depth=None, _timing=None, **kwargs):
        """前向推理函数，调用云端检测服务

        执行步骤：
        1. 图像预处理（padding、压缩）
        2. 发送HTTP请求到检测服务器
        3. 解析返回结果（边界框、掉码、抓取点）
        4. 后处理：计算接触点、优化抓取姿态

        Args:
            rgb: RGB图像，支持numpy数组、文件路径或bytes
            depth: 深度图（可选）
            _timing: 可选的 dict，用于记录内部耗时分布
            **kwargs: 额外参数，如bag模式、接触点使用等

        Returns:
            tuple: (检测结果, 填充后的图像)
        """
        import time as _time
        _t0 = _time.time()

        # 图像类型处理
        if isinstance(rgb, str):
            img = np.array(Image.open(rgb).convert('RGB'))[:, :, ::-1]  # RGB->BGR
        elif isinstance(rgb, np.ndarray):
            img = rgb  # numpy数组
        elif isinstance(rgb, bytes):
            # 从bytes数据解码
            img = cv2.imdecode(
                np.frombuffer(rgb, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
        else:
            raise NotImplementedError(f'{type(rgb)}')

        # 记录原始尺寸
        orig_h, orig_w = img.shape[:2]

        # Resize 图像（减少传输时间）
        if self.resize is not None:
            target_w, target_h = self.resize
            img = cv2.resize(img, (target_w, target_h))

        h, w = img.shape[:2]
        max_dim = max(h, w)
        # 计算上下和左右的padding量
        top = 0
        bottom = max_dim - h - top
        left = 0
        right = max_dim - w - left

        padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)

        _t1 = _time.time()  # 预处理完成

        _, img_encoded = cv2.imencode('.jpg', padded_img, [cv2.IMWRITE_JPEG_QUALITY, 30])
        img_bytes = img_encoded.tobytes()

        _t2 = _time.time()  # JPEG编码完成

        # 使用不受代理环境变量影响的Session访问内网服务器
        response = self.session.post(self.url, files={'img_source': img_bytes})

        _t3 = _time.time()  # HTTP请求完成

        # 记录时间分布
        if _timing is not None:
            _timing['grasp_preprocess'] = (_t1 - _t0) * 1000
            _timing['grasp_encode'] = (_t2 - _t1) * 1000
            _timing['grasp_http'] = (_t3 - _t2) * 1000

        _t4 = _time.time()
        objs = response.json()  # [bs, obj]
        _t5 = _time.time()

        bs = len(objs)
        assert bs == 1, f'{bs=}'
        obj_num = len(objs[0])
        if obj_num == 0:
            print("[GraspAPI] No objects detected!")
            return objs, padded_img
        obj0 = objs[0][0]

        _t6 = _time.time()
        if 'dt_mask' in obj0 and obj0['dt_mask'] is not None:
            # 优化1: 批量解码所有 RLE masks
            rle_list = [obj['dt_mask'] for obj in objs[0]]
            for obj in objs[0]:
                obj['dt_rle'] = obj['dt_mask']  # 保存原始 RLE

            # 批量解码 (pycocotools 支持批量解码)
            masks_decoded = coco_mask.decode(rle_list)  # (H, W, N)

            kernel = np.ones((5, 5), np.uint8)
            size = (max_dim, max_dim)

            for i, obj in enumerate(objs[0]):
                m = masks_decoded[:, :, i] if masks_decoded.ndim == 3 else masks_decoded
                m = cv2.resize(m, dsize=size)
                m = cv2.dilate(m, kernel, iterations=1)
                m = m[:h, :w].astype(bool)
                obj['dt_mask'] = m
        _t7 = _time.time()

        ret = objs
        if kwargs.get('bag', False):
            ret = self.extract_bag(ret, rgb, depth, **kwargs)
        if kwargs.get('use_touching_points', True):
            ret = self.post_process(ret, rgb, depth, **kwargs)
        _t8 = _time.time()

        # 记录后处理时间
        if _timing is not None:
            _timing['grasp_json'] = (_t5 - _t4) * 1000
            _timing['grasp_mask_decode'] = (_t7 - _t6) * 1000
            _timing['grasp_postproc'] = (_t8 - _t7) * 1000

        return ret, padded_img

    def post_process(self, ret, rgb, depth, **kwargs):
        """后处理函数，计算抓取接触点 (优化版)

        使用 Bresenham 算法直接获取线段点，避免创建大数组。
        """
        for obj in ret[0]:
            if 'dt_mask' not in obj:
                continue
            m = obj['dt_mask']
            affs = obj['affs']
            touching_points = []
            mask_h, mask_w = m.shape

            for rb in affs:
                xc, yc, w, h, angle2 = rb

                # 计算旋转矩形的4个顶点
                p1, p2, p3, p4 = cv2.boxPoints(((xc, yc), (w, h), angle2 * 180 / math.pi))
                c1 = [int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)]
                c2 = [int((p4[0] + p3[0]) / 2), int((p4[1] + p3[1]) / 2)]

                # 优化: 使用 Bresenham 算法直接获取线段上的点
                line_pts = self._bresenham_line(c1[0], c1[1], c2[0], c2[1])

                # 过滤有效点并检查 mask
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
        """Bresenham 线段算法 - 高效获取线段上所有整数点"""
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

    def extract_bag(self, ret, rgb, depth, **kwargs):
        ret0 = ret[0]
        img_h, img_w = rgb.shape[:2]  # (720, 1280)
        for obj in ret0:
            mask = obj['dt_mask']
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return
            largest_contour = max(contours, key=cv2.contourArea)
            guess_mask = np.zeros_like(mask)
            guess_mask = np.ascontiguousarray(guess_mask)  # this is must
            guess = cv2.minAreaRect(largest_contour)
            (cx, cy), (w, h), angle = guess
            if w < h:
                w, h = h, w  # 交换宽高
                angle += 90  # 角度调整
                angle = angle % 180  # in [0, 180] and w >=h which is the same as affordance

            guess = [(cx, cy), (w, h), angle]

            p1, p2, p3, p4 = cv2.boxPoints(guess).astype(int)  # 4 points in (x,y)
            # breakpoint()
            guess = flatten(guess)  # xc, yc, w, h, angle(degree)
            guess[4] = guess[4] / 180 * math.pi
            # breakpoint()
            # cv2.drawContours(guess_mask, [p4], -1, 255, -1)

            c1 = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
            c2 = [(p4[0] + p3[0]) / 2, (p4[1] + p3[1]) / 2]
            c1 = [int(x) for x in c1]
            c2 = [int(x) for x in c2]
            c3 = [0.2 * c2[0] + 0.8 * c1[0], 0.2 * c2[1] + 0.8 * c1[1]]  # near c1
            c4 = [0.2 * c1[0] + 0.8 * c2[0], 0.2 * c1[1] + 0.8 * c2[1]]  # near c2
            c3 = [int(x) for x in c3]
            c4 = [int(x) for x in c4]
            if c3[0] >= img_w or c3[1] >= img_h or c4[0] >= img_w or c4[1] >= img_h:
                aff = None
            else:
                if mask[c3[1], c3[0]] and (not mask[c4[1], c4[0]]):  # near c2 part is background
                    aff = [*c2, max(abs(c4[0] - c2[0]), abs(c4[1] - c2[1])), 30, guess[4]]
                elif (not mask[c3[1], c3[0]]) and mask[c4[1], c4[0]]:  # near c1 part is background
                    aff = [*c1, max(abs(c3[0] - c1[0]), abs(c3[1] - c1[1])), 30, guess[4]]
                else:
                    aff = None
                    print('can determine')
            # breakpoint()
            # dic = copy.deepcopy(obj[0])
            # dic['affs'] = aff
            # dic['scores'] = 1.0
            # breakpoint()
            if aff is not None:
                obj['scores'][-1] = 1.0
                obj['affs'][-1] = aff

                # breakpoint()
        return ret

    def vis(self, objs, rgb, padded_img, img_fp, to_save=True):
        import supervision as sv

        objs = objs[0]
        # breakpoint()
        dt_mask = [obj['dt_mask'] for obj in objs]
        dt_bbox = [obj['dt_bbox'] for obj in objs]

        N = len(dt_mask)

        lines = np.array([obj['touching_points'] for obj in objs])  # (obj#, aff#, 2, 2)
        lines = lines.reshape(-1, *lines.shape[2:])
        # breakpoint()
        rgb = cv2.polylines(rgb, lines, isClosed=False, color=(0, 0, 255), thickness=2)
        vis_img = np.asfortranarray(rgb)
        masks = np.array(dt_mask).astype(bool)  # must be bool!!!!
        bboxes = np.array(dt_bbox)
        # breakpoint()
        detections = sv.Detections(
            xyxy=bboxes,
            mask=masks,
            class_id=np.array([1] * N),
        )
        labels = ['o'] * N

        pil = Image.fromarray(vis_img)
        annotated_frame = pil
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(annotated_frame, detections=detections)
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections)
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(annotated_frame, detections=detections)
        if to_save:
            fp_out = img_fp.replace('.png', '.jpg').replace('.jpg', '_vis.jpg')
            annotated_frame.save(fp_out)
            print(f'save vis to : {fp_out}')
        return np.array(annotated_frame)

    def preprocess(self, image):
        # Implement preprocessing steps like resizing, normalization, etc.
        pass  # Placeholder for actual preprocessing logic

    def to_coco(self, dts, height, width):
        imgs = []
        anns = []
        cats = [{'id': 0, 'name': 'object'}]
        for frame_id, dt in enumerate(dts):
            img = dict(id=frame_id, width=width, height=height, file_name=frame_id)
            imgs.append(img)
            objs = dt[0]
            for obj in objs:
                obj['image_id'] = frame_id
                obj['category_id'] = 0
                if 'dt_mask' in obj and obj['dt_mask'] is not None:
                    mask = obj['dt_mask']
                    mask = np.asfortranarray(mask)
                    rle = coco_mask.encode(np.asfortranarray(mask))
                    rle['counts'] = rle['counts'].decode('utf-8')
                    obj['dt_mask'] = rle
            anns.extend(objs)

        coco = dict(images=imgs, annotations=anns, categories=cats)
        return coco

    def demo_video(self, fp, fp_out, fps=None, func=None, **kwargs):
        if not os.path.isfile(fp):
            print(f'file not found: {fp}')
            return
        cap = cv2.VideoCapture(fp)
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened():
            print('fail to open: {fp}')
            sys.exit(1)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
        out = None
        frame_id = 0
        dts = []
        while cap.isOpened():
            if frame_id % kwargs.get('log_interval', 10) == 0:
                print(f'process on frame {frame_id}')
            if kwargs.get('debug', False) and frame_id > kwargs.get('max_frame', 50):
                break
            frame_id += 1

            ret, fra = cap.read()
            if not ret:
                break

            if func is not None:
                fra = func(fra)
            det, padded_img = self.forward(rgb=fra, depth=None)
            dts.append(det)
            # breakpoint()
            img_vis = self.vis(objs=det, rgb=fra, padded_img=padded_img, img_fp=None, to_save=False)
            if out is None:
                height, width, _ = img_vis.shape
                out = cv2.VideoWriter(fp_out, fourcc, fps, (width, height), True)
            out.write(img_vis)
        cap.release()
        out.release()
        print(f'read {fp} and write to {fp_out}: {frame_id} frames')

        coco = self.to_coco(dts=dts, width=width, height=height)
        fp_out2 = fp_out.replace('.mp4', '.json')
        fo = open(fp_out2, 'w')
        json.dump(coco, fo, indent=2)
        fo.close()
        print(f'write coco json to {fp_out2}')
        return coco


class SAM2TrackerOnline:
    """在线目标跟踪服务类

    基于 SAM2 的目标跟踪服务，支持多目标跟踪和实例分割。
    调用 /api/predict 接口进行逐帧跟踪。

    接口说明：
    - 输入：图像 + 检测结果（首帧必须提供检测结果用于初始化）
    - 输出：跟踪结果，包含 ids, boxes, masks 等
    - frame_idx=0 时 tracker 会自动 reset
    """
    def __init__(self, cfg):
        """初始化在线跟踪服务

        Args:
            cfg: 配置对象，包含：
                - url: 服务地址，默认为 http://192.168.112.14:11086
                - resize: 图像缩放尺寸 (width, height)，默认 (512, 512)，tracker 固定使用此尺寸
                - jpeg_quality: JPEG压缩质量，默认 30
                - warmup: 预热次数，默认 0
        """
        self.base_url = getattr(cfg, 'url', 'http://192.168.112.14:11086')
        self.api_url = f'{self.base_url}/api/predict'
        self.resize = getattr(cfg, 'resize', (512, 512))  # tracker 固定使用 512x512
        self.jpeg_quality = getattr(cfg, 'jpeg_quality', 30)

        # 创建不使用代理的 Session，用于内网服务器访问
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies = {'http': None, 'https': None}

        # 内部状态：记录缩放比例，用于坐标转换
        self._scale_x = 1.0
        self._scale_y = 1.0
        self._orig_size = None

        # Warmup
        warmup_runs = getattr(cfg, 'warmup', 0)
        if warmup_runs > 0:
            self._warmup(warmup_runs)

    def _warmup(self, num_runs):
        """预热，排除冷启动影响"""
        print(f"[SAM2TrackerOnline] Warmup ({num_runs} 次)...")
        dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
        dummy_dets = [{'dt_bbox': [100, 100, 200, 200], 'dt_score': 0.9, 'name': 'object'}]
        for i in range(num_runs):
            self.forward(dummy_img, dets=dummy_dets, frame_idx=0)
            print(f"  warmup {i+1}/{num_runs}")
        print("[SAM2TrackerOnline] Warmup 完成")

    def forward(self, rgb, dets=None, frame_idx=0, **kwargs):
        """逐帧跟踪接口

        Args:
            rgb: RGB/BGR图像，支持:
                - numpy数组 (H, W, 3)
                - 文件路径字符串
                - bytes数据
            dets: 检测结果列表，每个元素包含:
                - dt_bbox: [x1, y1, x2, y2] xyxy 格式像素坐标
                - dt_score: 检测分数
                - name: 类别名称
                首帧(frame_idx=0)必须提供检测结果用于初始化 tracker
            frame_idx: 帧索引，0 表示首帧（会触发 tracker reset）
            **kwargs: 额外参数（预留）

        Returns:
            dict: 跟踪结果，包含：
                - success: 是否成功
                - ids: 跟踪ID列表
                - boxes: 边界框列表 [[x1,y1,x2,y2], ...] 原始图像坐标
                - scores: 检测分数列表
                - track_scores: 跟踪分数列表
                - cats: 类别列表
                - masks: RLE格式的mask列表（512x512尺寸）
                - error: 错误信息（如果失败）
        """
        # 处理图像输入
        if isinstance(rgb, str):
            img = cv2.imread(rgb)
            if img is None:
                return {'success': False, 'error': f'无法读取图像: {rgb}'}
        elif isinstance(rgb, np.ndarray):
            img = rgb
        elif isinstance(rgb, bytes):
            img = cv2.imdecode(np.frombuffer(rgb, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return {'success': False, 'error': '无法解码图像数据'}
        else:
            return {'success': False, 'error': f'不支持的图像类型: {type(rgb)}'}

        # 记录原始尺寸
        orig_h, orig_w = img.shape[:2]
        self._orig_size = (orig_w, orig_h)

        # Resize 到 512x512（tracker 固定尺寸）
        target_w, target_h = self.resize
        self._scale_x = orig_w / target_w
        self._scale_y = orig_h / target_h

        img_resized = cv2.resize(img, (target_w, target_h))

        # 缩放检测框坐标到 512x512
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

        # 编码图像
        _, img_encoded = cv2.imencode('.jpg', img_resized,
                                       [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        img_bytes = img_encoded.tobytes()

        # 准备请求
        files = {'img_source': ('frame.jpg', img_bytes, 'image/jpeg')}
        data = {'frame_idx': str(frame_idx)}
        if dets_scaled is not None:
            data['dets'] = json.dumps(dets_scaled)

        try:
            response = self.session.post(self.api_url, files=files, data=data, timeout=30)
            response.raise_for_status()
            result = response.json()

            # 检查错误
            if 'error' in result:
                return {'success': False, 'error': result['error']}

            # 空结果
            if not result or len(result) == 0:
                return {'success': True, 'ids': [], 'boxes': [], 'scores': [],
                        'track_scores': [], 'cats': [], 'masks': []}

            # 将 boxes 坐标缩放回原始图像尺寸
            if 'boxes' in result and len(result['boxes']) > 0:
                scaled_boxes = []
                for box in result['boxes']:
                    scaled_boxes.append([
                        box[0] * self._scale_x,
                        box[1] * self._scale_y,
                        box[2] * self._scale_x,
                        box[3] * self._scale_y,
                    ])
                result['boxes'] = scaled_boxes

            result['success'] = True
            return result

        except requests.exceptions.RequestException as e:
            print(f"[SAM2TrackerOnline] 请求失败: {e}")
            return {'success': False, 'error': str(e)}
        except json.JSONDecodeError as e:
            print(f"[SAM2TrackerOnline] JSON 解析失败: {e}")
            return {'success': False, 'error': str(e)}

    def track_video(self, video_path, detector=None, text_prompt='object',
                    output_path=None, **kwargs):
        """跟踪整个视频

        Args:
            video_path: 输入视频路径
            detector: 检测器实例（如 DinoXDetectorOnline），用于首帧检测
            text_prompt: 检测文本提示
            output_path: 输出视频路径，默认在原文件名后加 _tracked
            **kwargs: 额外参数
                - det_interval: 检测间隔帧数，默认 -1（仅首帧检测）
                - start: 起始帧，默认 0
                - limit: 帧数限制，默认 -1（全部）
                - select_center: 是否选择居中目标，默认 False

        Returns:
            dict: 跟踪结果统计
        """
        if not os.path.isfile(video_path):
            return {'success': False, 'error': f'视频文件不存在: {video_path}'}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'success': False, 'error': f'无法打开视频: {video_path}'}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 输出路径
        if output_path is None:
            base, ext = os.path.splitext(video_path)
            output_path = f'{base}_tracked{ext}'

        # 参数
        det_interval = kwargs.get('det_interval', -1)
        start = kwargs.get('start', 0)
        limit = kwargs.get('limit', -1)
        select_center = kwargs.get('select_center', False)

        # 视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        processed = 0
        tracked = 0
        all_tracks = {}

        print(f'[SAM2TrackerOnline] 开始跟踪视频: {video_path}')
        print(f'  总帧数: {total_frames}, FPS: {fps:.2f}, 尺寸: {width}x{height}')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < start:
                frame_idx += 1
                continue
            if limit > 0 and processed >= limit:
                break

            # 首帧或定期检测
            dets = None
            if frame_idx == start or (det_interval > 0 and (frame_idx - start) % det_interval == 0):
                if detector is not None:
                    task = detector.forward(text=text_prompt, rgb=frame)
                    if task.result and 'objects' in task.result and len(task.result['objects']) > 0:
                        objs = task.result['objects']

                        # 选择居中目标
                        if select_center and len(objs) > 1:
                            center_x = width / 2
                            center_y = height / 2
                            min_dist = float('inf')
                            selected = objs[0]
                            for obj in objs:
                                bbox = obj['bbox']
                                obj_cx = (bbox[0] + bbox[2]) / 2
                                obj_cy = (bbox[1] + bbox[3]) / 2
                                dist = (obj_cx - center_x) ** 2 + (obj_cy - center_y) ** 2
                                if dist < min_dist:
                                    min_dist = dist
                                    selected = obj
                            objs = [selected]

                        dets = []
                        for obj in objs:
                            dets.append({
                                'dt_bbox': obj['bbox'],
                                'dt_score': obj.get('score', 0.9),
                                'name': obj.get('category', 'object'),
                            })
                        print(f'  帧 {frame_idx}: 检测到 {len(dets)} 个目标')

            # 跟踪
            tracker_frame_idx = 0 if frame_idx == start else frame_idx - start
            result = self.forward(frame, dets=dets, frame_idx=tracker_frame_idx)

            # 可视化
            vis_frame = frame.copy()
            if result.get('success') and 'boxes' in result and len(result['boxes']) > 0:
                tracked += 1
                all_tracks[frame_idx] = result

                for i, (box, track_id) in enumerate(zip(result['boxes'], result.get('ids', []))):
                    x1, y1, x2, y2 = [int(b) for b in box]
                    color = ((37 * track_id) % 255, (17 * track_id + 100) % 255, (29 * track_id + 50) % 255)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    cat = result['cats'][i] if 'cats' in result and i < len(result['cats']) else ''
                    label = f"ID:{track_id} {cat}"
                    cv2.putText(vis_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 帧号
            cv2.putText(vis_frame, f'Frame: {frame_idx}', (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            writer.write(vis_frame)
            processed += 1
            frame_idx += 1

            if processed % 50 == 0:
                print(f'  进度: {processed}/{total_frames if limit < 0 else limit}')

        cap.release()
        writer.release()

        print(f'[SAM2TrackerOnline] 跟踪完成!')
        print(f'  处理帧数: {processed}, 跟踪帧数: {tracked}')
        print(f'  输出视频: {output_path}')

        return {
            'success': True,
            'processed': processed,
            'tracked': tracked,
            'output_path': output_path,
            'tracks': all_tracks,
        }


class SAM3Online:
    """在线 SAM3 分割检测服务类

    基于 SAM3 (Segment Anything Model 3) 的图像分割服务。
    通过 TensorRT 加速，支持文本提示和几何提示的图像分割。
    提供与 DinoXDetectorOnline 兼容的检测接口。
    """
    def __init__(self, cfg):
        """初始化 SAM3 服务

        Args:
            cfg: 配置对象，包含：
                - url: 服务地址，默认为 http://192.168.112.14:8080
                - confidence: 置信度阈值，默认 0.30
                - return_mask: 是否返回 mask，默认 True
                - tiled: 是否使用平铺模式，默认 False
                - jpeg_quality: JPEG 压缩质量，默认 85
                - resize: 图片缩放尺寸 (width, height)，默认 None（不缩放）
                - warmup: 预热次数，默认 0（不预热）
        """
        self.base_url = getattr(cfg, 'url', 'http://192.168.112.14:8080')
        self.api_url = f'{self.base_url}/api/predict'
        self.health_url = f'{self.base_url}/api/health'
        self.confidence = getattr(cfg, 'confidence', 0.30)
        self.return_mask = getattr(cfg, 'return_mask', True)
        self.tiled = getattr(cfg, 'tiled', False)
        self.jpeg_quality = getattr(cfg, 'jpeg_quality', 85)
        self.resize = getattr(cfg, 'resize', None)

        # 创建不使用代理的 Session，用于内网服务器访问
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies = {'http': None, 'https': None}

        # Warmup
        warmup_runs = getattr(cfg, 'warmup', 0)
        if warmup_runs > 0:
            self._warmup(warmup_runs)

    def _warmup(self, num_runs):
        """预热，排除冷启动影响"""
        print(f"[SAM3Online] Warmup ({num_runs} 次)...")
        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        dummy_text = 'object'
        for i in range(num_runs):
            self.forward(dummy_text, dummy_img)
            print(f"  warmup {i+1}/{num_runs}")
        print("[SAM3Online] Warmup 完成")

    def check_health(self):
        """检查服务健康状态

        Returns:
            dict: 健康状态信息，包含 status, engine_loaded, gpu_id 等
        """
        try:
            response = self.session.get(self.health_url, timeout=5)
            return response.json()
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def forward(self, text, rgb, _timing=None, **kwargs):
        """前向推理函数，根据文本描述进行图像分割检测

        提供与 DinoXDetectorOnline 兼容的接口。

        Args:
            text: 目标描述文本，多个类别用逗号或句点分隔
            rgb: RGB 图像 (numpy 数组)
            _timing: 可选的 dict，用于记录内部耗时分布
            **kwargs: 额外参数
                - confidence: 覆盖默认的置信度阈值
                - return_mask: 覆盖默认的 mask 返回设置
                - tiled: 覆盖默认的平铺模式设置
                - boxes: 可选的 box prompt (JSON 格式字符串)

        Returns:
            LocalTaskResult: 包含 result 属性的对象，格式为:
                {'objects': [{'bbox': [x1, y1, x2, y2], 'score': float, 'category': str}, ...]}
        """
        import time as _time
        _t0 = _time.time()

        # 获取参数
        confidence = kwargs.get('confidence', self.confidence)
        return_mask = kwargs.get('return_mask', self.return_mask)
        tiled = kwargs.get('tiled', self.tiled)
        boxes = kwargs.get('boxes', None)

        # 记录原始尺寸，用于坐标映射
        orig_h, orig_w = rgb.shape[:2]
        scale_x, scale_y = 1.0, 1.0

        # Resize 图像（如果配置了）
        if self.resize is not None:
            target_w, target_h = self.resize
            rgb_resized = cv2.resize(rgb, (target_w, target_h))
            scale_x = orig_w / target_w
            scale_y = orig_h / target_h
        else:
            rgb_resized = rgb

        _t1 = _time.time()  # 预处理完成

        # 将图像编码为 JPEG 格式
        _, img_encoded = cv2.imencode('.jpg', rgb_resized, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        img_bytes = img_encoded.tobytes()

        _t2 = _time.time()  # JPEG 编码完成

        # 准备 multipart form data
        files = {
            'images': ('image.jpg', img_bytes, 'image/jpeg')
        }
        data = {
            'text_prompt': text,
            'confidence': str(confidence),
            'return_mask': 'true' if return_mask else 'false',
            'tiled': 'true' if tiled else 'false',
        }
        if boxes:
            data['boxes'] = boxes

        try:
            response = self.session.post(self.api_url, files=files, data=data, timeout=30)
            _t3 = _time.time()  # HTTP 请求完成

            # 记录时间分布
            if _timing is not None:
                _timing['sam3_preprocess'] = (_t1 - _t0) * 1000
                _timing['sam3_encode'] = (_t2 - _t1) * 1000
                _timing['sam3_http'] = (_t3 - _t2) * 1000

            response.raise_for_status()
            result_json = response.json()

            # 检查服务端错误
            if not result_json.get('success'):
                error_msg = result_json.get('error', 'Unknown error')
                print(f"[SAM3Online] 服务端错误: {error_msg}")
                return LocalTaskResult({'objects': [], 'error': error_msg})

            # 解析返回结果，转换为与 DinoXDetectorOnline 兼容的格式
            objects = []
            if 'results' in result_json and len(result_json['results']) > 0:
                result = result_json['results'][0]  # 只处理第一张图片
                for obj in result.get('objects', []):
                    # 获取 bbox 并映射回原始尺寸
                    bbox = obj.get('bbox')
                    if bbox and (scale_x != 1.0 or scale_y != 1.0):
                        bbox = [
                            bbox[0] * scale_x,
                            bbox[1] * scale_y,
                            bbox[2] * scale_x,
                            bbox[3] * scale_y
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
            print(f"[SAM3Online] 请求失败: {e}")
            return LocalTaskResult({'objects': [], 'error': str(e)})
        except json.JSONDecodeError as e:
            print(f"[SAM3Online] JSON 解析失败: {e}")
            return LocalTaskResult({'objects': [], 'error': str(e)})


class DepthOptimizerOnline:
    """在线深度图去噪优化服务类

    基于 CDM (Depth Map Denoising with RGB-D Fusion) 的深度图优化服务。
    通过 RGB 图像引导对深度图进行去噪和修复。
    """
    def __init__(self, cfg):
        """初始化深度优化服务

        Args:
            cfg: 配置对象，包含：
                - url: 服务地址，默认为 http://192.168.112.14:8086
                - chosen_policy: 输出策略，可选值：
                    - 'dn,vis': 去噪+可视化（默认）
                    - 'dn': 仅去噪
                    - 'vis': 仅可视化
                - warmup: 预热次数，默认 0（不预热）
        """
        self.base_url = getattr(cfg, 'url', 'http://192.168.112.14:8086')
        self.api_url = f'{self.base_url}/api/predict'
        self.chosen_policy = getattr(cfg, 'chosen_policy', 'dn,vis')

        # 创建不使用代理的 Session，用于内网服务器访问
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies = {'http': None, 'https': None}

        # Warmup
        warmup_runs = getattr(cfg, 'warmup', 0)
        if warmup_runs > 0:
            self._warmup(warmup_runs)

    def _warmup(self, num_runs):
        """预热，排除冷启动影响"""
        print(f"[DepthOptimizerOnline] Warmup ({num_runs} 次)...")
        # 生成简单的测试图像
        dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_depth = np.zeros((480, 640), dtype=np.uint16)
        for i in range(num_runs):
            self.forward(dummy_rgb, dummy_depth)
            print(f"  warmup {i+1}/{num_runs}")
        print("[DepthOptimizerOnline] Warmup 完成")

    def forward(self, rgb, depth, chosen_policy=None, _timing=None, **kwargs):
        """前向推理函数，调用深度去噪服务

        Args:
            rgb: RGB图像，支持:
                - numpy数组 (H, W, 3) BGR格式
                - 文件路径字符串
                - bytes数据
            depth: 深度图，支持:
                - numpy数组 (H, W) uint16格式，单位mm
                - 文件路径字符串（16-bit PNG）
                - bytes数据
            chosen_policy: 输出策略，覆盖初始化设置
                - 'dn,vis': 去噪+可视化
                - 'dn': 仅去噪
                - 'vis': 仅可视化
            _timing: 可选的 dict，用于记录内部耗时分布
            **kwargs: 额外参数（预留）

        Returns:
            dict: 包含以下字段：
                - success: 是否成功
                - depth: 去噪后的深度图 numpy数组 (H, W) uint16
                - vis_image: 可视化图像 numpy数组 (H, W, 3) BGR（如果请求）
                - original_resolution: 原始分辨率
                - depth_resolution: 深度图分辨率
                - device: 计算设备
                - error: 错误信息（如果失败）
        """
        import time as _time
        _t0 = _time.time()

        policy = chosen_policy or self.chosen_policy

        # 处理 RGB 图像
        if isinstance(rgb, str):
            with open(rgb, 'rb') as f:
                rgb_bytes = f.read()
            rgb_filename = os.path.basename(rgb)
        elif isinstance(rgb, np.ndarray):
            _, rgb_encoded = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rgb_bytes = rgb_encoded.tobytes()
            rgb_filename = 'rgb.jpg'
        elif isinstance(rgb, bytes):
            rgb_bytes = rgb
            rgb_filename = 'rgb.jpg'
        else:
            raise TypeError(f'不支持的 RGB 类型: {type(rgb)}')

        _t1 = _time.time()  # RGB编码完成

        # 处理深度图
        if isinstance(depth, str):
            with open(depth, 'rb') as f:
                depth_bytes = f.read()
            depth_filename = os.path.basename(depth)
        elif isinstance(depth, np.ndarray):
            # 确保是 uint16 格式
            if depth.dtype != np.uint16:
                depth = depth.astype(np.uint16)
            _, depth_encoded = cv2.imencode('.png', depth)
            depth_bytes = depth_encoded.tobytes()
            depth_filename = 'depth.png'
        elif isinstance(depth, bytes):
            depth_bytes = depth
            depth_filename = 'depth.png'
        else:
            raise TypeError(f'不支持的深度图类型: {type(depth)}')

        _t2 = _time.time()  # Depth编码完成

        # 准备请求
        files = {
            'rgb': (rgb_filename, rgb_bytes, 'image/jpeg'),
            'dpt': (depth_filename, depth_bytes, 'image/png'),
        }
        data = {'chosen_policy': policy}

        try:
            response = self.session.post(self.api_url, files=files, data=data, timeout=60)
            _t3 = _time.time()  # HTTP请求完成

            # 记录时间分布
            if _timing is not None:
                _timing['cdm_rgb_encode'] = (_t1 - _t0) * 1000
                _timing['cdm_depth_encode'] = (_t2 - _t1) * 1000
                _timing['cdm_http'] = (_t3 - _t2) * 1000

            result = response.json()

            # 检查服务端错误
            if not response.ok or result.get('error'):
                error_msg = result.get('error', f'HTTP {response.status_code}')
                print(f"[DepthOptimizerOnline] 服务端错误: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                }

            if not result.get('success'):
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                }

            output = {
                'success': True,
                'device': result.get('device'),
                'original_resolution': result.get('original_resolution'),
                'depth_resolution': result.get('depth_resolution'),
                'chosen_policy': result.get('chosen_policy'),
            }

            # 解码去噪后的深度图
            if result.get('depth'):
                depth_b64 = result['depth']
                depth_data = base64.b64decode(depth_b64)
                depth_arr = cv2.imdecode(
                    np.frombuffer(depth_data, dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED  # 保持 16-bit
                )
                output['depth'] = depth_arr

            # 解码可视化图像
            if result.get('vis_images') and len(result['vis_images']) > 0:
                vis_b64 = result['vis_images'][0]
                vis_data = base64.b64decode(vis_b64)
                vis_arr = cv2.imdecode(
                    np.frombuffer(vis_data, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )
                output['vis_image'] = vis_arr

            return output

        except requests.exceptions.RequestException as e:
            print(f"[DepthOptimizerOnline] 请求失败: {e}")
            return {'success': False, 'error': str(e)}
        except json.JSONDecodeError as e:
            print(f"[DepthOptimizerOnline] JSON 解析失败: {e}")
            return {'success': False, 'error': str(e)}

    def optimize_depth(self, rgb, depth, **kwargs):
        """便捷接口：仅返回去噪后的深度图

        Args:
            rgb: RGB图像
            depth: 深度图
            **kwargs: 传递给 forward 的参数

        Returns:
            numpy.ndarray: 去噪后的深度图 (H, W) uint16，失败返回 None
        """
        result = self.forward(rgb, depth, chosen_policy='dn', **kwargs)
        if result.get('success') and 'depth' in result:
            return result['depth']
        return None

    def get_visualization(self, rgb, depth, **kwargs):
        """便捷接口：仅返回可视化图像

        Args:
            rgb: RGB图像
            depth: 深度图
            **kwargs: 传递给 forward 的参数

        Returns:
            numpy.ndarray: 可视化图像 (H, W, 3) BGR，失败返回 None
        """
        result = self.forward(rgb, depth, chosen_policy='vis', **kwargs)
        if result.get('success') and 'vis_image' in result:
            return result['vis_image']
        return None


if __name__ == '__main__':
    cfg = Config()
    cfg.server_list = r'config/server_grasp.json'
    cfg.model_name = 'full'
    service0 = GraspAnythingOnline(cfg)

    cfg = Config()
    cfg.uri = r'/v2/task/dinox/detection'
    cfg.status_uri = r'/v2/task_status'
    cfg.token = 'c4cdacb48bc4d1a1a335c88598a18e8c'
    service1 = DinoXDetectorCloud(cfg)

    # 初始化本地 DINO-X 服务
    cfg = Config()
    cfg.url = 'http://192.168.112.14:10086'
    cfg.min_score = 0.25
    cfg.iou_threshold = 0.5
    service2 = DinoXDetectorOnline(cfg)

    img_fp = 'samples/dino_test.jpg'
    rgb0 = cv2.imread(img_fp)

    if 0:  # GraspAnythingOnline test
        objs, padded_img = service0.forward(rgb=rgb0)
        print(objs)
        service0.vis(objs, rgb=rgb0.copy(), padded_img=padded_img, img_fp=img_fp)

    if 0:  # DinoXDetectorCloud test
        task = service1.forward(rgb=rgb0, text='pen.remote.bottle.eraser.battery.tape.rubik cube.tissue')
        print(task.result)

        # 可视化多目标检测结果
        vis_img = rgb0.copy()
        labels = 'pen.remote.bottle.eraser.battery.tape.rubik cube.tissue'.split('.')
        if task.result and 'objects' in task.result:
            for i, obj in enumerate(task.result['objects']):
                bbox = obj.get('bbox')
                if bbox:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    color = ((i * 50) % 255, (i * 80 + 100) % 255, (i * 120 + 50) % 255)
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                    label = labels[i] if i < len(labels) else f'obj{i}'
                    cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        out_fp = 'samples/refer_multi.jpg'
        cv2.imwrite(out_fp, vis_img)
        print(f'saved to {out_fp}')

        task = service1.forward(rgb=rgb0, text='tissue')
        print(task.result)

        # 可视化单目标检测结果
        vis_img = rgb0.copy()
        if task.result and 'objects' in task.result:
            for obj in task.result['objects']:
                bbox = obj.get('bbox')
                if bbox:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_img, 'tissue', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out_fp = 'samples/refer_tissue.jpg'
        cv2.imwrite(out_fp, vis_img)
        print(f'saved to {out_fp}')

    # 测试本地 DINO-X 服务 (DinoXDetectorOnline)
    if 0:  # DinoXDetectorOnline test
        print("\n=== 测试 DinoXDetectorOnline ===")
        task = service2.forward(rgb=rgb0, text='pen.remote.bottle.eraser.battery.tape.rubik cube.tissue')
        print(f"本地服务返回结果: {task.result}")

        # 可视化本地服务的多目标检测结果
        vis_img = rgb0.copy()
        labels = 'pen.remote.bottle.eraser.battery.tape.rubik cube.tissue'.split('.')
        if task.result and 'objects' in task.result:
            for i, obj in enumerate(task.result['objects']):
                bbox = obj.get('bbox')
                category = obj.get('category', f'obj{i}')
                score = obj.get('score', 0)
                if bbox:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    color = ((i * 50) % 255, (i * 80 + 100) % 255, (i * 120 + 50) % 255)
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                    label = f"{category}:{score:.2f}"
                    cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        out_fp = 'samples/refer_local_multi.jpg'
        cv2.imwrite(out_fp, vis_img)
        print(f'saved to {out_fp}')

        # 测试单目标检测
        task = service2.forward(rgb=rgb0, text='tissue')
        print(f"单目标检测结果: {task.result}")

        vis_img = rgb0.copy()
        if task.result and 'objects' in task.result:
            for obj in task.result['objects']:
                bbox = obj.get('bbox')
                score = obj.get('score', 0)
                if bbox:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_img, f'tissue:{score:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out_fp = 'samples/refer_local_tissue.jpg'
        cv2.imwrite(out_fp, vis_img)
        print(f'saved to {out_fp}')

    # 测试 DepthOptimizerOnline 深度图去噪服务
    if 0:  # DepthOptimizerOnline test
        print("\n=== 测试 DepthOptimizerOnline ===")
        cfg = Config()
        cfg.url = 'http://192.168.112.14:8086'
        cfg.chosen_policy = 'dn,vis'
        depth_service = DepthOptimizerOnline(cfg)

        # 使用 cdm 样本测试
        rgb_fp = 'samples/cdm/001-rgb.jpg'
        dpt_fp = 'samples/cdm/001-dpt.png'

        if os.path.isfile(rgb_fp) and os.path.isfile(dpt_fp):
            print(f"测试文件: RGB={rgb_fp}, Depth={dpt_fp}")

            # 方式1: 直接传文件路径
            result = depth_service.forward(rgb_fp, dpt_fp)
            print(f"文件路径方式结果: success={result.get('success')}")
            if result.get('success'):
                print(f"  设备: {result.get('device')}")
                print(f"  原始分辨率: {result.get('original_resolution')}")
                print(f"  深度图分辨率: {result.get('depth_resolution')}")
                if 'depth' in result:
                    print(f"  去噪深度图尺寸: {result['depth'].shape}, dtype={result['depth'].dtype}")
                    # 保存去噪后的深度图
                    out_depth_fp = 'samples/cdm/001-dpt-denoised.png'
                    cv2.imwrite(out_depth_fp, result['depth'])
                    print(f"  保存去噪深度图: {out_depth_fp}")
                if 'vis_image' in result:
                    print(f"  可视化图像尺寸: {result['vis_image'].shape}")
                    out_vis_fp = 'samples/cdm/001-dpt-vis-new.jpg'
                    cv2.imwrite(out_vis_fp, result['vis_image'])
                    print(f"  保存可视化图像: {out_vis_fp}")
            else:
                print(f"  错误: {result.get('error')}")

            # 方式2: 传 numpy 数组
            rgb_arr = cv2.imread(rgb_fp)
            dpt_arr = cv2.imread(dpt_fp, cv2.IMREAD_UNCHANGED)
            print(f"\nnumpy数组方式: RGB shape={rgb_arr.shape}, Depth shape={dpt_arr.shape}, dtype={dpt_arr.dtype}")

            result2 = depth_service.forward(rgb_arr, dpt_arr)
            print(f"numpy数组方式结果: success={result2.get('success')}")
            if result2.get('success'):
                if 'depth' in result2:
                    print(f"  去噪深度图尺寸: {result2['depth'].shape}, dtype={result2['depth'].dtype}")
            else:
                print(f"  错误: {result2.get('error')}")

            # 测试便捷接口
            print("\n测试便捷接口 optimize_depth:")
            optimized = depth_service.optimize_depth(rgb_fp, dpt_fp)
            if optimized is not None:
                print(f"  成功，尺寸: {optimized.shape}, dtype={optimized.dtype}")
            else:
                print("  失败")

            print("\n测试便捷接口 get_visualization:")
            vis = depth_service.get_visualization(rgb_fp, dpt_fp)
            if vis is not None:
                print(f"  成功，尺寸: {vis.shape}")
            else:
                print("  失败")
        else:
            print(f"测试文件不存在: {rgb_fp} 或 {dpt_fp}")

    # 测试 SAM2TrackerOnline 跟踪服务
    if 1:
        print("\n=== 测试 SAM2TrackerOnline ===")

        # 初始化 tracker
        cfg = Config()
        cfg.url = 'http://192.168.112.14:11086'
        tracker = SAM2TrackerOnline(cfg)

        # 使用 DinoXDetectorOnline 作为检测器
        cfg_det = Config()
        cfg_det.url = 'http://192.168.112.14:10086'
        cfg_det.min_score = 0.25
        detector = DinoXDetectorOnline(cfg_det)

        # 测试视频
        video_fp = 'samples/example_dancetrack0058.mp4'
        if os.path.isfile(video_fp):
            print(f"测试视频: {video_fp}")

            # 读取第一帧
            cap = cv2.VideoCapture(video_fp)
            ret, first_frame = cap.read()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            if not ret:
                print("无法读取视频第一帧")
            else:
                print(f"视频信息: {width}x{height}, {total_frames} 帧, {fps:.2f} FPS")

                # 使用 DinoXDetectorOnline 检测第一帧中的 people
                print("\n1. 检测第一帧中的 people...")
                task = detector.forward(text='people', rgb=first_frame)
                print(f"检测结果: {len(task.result.get('objects', []))} 个目标")

                if task.result and 'objects' in task.result and len(task.result['objects']) > 0:
                    objs = task.result['objects']

                    # 选择置信度最高且尽量居中的目标
                    print("\n2. 选择置信度最高且居中的目标...")
                    center_x = width / 2
                    center_y = height / 2

                    # 计算最大距离用于归一化
                    max_dist = (width / 2) ** 2 + (height / 2) ** 2

                    # 综合评分：score * 0.6 + (1 - norm_dist) * 0.4
                    # 置信度权重更高，同时考虑居中程度
                    best_score = -1
                    selected_obj = objs[0]

                    for obj in objs:
                        bbox = obj['bbox']
                        score = obj.get('score', 0.5)
                        obj_cx = (bbox[0] + bbox[2]) / 2
                        obj_cy = (bbox[1] + bbox[3]) / 2
                        dist = (obj_cx - center_x) ** 2 + (obj_cy - center_y) ** 2
                        norm_dist = dist / max_dist  # 归一化到 [0, 1]

                        # 综合评分：置信度 60% + 居中程度 40%
                        combined_score = score * 0.6 + (1 - norm_dist) * 0.4
                        print(f"  目标: bbox={[int(b) for b in bbox]}, score={score:.2f}, 距离={dist:.0f}, 综合={combined_score:.3f}")

                        if combined_score > best_score:
                            best_score = combined_score
                            selected_obj = obj

                    print(f"  选中目标: bbox={[int(b) for b in selected_obj['bbox']]}, score={selected_obj.get('score', 0):.2f}")

                    # 转换为 tracker 格式
                    init_dets = [{
                        'dt_bbox': selected_obj['bbox'],
                        'dt_score': selected_obj.get('score', 0.9),
                        'name': 'people',
                    }]

                    # 逐帧跟踪
                    print("\n3. 开始逐帧跟踪...")
                    cap = cv2.VideoCapture(video_fp)
                    output_fp = 'samples/example_dancetrack0058_tracked.mp4'

                    frame_idx = 0
                    tracked_count = 0
                    frames_to_save = []  # 收集帧用于 imageio 写入

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # 首帧传入检测结果，后续帧不传
                        dets = init_dets if frame_idx == 0 else None
                        result = tracker.forward(frame, dets=dets, frame_idx=frame_idx)

                        # 可视化
                        vis_frame = frame.copy()
                        if result.get('success') and 'boxes' in result and len(result['boxes']) > 0:
                            tracked_count += 1
                            for i, box in enumerate(result['boxes']):
                                x1, y1, x2, y2 = [int(b) for b in box]
                                track_id = result['ids'][i] if 'ids' in result and i < len(result['ids']) else 0
                                color = (0, 255, 0)  # 绿色
                                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                                label = f"ID:{track_id}"
                                cv2.putText(vis_frame, label, (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        # 帧号
                        cv2.putText(vis_frame, f'Frame: {frame_idx}', (20, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                        # BGR -> RGB for imageio
                        frames_to_save.append(vis_frame[:, :, ::-1])
                        frame_idx += 1

                        if frame_idx % 50 == 0:
                            print(f"  进度: {frame_idx}/{total_frames}")

                    cap.release()

                    # 使用 imageio 写入 mp4（更兼容）
                    print("  写入视频...")
                    import imageio
                    imageio.mimsave(output_fp, frames_to_save, fps=fps)

                    print(f"\n跟踪完成!")
                    print(f"  处理帧数: {frame_idx}")
                    print(f"  成功跟踪帧数: {tracked_count}")
                    print(f"  输出视频: {output_fp}")
                else:
                    print("第一帧未检测到目标，跳过跟踪测试")
        else:
            print(f"测试视频不存在: {video_fp}")
