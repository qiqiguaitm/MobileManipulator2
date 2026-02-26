import os
import sys
import time

import cv2
import numpy as np
import torch
from mmengine.config import Config
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.tracker import matching
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking

# tracker = BYTETracker(self.cfg)
# for image in images:
#     dets = detector(image)
#     online_targets = tracker.update(dets, info_imgs, img_size)


class Tracker(BYTETracker):
    """多目标跟踪器类，基于BYTETracker实现
    
    继承自BYTETracker并扩展功能，实现对检测到的多个目标进行
    跨帧跟踪和ID分配。支持目标出现、消失、重新出现等情况的处理。
    
    主要特性：
    - 多目标同时跟踪
    - ID保持一致性
    - 跨帧关联匹配
    - 目标状态管理
    """
    def __init__(self, cfg):
        """初始化跟踪器
        
        Args:
            cfg: 配置对象，包含跟踪参数：
                - fps: 帧率
                - track_thresh: 跟踪阈值
                - track_buffer: 跟踪缓冲器大小
                - match_thresh: 匹配阈值
                - aspect_ratio_thresh: 长宽比阈值
                - min_box_area: 最小边界框面积
        """
        super().__init__(cfg, frame_rate=cfg.get('fps', 30))
        self.cfg = cfg

    def update2(self, objs, height, width):
        """更新跟踪结果并分配ID
        
        根据当前帧的检测结果更新跟踪器状态，并为每个检济到的目标
        分配一个唯一的track_id。使用匈牙利算法进行最优匹配。
        
        Args:
            objs: 检测结果列表，包含边界框和置信度
            height: 图像高度
            width: 图像宽度
            
        Returns:
            更新后的检测结果，每个目标包含track_id字段
        """
        # 准备检测数据：边界框 + 置信度
        dts = []
        for obj in objs[0]:
            # 格式：[x1, y1, x2, y2, score]
            tmp = [*obj['dt_bbox'], obj['dt_score']]
            dts.append(tmp)
            obj['track_id'] = -1  # 初始化为无效ID
        dts = torch.tensor(dts)

        # 调用ByteTracker的update方法进行跟踪
        targets = self.update(dts, [height, width], [height, width])
        
        # 提取跟踪目标的边界框（tlwh格式转换为xyxy格式）
        target_bbox = [t.tlwh for t in targets]  # top-left width height
        target_bbox = np.array(target_bbox)
        target_bbox[:, 2] = target_bbox[:, 0] + target_bbox[:, 2]  # x1 + w = x2
        target_bbox[:, 3] = target_bbox[:, 1] + target_bbox[:, 3]  # y1 + h = y2

        # 提取跟踪目标的ID
        target_ids = [t.track_id for t in targets]
        
        # 计算IoU距离矩阵进行关联匹配
        ious = matching.ious(target_bbox, dts[:, :4])  # 计算交并比
        dists = 1 - ious  # IoU转换为距离度量
        
        # 使用匈牙利算法进行最优二分匹配
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=0.1)  # matches: 匹配对, u_track: 未匹配跟踪, u_detection: 未匹配检测
        
        # 为匹配成功的检测目标分配track_id
        for itracked, idet in matches:
            objs[0][idet]['track_id'] = target_ids[itracked]
            
        return objs

    def track_video(self, fp, fp_out, **kwargs):
        """对视频文件执行多目标跟踪
        
        读取视频文件，逐帧进行目标检测和跟踪，并将结果可视化输出。
        支持调试模式和帧数限制。
        
        Args:
            fp: 输入视频文件路径
            fp_out: 输出视频文件路径
            **kwargs: 额外参数
                - debug: 是否为调试模式
                - max_count: 最大处理帧数
        """
        # 检查输入文件存在
        if not os.path.isfile(fp):
            print(f'file not found: {fp}')
            return
            
        # 导入并初始化检测器
        from percept import GraspAnythingOnline

        cfg0 = Config()
        cfg0.server_list = r'/comp_robot/common_config/server_grasp.json'
        cfg0.model_name = 'full2'
        det = GraspAnythingOnline(cfg=cfg0)

        vi = cv2.VideoCapture(fp)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
        vo = None
        # vo = cv2.VideoWriter(fp, fourcc, fps, (width, height), True)
        frame_id = 0
        results = []
        while vi.isOpened():
            ret, fr = vi.read()
            frame_id += 1
            if not ret:
                print(f'pass: {ret}')
                break
            if kwargs.get('debug', False) and frame_id > kwargs.get('max_count', 100):
                break
            if frame_id % 5 != 0:
                continue
            if frame_id % 50 == 0:
                print(f'process {frame_id=}')
            height, width, _ = fr.shape
            if vo is None:
                fps = vi.get(cv2.CAP_PROP_FPS)
                vo = cv2.VideoWriter(fp_out, fourcc, fps, (width, height), True)

            start = time.time()
            dt, padded_img = det.forward(rgb=fr, use_touching_points=False, bag=False)
            if 0:
                dts = []  # [x1, y1, x2, y2, score]
                for obj in dt[0]:
                    # breakpoint()
                    tmp = [*obj['dt_bboxes'], obj['dt_scores']]
                    dts.append(tmp)
                dts = torch.tensor(dts)
                print(dts)
                if dts.shape[0] > 0:
                    # breakpoint()
                    online_targets = self.update(dts, [height, width], [height, width])
                    online_tlwhs = []  # top-left, wh
                    online_ids = []  # track ids
                    online_scores = []  # track scores
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > self.cfg.aspect_ratio_thresh
                        if tlwh[2] * tlwh[3] > self.cfg.min_box_area and not vertical:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            results.append(f'{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n')
                    end = time.time()
                    print(f'{frame_id=} {online_tlwhs=} {online_ids=}')
                    online_im = plot_tracking(fr, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1.0 / (end - start))
                    # cv2.imwrite(f'vis/{frame_id}.jpg', online_im)
                    # breakpoint()
                    vo.write(online_im)
            else:
                objs = self.update2(dt, height=height, width=width)
                for obj in objs[0]:
                    if obj['track_id'] == -1:
                        continue
                    bb = obj['dt_bboxes']
                    cv2.rectangle(fr, (bb[0], bb[1]), (bb[2], bb[3]), color=(0, 255, 0), thickness=2)
                    cv2.putText(
                        fr,
                        f'{obj["track_id"]}',
                        (bb[0], bb[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )
                vo.write(fr)

        vi.release()
        if vo is not None:
            vo.release()


if __name__ == '__main__':
    cfg = Config()
    cfg.aspect_ratio_thresh = 1.6
    cfg.min_box_area = 10
    cfg.track_buffer = 300
    cfg.track_thresh = 0.12  # obj.score >= track_thresh will be treated as positive obj. obj.score in [0.1, track_thresh] will be considered high possible objs. obj.score <0.1 will be ignore
    cfg.det_thresh = 0.2  # obj.score >=det_thresh will be activated
    cfg.match_thresh = 0.8  # only dist <= match_thresh are allowed
    cfg.mot20 = False
    debug = False
    max_count = 500
    fp = r'/Users/shock/workspace/cocoopt/data/demo/grasp.mp4'
    fp = r'/Users/shock/workspace/cocoopt/data/demo/screen.mp4'
    fp = r'/Users/shock/workspace/cocoopt/data/demo/mobile.mp4'
    # fp = r'/Users/shock/workspace/ByteTrack/videos/palace.mp4'
    fp_out = 'vis.mp4'
    tracker = Tracker(cfg=cfg)
    tracker.track_video(fp, fp_out=fp_out, debug=debug, max_count=max_count)
