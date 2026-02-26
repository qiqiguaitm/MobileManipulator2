import json
import multiprocessing as mp
import queue
import sys
import threading
import time
from typing import Optional, Union

import cv2
import h5py
import numpy as np
from mmengine.config import Config


def numpy_to_lists(obj):
    """递归将numpy数组转换为Python列表
    
    用于JSON序列化时将numpy数据类型转换为Python原生类型。
    递归处理字典、列表、元组等复合数据结构。
    
    Args:
        obj: 任意数据对象（字典、列表、numpy数组等）
        
    Returns:
        转换后的Python原生数据结构
    """
    if isinstance(obj, dict):
        # 递归处理字典中的每个值
        return {k: numpy_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # 递归处理列表/元组中的每个元素
        return [numpy_to_lists(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        # numpy数组转换为列表
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.float_, np.bool_)):
        # numpy标量转换为Python原生类型
        return obj.item()
    else:
        # 其他类型保持不变
        return obj


class Visualizer:
    def __init__(self, cfg):
        """
        可视化类，可以在线程或进程中运行，通过队列接收帧并显示

        参数:
            window_name: 窗口名称
            q_size: 队列最大大小，防止内存溢出
        """
        self.window_name = cfg.window_name
        self.q_size = cfg.q_size
        self.cfg = cfg
        self.input_queue: Union[queue.Queue, mp.Queue, None] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.process: Optional[mp.Process] = None
        self.enable_show = cfg.get('enable_show', False)
        self.enable_write = cfg.get('enable_write', True)
        self.enable_dpt = cfg.get('enable_dpt', True)
        self.out_video = cfg.get('out_video')
        self.out_msg = self.out_video.replace('.mp4', '.log')
        if self.enable_dpt:
            self.out_dpt = self.out_video.replace('.mp4', '.h5')

    def start(self, in_thread: bool = True):
        """启动可视化器

        支持三种运行模式：线程模式、进程模式、同步模式。
        根据系统要求选择合适的运行模式。

        Args:
            in_thread: 运行模式选择
                - True: 在线程中运行（推荐，占用资源少）
                - False: 在进程中运行（独立性强）
                - None: 同步运行（调试用）
        """
        if in_thread is True:
            # 线程模式：共享内存，通信开销小
            self.input_queue = queue.Queue(maxsize=self.q_size)
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.running = True
            self.thread.start()
        elif in_thread is False:
            # 进程模式：独立内存空间，更安全但开销大
            self.input_queue = mp.Queue(maxsize=self.q_size)
            self.process = mp.Process(target=self._run, daemon=True)
            self.running = True
            self.process.start()
        elif in_thread == None:
            # 同步模式：直接在当前线程运行，主要用于调试
            self.input_queue = queue.Queue(maxsize=self.q_size)
            self.running = True
            self._run()

    def stop(self):
        """停止可视化器"""
        self.running = False
        if self.thread is not None:
            self.thread.join()
        if self.process is not None and 0:
            self.process.join()

        if self.enable_show:
            cv2.destroyWindow(self.window_name)

    def put_frame(self, frame, dpt=None, msg=None):
        """将帧数据放入队列以供处理

        实现了智能的队列管理：当队列满时自动丢弃最旧的帧，
        确保实时性和防止内存溢出。

        Args:
            frame: 要处理的RGB图像帧 (numpy数组, 通常为(H,W,3))
            dpt: 深度图像 (numpy数组, 可选)
            msg: 状态信息字典 (包含检测结果、机械臂位置等)
        """
        if self.input_queue is not None:
            try:
                # 队列满时的智能管理策略
                if self.input_queue.full():
                    # 丢弃最旧的帧，保持实时性
                    try:
                        self.input_queue.get_nowait()
                    except queue.Empty:
                        pass
                        
                # 非阻塞式放入新帧
                self.input_queue.put_nowait((frame, dpt, msg))
            except (queue.Full, mp.queues.Full):
                # 队列依然满时静默忽略，防止阻塞主线程
                pass

    def _run(self):
        msg_log = open(self.out_msg, 'w')
        if self.enable_show:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.cfg.img_w, self.cfg.img_h)
        if self.enable_write:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
            fps = self.cfg.get('fps', 30)
            out = cv2.VideoWriter(self.cfg.out_video, fourcc, fps, (self.cfg.img_w, self.cfg.img_h), True)
        if self.enable_dpt:
            dpt_fo = h5py.File(self.out_dpt, 'w')
            img_h = self.cfg.get('dpt_img_h', self.cfg.img_h)
            img_w = self.cfg.get('dpt_img_w', self.cfg.img_w)
            frame_shape = (img_h, img_w)
            max_frames = None  # None 表示无限扩展（实际受磁盘空间限制）

            dset = dpt_fo.create_dataset(
                'frames',
                shape=(0, *frame_shape),  # 初始帧数为0
                maxshape=(max_frames, *frame_shape),  # 可扩展维度
                dtype=np.float32,
                chunks=(1, *frame_shape),  # 分块存储（每帧一个chunk）
            )
        frame_id = 0
        while self.running:
            try:
                frame, dpt, msg = self.input_queue.get(timeout=0.1)
                if self.enable_show:
                    cv2.imshow(self.window_name, frame)
                    key = cv2.waitKey(1)

                    if key & 0xFF == ord('q'):
                        self.running = False

                if self.enable_write:
                    # breakpoint()
                    if frame.shape[0] != self.cfg.img_h or frame.shape[1] != self.cfg.img_w:
                        print(f'wrong shape: {frame.shape=}')
                    out.write(frame)

                if self.enable_dpt:
                    dset.resize(dset.shape[0] + 1, axis=0)
                    dset[-1] = dpt

                if isinstance(msg, dict):
                    msg = numpy_to_lists(msg)
                    msg = json.dumps(msg)
                msg_log.write(f'{frame_id}: {msg}\n')
                # print(f'{msg}')
                frame_id += 1
            except queue.Empty:
                continue
            except Exception as e:
                print(f'Visualizer error: {e}')
                self.running = False
            finally:
                # out.flush()
                pass
            # print('-')
        msg_log.flush()
        msg_log.close()
        if self.enable_write:
            out.release()
            print(f'visualizer save video to :{self.cfg.out_video}')
        if self.enable_dpt:
            dpt_fo.close()
            print(f'visualizer save depth to :{self.out_dpt}')

        # print('done _run')  # won't run this if stop()


# 使用示例
if __name__ == '__main__':
    cfg = Config()
    cfg.window_name = 'Demo'
    cfg.q_size = 10
    cfg.img_h = 720
    cfg.img_w = 1280
    cfg.dpt_img_h = 720
    cfg.dpt_img_w = 1280
    cfg.enable_dpt = True
    cfg.out_video = 'test_vis.mp4'
    cfg.enable_show = True
    visualizer = Visualizer(cfg=cfg)

    # 在子线程中启动
    visualizer.start(in_thread=False)

    # 模拟生成一些帧并发送到可视化器
    i = 0
    while True:
        # 创建一个简单的彩色渐变图像
        frame = np.zeros((cfg.img_h, cfg.img_w, 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(0, 255, cfg.img_w, dtype=np.uint8)  # 蓝色通道渐变
        frame[:, :, 1] = i * 2 % 256  # 绿色通道随时间变化
        frame[:, :, 2] = 128  # 红色通道固定

        dpt = np.random.uniform(0.0, 10.0, (cfg.img_h, cfg.img_w)).astype(np.float32)
        msg = dict(action='fake')
        cv2.putText(frame, f'Frame {i}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 发送帧到可视化器
        visualizer.put_frame(frame, dpt=dpt, msg=msg)
        time.sleep(0.05)
        i += 1
        if i > 200:
            break
    # 停止可视化器
    # time.sleep(1)  # 等待最后几帧显示
    print('to stop')
    visualizer.running = False
    visualizer.stop()
    print('done')
