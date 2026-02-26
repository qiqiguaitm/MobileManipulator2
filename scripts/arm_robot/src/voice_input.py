import platform
import threading
from queue import Queue

import numpy as np
import sounddevice as sd
from funasr import AutoModel
from mmengine.config import Config

SAMPLE_RATE = 16000  # FunASR标准输入采样率
CHUNK_DURATION = 1  # 每次处理1秒音频
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)


class VoiceInput:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hotword = cfg.hotword
        self.keyword = cfg.keyword
        self.audio_q = Queue()  # in audio
        self.cmd_q = Queue()  # output cmd
        self.is_recording = True

        # === 初始化主要ASR模型 ===
        self.model = AutoModel(
            model='paraformer-zh',              # 中文语音识别模型
            model_revision='v2.0.4',           # 模型版本
            vad_model='fsmn-vad',              # 语音活动检测模型
            vad_model_revision='v2.0.4',       # VAD模型版本
            vad_kwargs={'max_single_segment_time': 3000},  # VAD参数：最大单段时长3秒
            disable_update=True,               # 禁用自动更新
            disable_log=True,                  # 禁用日志输出
            disable_pbar=True,                 # 禁用进度条
            use_timestamp=False,               # 不使用时间戳
        )
        # === 平台特定模型配置 ===
        self.os = platform.system()
        if self.os == 'Linux' and 0:  # Linux下的关键词检测模型（暂时禁用）
            self.model0 = AutoModel(
                model='iic/speech_sanm_kws_phone-xiaoyun-commands-offline',  # 离线关键词检测模型
                keywords=self.keyword,           # 关键词列表
                output_dir='.',                 # 输出目录
                device='cpu',                  # 使用CPU推理
                disable_update=True,
                disable_log=True,
                disable_pbar=True,
                use_timestamp=False,
            )
        else:
            # 默认使用通用ASR模型
            self.model0 = self.model
            
        # === 状态管理 ===
        self.state = 'wait'                    # 初始状态：等待唤醒词
        self.debug = cfg.get('debug', False)   # 调试模式开关

    def listen(self):
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK_SIZE, callback=self.audio_callback):
            while True:
                sd.sleep(1)

    # 音频录制回调
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f'audio input error: {status}')
        if self.is_recording:
            # print(frames)  # int, == chunksize
            # print(time)  # PaStreamCallbackTimeInfo, ['currentTime', 'inputBufferAdcTime', 'outputBufferDacTime']
            # print(status)  # <sounddevice.CallbackFlags, [input_underflow', 'output_overflow', 'output_underflow', 'priming_output']
            # breakpoint()
            self.audio_q.put(indata.copy())

    def asr_process(self):
        # audio_buffer = np.array([], dtype=np.float32)
        command = ''
        cache = {}
        cmd_buff = []
        while self.is_recording:
            try:
                # === 获取并预处理音频数据 ===
                chunk = self.audio_q.get()      # 从音频队列获取音频块
                chunk = chunk.squeeze()         # 去除多余维度，确保为1D数组

                if 1:
                    # === ASR语音识别 ===
                    result = self.model0.generate(input=chunk)
                    
                    # 处理识别结果
                    if len(result) == 0:
                        if self.debug:
                            print(f'empty result: {chunk.shape}')
                        result = [{'text': ''}]  # 空结果的默认格式
                        
                    text = result[0]['text']  # 提取识别的文本
                    
                    # Linux系统下的特殊处理
                    if self.os == 'Linux' and text == 'rejected':
                        print(f'{text}')
                        text = ''  # 被拒绝的识别结果置空
                        
                    # === 根据识别结果更新状态机 ===
                    if not text:  # 识别结果为空
                        if self.debug and 0:
                            print(f'invalid audio chunk text: {text=} {self.state=}')
                        if self.state == 'partial-cmd':     # 部分命令状态下的空识别
                            cmd_buff.append(chunk)              # 可能是唤醒词后的延续
                            self.state = 'cmd'                  # 转换到完整命令状态
                        elif self.state == 'cmd':              # 命令状态下的空识别（异常）
                            print(f'unexpected state: {self.state=} {command=} {text=}')
                        else:  # 等待状态下的空识别（正常静音）
                            self.state = 'wait'
                    else:  # 有识别文本的情况
                        # === 文本预处理 ===
                        text = text.strip().lower().replace(' ', '')  # 标准化：去空格、小写
                        
                        # === 关键词匹配和状态转换 ===
                        if self.keyword in text:               # 检测到唤醒词
                            cmd_buff.append(chunk)             # 将音频块加入命令缓冲
                            self.state = 'partial-cmd'         # 转换到部分命令状态
                        else:                                   # 未检测到唤醒词
                            if self.state == 'wait':           # 等待状态：忽略无关语音
                                pass
                            elif self.state == 'partial-cmd':  # 部分命令状态：继续收集
                                cmd_buff.append(chunk)
                            else:                               # 其他状态：异常
                                print(f'unexpected state: {self.state=} {command=} {text=}')
                                
                        # 调试输出
                        if self.debug:
                            print(f'audio chunk text: {text=} {self.state=}')

                    # === 完整命令处理 ===
                    if self.state == 'cmd':
                        # 拼接所有音频块进行完整识别
                        cmd_buff = np.concatenate(cmd_buff)
                        
                        # 使用主模型进行精确识别（带热词增强）
                        result = self.model.generate(
                            input=cmd_buff,                    # 完整音频数据
                            hotword=self.hotword,             # 热词列表，提高特定词汇识别率
                            is_final=True,                    # 最终识别标志
                            chunk_size=len(cmd_buff),         # 音频块大小
                            batch_size_s=3,                   # 批处理大小（秒）
                            batch_size_threshold_s=3,         # 批处理阈值（秒）
                            rich_transcription_postprocess=True,  # 丰富转录后处理
                        )
                        
                        # === 命令解析和输出 ===
                        if result and len(result) > 0:
                            # 提取并处理命令文本
                            command = result[0]['text'].strip().lower().replace(' ', '')
                            # 移除唤醒词，提取实际命令
                            command = command.split(self.keyword)[-1].strip()
                            print(f'command: {command}')
                            
                            # 将有效命令放入输出队列
                            if len(command) > 0:
                                self.cmd_q.put(command)
                                
                        # === 重置状态 ===
                        cache = {}                  # 清空缓存
                        cmd_buff = []              # 清空命令缓冲
                        self.state = 'wait'        # 回到等待状态
                else:
                    # print(f'unexpected {audio_buffer.shape}')
                    pass
            except Exception as e:
                # === 异常处理和状态重置 ===
                cache = {}                      # 清空ASR缓存
                cmd_buff = []                   # 清空命令缓冲
                self.state = 'wait'            # 重置为等待状态
                
                if self.debug:
                    print(f'语音识别异常: {e}')
                    import traceback
                    traceback.print_exc()      # 调试模式下打印详细错误信息
                pass


def act(q):
    while True:
        try:
            cmd = q.get(block=True)
            print(f'act: {cmd}')
        except Exception as e:
            print(f'{e}')


def main():
    cfg = Config()
    cfg.hotword = '恐龙,抓,黑色,红色,黄色,物体,盒子,桌子'
    cfg.keyword = '恐龙'
    cfg.debug = True
    voc = VoiceInput(cfg=cfg)

    print('🎤 开始录音 (按Ctrl+C停止)...')

    asr_thread = threading.Thread(target=voc.asr_process)
    # asr_thread.daemon = True
    asr_thread.start()

    actor_thread = threading.Thread(target=act, args=(voc.cmd_q,))
    # actor_thread.daemon = True
    actor_thread.start()

    listen_thread = threading.Thread(target=voc.listen)
    listen_thread.start()
    listen_thread.join()


if __name__ == '__main__':
    main()
