"""Qwen-VL Chat Completions transport for the shared visual grounding pipeline."""
from pathlib import Path
from types import SimpleNamespace
from .gpt_detector import DeepSeekDetector, read_settings, DetectorFormatError
from .perception import DetectionError

ENDPOINTS = {
    'beijing':'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
    'singapore':'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions',
}


class QwenTransport:
    def __init__(self, session):self.session=session

    def post(self, url, *, json, **kwargs):
        messages=[]
        for message in json['input']:
            content=[]
            for item in message['content']:
                if item['type']=='input_text':content.append(dict(type='text',text=item['text']))
                elif item['type']=='input_image':content.append(dict(type='image_url',image_url=dict(url=item['image_url'])))
                else:raise DetectionError('Unsupported Qwen input content')
            messages.append(dict(role=message['role'],content=content))
        body=dict(model=json['model'],messages=messages,max_tokens=json.get('max_output_tokens',500),
                  temperature=0,enable_thinking=False,stream=False)
        response=self.session.post(url,json=body,**kwargs)
        if response.status_code!=200:return response
        try:raw=response.json()
        except ValueError:raise DetectorFormatError('Qwen returned invalid JSON') from None
        choices=raw.get('choices',[])
        if len(choices)!=1:raise DetectorFormatError('Qwen response must contain one choice')
        choice=choices[0];reason=choice.get('finish_reason');content=choice.get('message',{}).get('content')
        if reason not in ('stop','length'):raise DetectionError('Qwen response did not finish normally')
        if not isinstance(content,str):raise DetectorFormatError('Qwen returned no text')
        status='completed' if reason=='stop' else 'incomplete'
        payload=dict(id=raw.get('id'),status=status,usage=raw.get('usage'),
            incomplete_details={'reason':'max_output_tokens'} if reason=='length' else None,
            output=[dict(type='message',status=status,content=[dict(type='output_text',text=content)])])
        return SimpleNamespace(status_code=200,json=lambda:payload)


class QwenDetector(DeepSeekDetector):
    provider='qwen'
    display_name='Qwen'
    key_env='DASHSCOPE_API_KEY'
    model_env='QWEN_MODEL'
    default_model='qwen3-vl-plus'
    settings_path=Path.home()/'.config/robot-object-approach/qwen.json'
    configure_script='configure_object_qwen.py'
    endpoint=ENDPOINTS['beijing']
    use_proxy=False

    def prepare_request(self, body):
        body = super().prepare_request(body)
        # Locate the object first; the inherited crop stage isolates its front face.
        body['input'][0]['content'][0]['text'] = (
            f'找出图中的{self.prompt}，返回紧贴物体的边界框。'
            '只返回JSON对象{"bbox_2d":[x_min,y_min,x_max,y_max]}。'
            'x为水平坐标，y为垂直坐标；坐标相对整张图片归一化到0到1000，不是像素坐标，也不是0到100的百分数。'
            '例如图像宽度30%的位置写300而不是30；高度43%的位置写430而不是43。请输出整数坐标。'
            '未找到时返回{"bbox_2d":null}，不要输出解释或多个框。')
        return body

    def __init__(self,prompt,model=None,api_key=None,max_calls=30):
        settings=read_settings(self.settings_path) if api_key is None else {}
        region=settings.get('region','beijing')
        if region not in ENDPOINTS:raise DetectionError('Unsupported Qwen region')
        super().__init__(prompt,model=model,api_key=api_key,max_calls=max_calls)
        self.endpoint=ENDPOINTS[region]
        self.session=QwenTransport(self.session)
