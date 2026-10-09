"""OpenAI Responses image detector. No control commands or inferred depth from GPT."""
import base64
import json
import math
import os
import re
from pathlib import Path
import stat
import time
import cv2
import numpy as np
import requests
from .perception import BoardUnavailable, DetectionError, Region
from .object_center import object_observation, object_region

class DetectorFormatError(DetectionError):
    """Malformed model output; discard this frame and retry within the observation budget."""


DEFAULT_MODEL = 'gpt-4.1'
CREDENTIAL_PATH = Path.home()/'.config/robot-object-approach/openai.json'
SCHEMA = {
    'type':'object', 'additionalProperties':False,
    'properties':{
        'status':{'type':'string','enum':['found','not_found']},
        'label':{'type':'string'},
        'bbox':{'anyOf':[{'type':'null'}, {
            'type':'object','additionalProperties':False,
            'properties':{k:{'type':'number'} for k in ('x_min','y_min','x_max','y_max')},
            'required':['x_min','y_min','x_max','y_max']}]}},
    'required':['status','label','bbox']}


def read_settings(path=None):
    path = CREDENTIAL_PATH if path is None else path
    settings = {}
    if path.exists():
        info = path.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise DetectionError('GPT credential file must be owned by this user and chmod 600')
        try:
            settings = json.loads(path.read_text())
        except (ValueError, OSError):
            raise DetectionError('Cannot read GPT credential configuration') from None
        if not isinstance(settings, dict):
            raise DetectionError('Invalid GPT credential configuration')
    return settings


def response_text(payload):
    if isinstance(payload,dict) and payload.get('status') == 'incomplete':
        raise DetectorFormatError('Model response incomplete; discard this frame and retry')
    if not isinstance(payload, dict) or payload.get('status') != 'completed':
        raise DetectionError('Model response failed or has invalid status')
    texts = []
    for item in payload.get('output', []):
        if item.get('type') != 'message':
            continue
        if item.get('status') == 'incomplete':
            raise DetectorFormatError('Model message incomplete; discard this frame and retry')
        for content in item.get('content', []):
            if content.get('type') == 'refusal':
                raise DetectionError('GPT declined the detection request')
            if content.get('type') == 'output_text':
                texts.append(content.get('text', ''))
    return ''.join(texts)


def decode_response(payload, width, height):
    try:
        data = json.loads(response_text(payload))
    except (ValueError, TypeError):
        raise DetectionError('GPT returned invalid detection JSON') from None
    if not isinstance(data, dict) or set(data) != {'status','label','bbox'}:
        raise DetectionError('GPT returned an unexpected detection schema')
    status = data['status']
    if status == 'not_found':
        raise BoardUnavailable('GPT did not find the requested object')
    if status != 'found':
        raise DetectionError('Detector returned an invalid detection status')
    box = data['bbox']
    keys = ('x_min','y_min','x_max','y_max')
    if not isinstance(box, dict) or set(box) != set(keys):
        raise DetectionError('GPT returned an invalid bounding box')
    values = [box[k] for k in keys]
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0 <= v <= 1000 for v in values):
        raise DetectionError('GPT bounding box is outside normalized image bounds')
    x1,y1,x2,y2 = values
    if x2 <= x1 or y2 <= y1:
        raise DetectionError('GPT bounding box has inverted edges')
    return [x1*width/1000,y1*height/1000,x2*width/1000,y2*height/1000], data['label']


class GroundingResolutionError(DetectorFormatError):
    """Box is too small for reliable front-face refinement; never guess its units."""


def decode_grounding_response(payload, width, height):
    """Accept one tagged DeepSeek grounding box, explicitly normalized 0..1000."""
    text = response_text(payload).strip()
    if text in ('未找到', '未找到。', '没有', '没有。', 'not_found'):
        raise BoardUnavailable('DeepSeek did not find the requested object')
    match = re.fullmatch(r'<｜｜ref｜｜>([^<>]+)<｜｜/ref｜｜>\s*<｜｜box｜｜>(.*?)<｜｜/box｜｜>', text, re.DOTALL)
    if match:
        label=match.group(1).strip()
        try:
            boxes=json.loads(match.group(2))
        except ValueError:
            raise DetectorFormatError('DeepSeek returned invalid grounding coordinates') from None
        if not isinstance(boxes,list) or len(boxes)!=1:
            raise DetectorFormatError('DeepSeek grounding must contain exactly one target box')
        values=boxes[0]
    else:
        fenced=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',text,re.DOTALL)
        try:
            data=json.loads(fenced.group(1) if fenced else text)
        except ValueError:
            raise DetectorFormatError('DeepSeek returned unsupported grounding output') from None
        if not isinstance(data,dict) or set(data)!={'bbox_2d'}:
            raise DetectorFormatError('DeepSeek returned unexpected grounding JSON schema')
        values=data['bbox_2d'];label='target'
        if values is None:
            raise BoardUnavailable('DeepSeek did not find the requested object')
    if not isinstance(values, list) or len(values) != 4:
        raise DetectorFormatError('DeepSeek grounding box must have four coordinates')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1000 for v in values):
        raise DetectorFormatError('DeepSeek grounding coordinates outside normalized image bounds')
    x1,y1,x2,y2 = values
    if x2 <= x1 or y2 <= y1:
        raise DetectorFormatError('DeepSeek grounding box has inverted edges')
    return [x1*width/1000,y1*height/1000,x2*width/1000,y2*height/1000], label


def door_hint(payload, origin, crop_size, object_box):
    """Validate a door independently of the possibly incorrect face polygon."""
    text=response_text(payload).strip()
    blocks=re.findall(r'```(?:json)?\s*(.*?)```',text,re.DOTALL)
    if len(blocks)==1:text=blocks[0].strip()
    try:data=json.loads(text)
    except ValueError:return None
    if not isinstance(data,dict):return None
    values=data.get('door_bbox')
    if not isinstance(values,list) or len(values)!=4:return None
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1000 for v in values):return None
    door=np.asarray(values,float).reshape(2,2)*np.asarray(crop_size)/1000+origin
    size=door[1]-door[0];box=np.asarray(object_box,float);span=box[2:]-box[:2]
    if min(size)<12 or size.prod()<.08*span.prod() or size.prod()>1.2*span.prod():return None
    if (door[0]<box[:2]-.15*span).any() or (door[1]>box[2:]+.15*span).any():return None
    return values


def decode_front_corners(payload, origin, crop_size, object_box, image_shape, require_door=False):
    text = response_text(payload).strip()
    if '```' in text:
        blocks=re.findall(r'```(?:json)?\s*(.*?)```',text,re.DOTALL)
        if len(blocks)!=1 or text.count('```')!=2:
            raise DetectorFormatError('DeepSeek returned multiple or malformed front-corner blocks')
        text=blocks[0].strip()
    try:
        data = json.loads(text)
    except ValueError:
        raise DetectorFormatError('DeepSeek returned invalid front-corner JSON') from None
    expected={'corners','door_bbox'} if require_door else {'corners'}
    if not isinstance(data, dict) or set(data) != expected:
        raise DetectorFormatError('DeepSeek returned an invalid front-corner schema')
    if data['corners'] is None:
        raise BoardUnavailable('DeepSeek could not identify the front face')
    coords = data['corners']
    if (not isinstance(coords, list) or len(coords) != 4 or
        any(not isinstance(row,list) or len(row)!=2 or
            any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1000 for v in row)
            for row in coords)):
        raise DetectorFormatError('Invalid normalized front corners')
    polygon = (np.asarray(coords,float)*np.asarray(crop_size)/1000 + origin).astype(np.float32)
    if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon,oriented=True)<=0:
        raise DetectorFormatError('Front corners must form an ordered convex face')
    if not (polygon[0,1]<polygon[3,1] and polygon[1,1]<polygon[2,1]
            and polygon[0,0]<polygon[1,0] and polygon[3,0]<polygon[2,0]):
        raise DetectorFormatError('Front corners are not ordered top-left clockwise')
    box=np.asarray(object_box,float);size=box[2:]-box[:2]
    margin=np.maximum(3.,size*.05)
    if ((polygon<box[:2]-margin).any() or (polygon>box[2:]+margin).any()
            or cv2.contourArea(polygon)<.2*size.prod()):
        raise DetectorFormatError('Front face is outside detected object or too small')
    bounds=[*polygon.min(0).tolist(),*polygon.max(0).tolist()]
    object_region(image_shape,bounds,allow_bottom_crop=require_door)
    if require_door:
        polygon[:,1]=np.minimum(polygon[:,1],image_shape[0]-1)
        bounds=[*polygon.min(0).tolist(),*polygon.max(0).tolist()]
    region=Region(polygon,polygon.mean(0),[],source='object',score=None)
    region.allow_bottom_crop=require_door
    if require_door:
        values=data['door_bbox']
        if (not isinstance(values,list) or len(values)!=4 or
                any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1000 for v in values)):
            raise DetectorFormatError('Missing or invalid washer door exclusion box')
        door=np.asarray(values,float).reshape(2,2)*np.asarray(crop_size)/1000+origin
        size=door[1]-door[0]
        if (min(size)<12 or (door[0]<polygon.min(0)-3).any() or
                (door[1]>polygon.max(0)+3).any() or size.prod()<.08*cv2.contourArea(polygon)):
            raise DetectorFormatError('Washer door exclusion is outside face or too small')
        # The detected round door bounds define an ellipse, including a small rim margin.
        angles=np.linspace(0,2*np.pi,48,endpoint=False)
        ellipse=door.mean(0)+np.c_[np.cos(angles),np.sin(angles)]*(size*.54)
        region.exclusion_polygons=(ellipse.astype(np.float32),)
        region.exclusion_core_scale=.75
        region.allow_door_support=True
        region.region_scale=.90
        region.min_base_height=None
    return region,bounds


class GPTDetector:
    provider = 'openai'
    display_name = 'OpenAI'
    endpoint = 'https://api.openai.com/v1/responses'
    key_env = 'OPENAI_API_KEY'
    model_env = 'OPENAI_MODEL'
    default_model = DEFAULT_MODEL
    settings_path = None
    configure_script = 'configure_object_gpt.py'
    use_proxy = True

    def __init__(self, prompt, model=None, api_key=None, max_calls=30):
        settings = read_settings(self.settings_path) if api_key is None else {}
        self.api_key = api_key or os.environ.get(self.key_env) or settings.get('api_key')
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise DetectionError(f'Missing {self.display_name} API key: run python3 scripts/{self.configure_script} on robot02')
        self.model = model or os.environ.get(self.model_env) or settings.get('model') or self.default_model
        self.prompt, self.calls, self.max_calls = prompt, 0, max_calls
        self.last_metadata = None
        self.last_response = None
        self.session = requests.Session()
        self.session.trust_env = self.use_proxy
        # Provider keys stay isolated; destinations cannot be overridden by configuration.

    def decode_payload(self, payload, width, height):
        return decode_response(payload, width, height)

    def prepare_request(self, body):
        return body

    def detect(self, image, depth, k):
        self.last_metadata = None
        self.last_response = None
        if image.shape[:2] != depth.shape:
            raise DetectionError('RGB/depth dimensions differ')
        self.detect_region(image)
        observation = object_observation(depth,k,self.last_metadata['bbox'],1.)
        observation.score = None
        return observation

    def detect_region(self, image):
        """Use only the RGB image; geometry is supplied by a separate sensor."""
        self.last_metadata = None
        self.last_response = None
        if self.calls >= self.max_calls:
            raise DetectionError('GPT request budget exceeded for this run')
        ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY,95])
        if not ok:
            raise DetectionError('Cannot encode camera image')
        h,w = image.shape[:2]
        prompt = (
            f'检测图片中的{self.prompt}，返回名称和紧贴边界框。'
            '边界框只包含物体正面面板，排除侧面、顶部和背景；不要返回物体整体框。'
            '检测到时 status 为 found，未检测到时为 not_found、bbox 为 null。'
            'bbox 使用 x_min、y_min、x_max、y_max，坐标相对整张图片归一化到 0～1000，左上角为原点。'
            '按指定 JSON 格式返回。')
        body = dict(model=self.model, store=False, max_output_tokens=500,
            input=[dict(role='user',content=[dict(type='input_text',text=prompt),
                dict(type='input_image',detail='high',image_url='data:image/jpeg;base64,'+
                     base64.b64encode(jpeg).decode('ascii'))])],
            text=dict(format=dict(type='json_schema', name='object_location', strict=True,schema=SCHEMA)))
        body = self.prepare_request(body)
        started = time.monotonic()
        self.calls += 1
        self.last_metadata = None
        self.last_response = None
        try:
            response = self.session.post(self.endpoint,json=body,
                headers={'Authorization':'Bearer '+self.api_key}, timeout=(5,25), allow_redirects=False)
        except requests.RequestException:
            raise DetectionError(f'{self.display_name} request failed or timed out; check robot network') from None
        # Do not print exception request headers, response bodies or credentials.
        if response.status_code != 200:
            raise DetectionError(f'{self.display_name} API HTTP {response.status_code}; check key, model access, quota or network')
        try:
            payload = response.json()
        except ValueError:
            raise DetectionError('OpenAI API returned invalid JSON') from None
        elapsed = time.monotonic()-started
        if elapsed > 30:
            raise DetectionError('GPT observation exceeded 30 seconds; discard stale result')
        self.last_response = dict(provider=self.provider, model=self.model,
                                  response_id=payload.get('id'), status=payload.get('status'),
                                  output=payload.get('output'), usage=payload.get('usage'))
        bbox,label = self.decode_payload(payload,w,h)
        self.last_metadata = dict(provider=self.provider,model=self.model,response_id=payload.get('id'),
                                  elapsed_seconds=elapsed,bbox=bbox,label=label,request_count=self.calls)
        allow_bottom = self.provider in ('deepseek','qwen') and ('洗衣机' in self.prompt or 'washing machine' in self.prompt.lower())
        return object_region(image.shape, bbox, allow_bottom_crop=allow_bottom)


class DeepSeekDetector(GPTDetector):
    provider = 'deepseek'
    display_name = 'DeepSeek'
    endpoint = 'https://api.deepseek.com/responses'
    key_env = 'DEEPSEEK_API_KEY'
    model_env = 'DEEPSEEK_MODEL'
    default_model = 'deepseek-flash'
    settings_path = Path.home()/'.config/robot-object-approach/deepseek.json'
    configure_script = 'configure_object_deepseek.py'
    use_proxy = False

    def prepare_request(self, body):
        body.pop('store', None)  # DeepSeek Responses is stateless.
        body['reasoning'] = {'effort':'none'}
        body['max_output_tokens'] = 500
        body['temperature'] = 0
        # Structured JSON detection repeatedly returned not_found on visible targets.
        # Let the model emit its grounding tokens; parse and validate locally.
        body.pop('text', None)
        body['input'][0]['content'][0]['text'] = (
            f'找出图中的{self.prompt}，只框出其正面面板，边界框紧贴正面边缘，'
            '不包含侧面、顶部或背景，不要给整个机身的框。'
            '只返回JSON对象{"bbox_2d":[x_min,y_min,x_max,y_max]}。'
            'x为水平坐标，y为垂直坐标；坐标相对整张图片归一化到0到1000，不是像素坐标，也不是0到100的百分数。'
            '例如图像宽度30%的位置写300而不是30；高度43%的位置写430而不是43。请输出整数坐标。'
            '未找到时返回{"bbox_2d":null}，不要输出解释或多个框。')
        return body

    def decode_payload(self, payload, width, height):
        box,label = decode_grounding_response(payload, width, height)
        if min(box[2]-box[0],box[3]-box[1]) < 32:
            raise GroundingResolutionError('Detected face is smaller than 32 pixels; check 0..1000 units, do not crop or rescale automatically')
        return box,label


    def detect_region(self, image):
        started=time.monotonic()
        attempts=[]
        for attempt in range(2):
            try:
                super().detect_region(image)
                break
            except GroundingResolutionError:
                attempts.append(self.last_response)
                self.last_response={'grounding_attempts':attempts}
                if attempt == 1 or self.calls >= self.max_calls:
                    raise
        coarse=dict(self.last_metadata)
        first_response=self.last_response
        self.last_metadata=None  # Never expose an unrefined whole-object box as a usable face.
        self.last_response={'object_detection':first_response,'grounding_attempts':attempts}
        if self.calls>=self.max_calls:
            raise DetectionError('DeepSeek request budget exhausted before front refinement')
        box=np.asarray(coarse['bbox'],float);size=box[2:]-box[:2]
        origin=np.maximum(0,np.floor(box[:2]-.3*size)).astype(int)
        end=np.minimum([image.shape[1],image.shape[0]],np.floor(box[2:]+.3*size)).astype(int)
        crop=image[origin[1]:end[1],origin[0]:end[0]]
        crop_size=np.array([crop.shape[1],crop.shape[0]])
        self.last_response['crop_geometry']=dict(origin=origin.tolist(),size=crop_size.tolist(),object_bbox=coarse['bbox'])
        scale=min(3.,900./max(crop_size))
        enlarged=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        ok,jpeg=cv2.imencode('.jpg',enlarged,[cv2.IMWRITE_JPEG_QUALITY,95])
        if not ok:raise DetectionError('Cannot encode front-refinement crop')
        descriptor='正面是带圆形舱门的那一面，侧面即使有反光或圆形纹理也不能计入。' if '洗衣机' in self.prompt else '正面是物体的功能操作面，不包含侧面、顶部或背景。'
        prompt=(f'观察{self.prompt}正面与侧面的分界。'+descriptor+
                '返回正面四角的坐标，按左上、右上、右下、左下排列。'
                '只返回JSON对象{"corners":[[x,y],[x,y],[x,y],[x,y]]}，坐标相对当前图片归一化为0到1000。'
                '如果不能确定正面，返回{"corners":null}，不要返回整体机身的四角。')
        require_door='洗衣机' in self.prompt or 'washing machine' in self.prompt.lower()
        if require_door:
            prompt=(f'观察{self.prompt}。'+descriptor+
                    '返回正面四角（左上、右上、右下、左下），并返回圆形舱门的外接矩形，包含门沿。'
                    '只输出JSON对象{"corners":[[x,y],[x,y],[x,y],[x,y]],"door_bbox":[x_min,y_min,x_max,y_max]}。'
                    '坐标相对当前局部图片归一化0到1000。底部超出画面时返回可见正面四角，底边沿图像下边缘，不要推测画面外位置。'
                    '不能确定时对应字段为null，不要编造。')
        body=dict(model=self.model,max_output_tokens=500,reasoning={'effort':'none'},temperature=0,
                  input=[dict(role='user',content=[dict(type='input_text',text=prompt),
                         dict(type='input_image',detail='high',image_url='data:image/jpeg;base64,'+base64.b64encode(jpeg).decode('ascii'))])])
        self.calls+=1
        try:
            response=self.session.post(self.endpoint,json=body,headers={'Authorization':'Bearer '+self.api_key},
                                       timeout=(5,25),allow_redirects=False)
        except requests.RequestException:
            raise DetectionError(f'{self.display_name} front refinement request failed or timed out') from None
        if response.status_code!=200:
            raise DetectionError(f'{self.display_name} front refinement HTTP {response.status_code}')
        try:payload=response.json()
        except ValueError:raise DetectionError(f'{self.display_name} front refinement returned invalid JSON') from None
        self.last_response['front_refinement']=dict(response_id=payload.get('id'),status=payload.get('status'),
                                                    output=payload.get('output'),usage=payload.get('usage'))
        repaired=False
        try:
            region,bounds=decode_front_corners(payload,origin,crop_size,coarse['bbox'],image.shape,require_door=require_door)
        except (DetectorFormatError,BoardUnavailable):
            hint=door_hint(payload,origin,crop_size,coarse['bbox']) if require_door else None
            if hint is None or self.calls>=self.max_calls or time.monotonic()-started>20:
                raise
            repair_prompt=('只输出JSON，不要解释或分析。洗衣机正面是带圆门和上方控制面板的一面，不含侧板。'
                '舱门候选框为'+json.dumps(hint)+'，请据图核对并重新给出正面四角。'
                '坐标为当前图片0到1000整数，四角顺序左上、右上、右下、左下。'
                '格式：{"corners":[[x,y],[x,y],[x,y],[x,y]],"door_bbox":[x1,y1,x2,y2]}。'
                '舱门候选正确则保留；不能确认时对应字段为null。底部裁切时取可见底边。')
            body['input'][0]['content'][0]['text']=repair_prompt
            self.calls+=1
            try:
                response=self.session.post(self.endpoint,json=body,headers={'Authorization':'Bearer '+self.api_key},
                                           timeout=(5,max(1.,min(12.,29.-(time.monotonic()-started)))),allow_redirects=False)
            except requests.RequestException:
                raise DetectorFormatError('Door-guided front refinement timed out') from None
            if response.status_code!=200:raise DetectionError(f'Door-guided refinement HTTP {response.status_code}')
            try:payload=response.json()
            except ValueError:raise DetectorFormatError('Door-guided refinement returned invalid JSON') from None
            self.last_response['door_guided_refinement']=dict(door_hint=hint,response_id=payload.get('id'),
                status=payload.get('status'),incomplete_details=payload.get('incomplete_details'),output=payload.get('output'),usage=payload.get('usage'))
            updated_hint=door_hint(payload,origin,crop_size,coarse['bbox'])
            if updated_hint is None or np.max(np.abs(np.asarray(updated_hint)-hint))>50:
                raise DetectorFormatError('Door-guided refinement changed the door anchor')
            region,bounds=decode_front_corners(payload,origin,crop_size,coarse['bbox'],image.shape,require_door=True)
            # Check ellipse against the actual polygon, not just its bounding rectangle.
            door=np.asarray(updated_hint).reshape(2,2)*crop_size/1000+origin
            angles=np.linspace(0,2*np.pi,24,endpoint=False)
            rim=door.mean(0)+np.c_[np.cos(angles),np.sin(angles)]*((door[1]-door[0])*.5)
            if any(cv2.pointPolygonTest(region.polygon.astype(np.float32),tuple(map(float,p)),True)<-3 for p in rim):
                raise DetectorFormatError('Door rim is not contained in repaired front face')
            repaired=True
        elapsed=time.monotonic()-started
        if elapsed>30:raise DetectionError(f'{self.display_name} two-stage observation exceeded 30 seconds')
        self.last_metadata=dict(coarse,bbox=bounds,object_bbox=coarse['bbox'],front_polygon=region.polygon.tolist(),
                                refinement='door_guided_front_corners' if repaired else 'local_crop_front_corners',front_response_id=payload.get('id'),
                                crop_origin=origin.tolist(),crop_size=crop_size.tolist(),
                                exclusion_polygons=[p.tolist() for p in region.exclusion_polygons],region_scale=region.region_scale,
                                min_base_height=region.min_base_height,
                                exclusion_core_scale=getattr(region,'exclusion_core_scale',1.),
                                allow_door_support=getattr(region,'allow_door_support',False),
                                elapsed_seconds=elapsed,request_count=self.calls)
        return region
