import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
import requests
import board_approach.gpt_detector as gpt
from board_approach.perception import DetectionError, BoardUnavailable


def result(status='found',bbox=None):
    return dict(status=status,label='washing machine',
                bbox=bbox or dict(x_min=250,y_min=200,x_max=650,y_max=800))


def response(data):
    return dict(status='completed',id='test_response',output=[dict(type='message',content=[
        dict(type='output_text',text=json.dumps(data))])])


def test_normalized_coordinates_and_explicit_failures():
    box,label=gpt.decode_response(response(result()),640,480)
    assert box == [160,96,416,384]
    assert label == 'washing machine'
    with pytest.raises(BoardUnavailable):
        gpt.decode_response(response(result('not_found')),640,480)
    for data in [result('ambiguous'),result('uncertain'),
                 result(bbox=dict(x_min=900,y_min=10,x_max=500,y_max=600)),
                 result(bbox=dict(x_min=float('nan'),y_min=10,x_max=500,y_max=600))]:
        with pytest.raises(DetectionError):
            gpt.decode_response(response(data),640,480)
    with pytest.raises(DetectionError,match='incomplete'):
        gpt.decode_response(dict(status='incomplete'),640,480)
    with pytest.raises(DetectionError,match='declined'):
        gpt.decode_response(dict(status='completed',output=[dict(type='message',content=[dict(type='refusal')])]),640,480)


def test_actual_request_schema_and_depth_fusion_without_network():
    detector=gpt.GPTDetector('washing machine',api_key='test-only-not-a-secret')
    calls=[]
    def post(url,**kwargs):
        assert url=='https://api.openai.com/v1/responses'
        assert kwargs['allow_redirects'] is False
        body=kwargs['json']
        assert body['store'] is False
        assert body['text']['format']['strict'] is True
        schema=body['text']['format']['schema']
        assert schema['properties']['status']['enum']==['found','not_found']
        assert set(schema['required'])=={'status','label','bbox'}
        prompt=body['input'][0]['content'][0]['text']
        assert 'washing machine' in prompt
        assert '边界框只包含物体正面面板' in prompt
        assert '不要返回物体整体框' in prompt
        assert not any(word in prompt.lower() for word in ['occlu', 'uncertain', '遮挡', '不确定'])
        assert body['input'][0]['content'][1]['type']=='input_image'
        assert body['input'][0]['content'][1]['image_url'].startswith('data:image/jpeg;base64,')
        calls.append(1)
        return NS(status_code=200,json=lambda:response(result()))
    detector.session=NS(post=post)
    image=np.zeros((480,640,3),np.uint8)
    depth=np.full((480,640),2.)
    k=np.array([[640.,0,320.],[0,640.,240.],[0,0,1.]])
    obs=detector.detect(image,depth,k)
    assert obs.ids==[] and obs.source=='object_depth' and obs.score is None
    assert obs.center[2]==pytest.approx(2.)
    assert detector.last_metadata['response_id']=='test_response'
    assert len(calls)==1


def test_http_failure_and_budget_never_return_an_observation():
    detector=gpt.GPTDetector('board',api_key='test-only',max_calls=1)
    detector.session=NS(post=lambda *a,**kw:NS(status_code=401))
    image=np.zeros((20,20,3),np.uint8); depth=np.ones((20,20));k=np.eye(3)
    with pytest.raises(DetectionError,match='HTTP 401'):
        detector.detect(image,depth,k)
    with pytest.raises(DetectionError,match='budget'):
        detector.detect(image,depth,k)


def test_image_only_detection_does_not_require_or_invent_depth(monkeypatch):
    detector=gpt.GPTDetector('pillar',api_key='test-only')
    detector.session=NS(post=lambda *a,**kw:NS(status_code=200,json=lambda:response(result())))
    def forbidden(*args):raise AssertionError('Depth lookup in image-only detection')
    monkeypatch.setattr(gpt,'object_observation',forbidden)
    region=detector.detect_region(np.zeros((480,640,3),np.uint8))
    assert region.source=='object' and not hasattr(region,'center')
    assert detector.last_metadata['bbox']==[160,96,416,384]


def test_no_key_and_private_file_permissions(monkeypatch,tmp_path):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    path=tmp_path/'openai.json';monkeypatch.setattr(gpt,'CREDENTIAL_PATH',path)
    with pytest.raises(DetectionError,match='Missing OpenAI API key'):
        gpt.GPTDetector('board')
    path.write_text(json.dumps(dict(api_key='test-only')));path.chmod(0o644)
    with pytest.raises(DetectionError,match='chmod 600'):
        gpt.GPTDetector('board')
    path.chmod(0o600)
    assert gpt.GPTDetector('board').model=='gpt-4.1'


def test_deepseek_direct_endpoint_and_separate_credentials(monkeypatch,tmp_path):
    monkeypatch.setenv('OPENAI_API_KEY','openai-must-not-be-reused')
    monkeypatch.delenv('DEEPSEEK_API_KEY',raising=False)
    monkeypatch.setenv('HTTPS_PROXY','http://127.0.0.1:1')
    monkeypatch.setattr(gpt.DeepSeekDetector,'settings_path',tmp_path/'deepseek.json')
    with pytest.raises(DetectionError,match='Missing DeepSeek API key'):
        gpt.DeepSeekDetector('washing machine')
    d=gpt.DeepSeekDetector('washing machine',api_key='deepseek-test-only')
    assert d.session.trust_env is False
    assert d.model=='deepseek-flash'
    def post(url,**kwargs):
        assert url=='https://api.deepseek.com/responses'
        assert kwargs['headers']['Authorization']=='Bearer deepseek-test-only'
        assert kwargs['allow_redirects'] is False
        body=kwargs['json']
        assert 'store' not in body
        assert body['reasoning']=={'effort':'none'}
        assert body['max_output_tokens']==500
        assert 'text' not in body
        prompt=body['input'][0]['content'][0]['text']
        if '返回正面四角' in prompt:
            return NS(status_code=200,json=lambda:response({'corners':[[190,190],[810,190],[810,815],[190,815]],'door_bbox':[290,290,700,700]}))
        assert '只框出其正面面板' in prompt
        return NS(status_code=200,json=lambda:grounding_response('[[250,200,650,800]]'))
    d.session.post=post
    o=d.detect(np.zeros((480,640,3),np.uint8),np.full((480,640),2.),
               np.array([[640.,0,320.],[0,640.,240.],[0,0,1.]]))
    assert o.center[2]==pytest.approx(2.)
    assert d.last_metadata['provider']=='deepseek'


def grounding_response(boxes):
    return dict(status='completed',id='grounding-test',output=[dict(type='message',content=[
        dict(type='output_text',text='<｜｜ref｜｜>洗衣机<｜｜/ref｜｜><｜｜box｜｜>'+boxes+'<｜｜/box｜｜>')])])


def test_grounding_scales_normalized_coordinates_without_guessing_units():
    box,label=gpt.decode_grounding_response(grounding_response('[[456,415,547,609]]'),1280,720)
    assert box==pytest.approx([583.68,298.8,700.16,438.48])
    assert label=='洗衣机'


@pytest.mark.parametrize('boxes', ['[]','[[1,2,3,4],[5,6,7,8]]','[[9,2,3,4]]',
    '[[1,2,1001,4]]','[[true,2,3,4]]','[[1,2,3]]','[[1,2,NaN,4]]','bad'])
def test_grounding_rejects_invalid_or_multiple_boxes(boxes):
    with pytest.raises(DetectionError):
        gpt.decode_grounding_response(grounding_response(boxes),1280,720)


def test_grounding_does_not_extract_random_numbers_or_truncated_responses():
    p=grounding_response('[[1,2,3,4]]');p['status']='incomplete'
    with pytest.raises(DetectionError):gpt.decode_grounding_response(p,1280,720)
    p=grounding_response('[[1,2,3,4]]');p['output'][0]['content'][0]['text']='可能在 [1,2,3,4]'
    with pytest.raises(DetectionError):gpt.decode_grounding_response(p,1280,720)
    p['output'][0]['content'][0]['text']='未找到'
    with pytest.raises(BoardUnavailable):gpt.decode_grounding_response(p,1280,720)


def test_front_corners_restore_full_image_coordinates():
    region,box=gpt.decode_front_corners(response({'corners':[[188,220],[668,199],[683,812],[206,799]]}),
        np.array([548,256]),np.array([187,224]),[583.68,298.8,700.16,438.48],(720,1280,3))
    assert region.polygon[1]==pytest.approx([672.916,300.576],abs=1e-4)
    assert box[2]<680 and region.polygon.shape==(4,2)


@pytest.mark.parametrize('corners', [None,[[0,0],[1000,0],[1000,1000],[0,1000]],
    [[400,400],[410,400],[410,410],[400,410]],
    [[188,220],[206,799],[683,812],[668,199]],
    [[188,220],[683,812],[668,199],[206,799]],
    [[188,220],[668,199],[683,812],[206,1001]]])
def test_invalid_or_unavailable_front_corners_are_rejected(corners):
    with pytest.raises((DetectionError,BoardUnavailable)):
        gpt.decode_front_corners(response({'corners':corners}),np.array([548,256]),np.array([187,224]),
            [583.68,298.8,700.16,438.48],(720,1280,3))


def test_refinement_failure_never_falls_back_to_whole_object():
    d=gpt.DeepSeekDetector('洗衣机',api_key='test-only',max_calls=2)
    replies=iter([grounding_response('[[250,200,650,800]]'),response({'corners':None,'door_bbox':None})])
    def post(*args,**kwargs):
        assert kwargs['json']['reasoning']=={'effort':'none'}
        assert kwargs['json']['max_output_tokens']==500
        return NS(status_code=200,json=lambda:next(replies))
    d.session=NS(post=post)
    with pytest.raises(BoardUnavailable):d.detect_region(np.zeros((480,640,3),np.uint8))
    assert d.last_metadata is None and 'front_refinement' in d.last_response
    assert d.calls==2


def test_front_corners_accept_one_fenced_json_block_with_explanation():
    p=response({'corners':[[188,220],[668,199],[683,812],[206,799]]})
    content=p['output'][0]['content'][0]
    content['text']='正面四角如下：\n```json\n'+content['text']+'\n```'
    _,box=gpt.decode_front_corners(p,np.array([548,256]),np.array([187,224]),
                                 [583.68,298.8,700.16,438.48],(720,1280,3))
    assert box[2]<680
    content['text']+='\n```json\n{}\n```'
    with pytest.raises(DetectionError):
        gpt.decode_front_corners(p,np.array([548,256]),np.array([187,224]),
                                [583.68,298.8,700.16,438.48],(720,1280,3))


def test_washer_door_is_required_and_maps_to_exclusion_ellipse():
    kwargs=dict(origin=np.array([548,256]),crop_size=np.array([187,224]),
                object_box=[583.68,298.8,700.16,438.48],image_shape=(720,1280,3),require_door=True)
    data={'corners':[[188,220],[668,199],[683,812],[206,799]],'door_bbox':[240,330,620,680]}
    region,_=gpt.decode_front_corners(response(data),**kwargs)
    assert region.region_scale==.90 and len(region.exclusion_polygons)==1
    ellipse=region.exclusion_polygons[0]
    assert ellipse.mean(0)==pytest.approx([548+430*.187,256+505*.224],abs=1e-3)
    for bad in [None,[0,0,1000,1000],[400,400,401,401],[600,600,300,300]]:
        with pytest.raises(DetectionError):gpt.decode_front_corners(response(dict(data,door_bbox=bad)),**kwargs)


@pytest.mark.parametrize('fenced',[False,True])
def test_thinking_json_grounding(fenced):
    payload=response({'bbox_2d':[394,486,465,806]})
    if fenced:
        content=payload['output'][0]['content'][0]
        content['text']='```json\n'+content['text']+'\n```'
    payload['output'].insert(0,{'type':'reasoning','content':[{'type':'reasoning_text','text':'not coordinates'}]})
    bbox,_=gpt.decode_grounding_response(payload,1280,720)
    assert bbox==pytest.approx([504.32,349.92,595.2,580.32])


@pytest.mark.parametrize('data',[{'bbox_2d':[10,10,5,5]}, {'bbox_2d':[True,0,500,500]},
    {'bbox_2d':[[1,2,3,4],[2,3,4,5]]},{'bbox_2d':[1,2,1001,4]},
    {'bbox_2d':[1,2,3,4],'unexpected':1}])
def test_thinking_json_rejects_invalid_boxes(data):
    with pytest.raises(DetectionError):gpt.decode_grounding_response(response(data),1280,720)


def test_thinking_json_not_found():
    with pytest.raises(BoardUnavailable):gpt.decode_grounding_response(response({'bbox_2d':None}),1280,720)


def test_washer_visible_face_can_touch_bottom_without_guessing_hidden_corners():
    payload=response({'corners':[[200,200],[800,200],[800,1000],[200,1000]],'door_bbox':[300,300,700,800]})
    region,bbox=gpt.decode_front_corners(payload,np.array([0,0]),np.array([640,480]),
                                       [128,96,512,480],(480,640,3),require_door=True)
    assert region.allow_bottom_crop and bbox[3]==479
    assert region.polygon[:,1].max()==479


def test_bottom_crop_exception_does_not_allow_side_or_top_crop():
    from board_approach.object_center import object_region
    object_region((480,640,3),[128,96,512,480],allow_bottom_crop=True)
    for box in ([0,96,512,480],[128,0,512,480],[128,96,640,480],[128,96,512,481]):
        with pytest.raises(BoardUnavailable):object_region((480,640,3),box,allow_bottom_crop=True)
    with pytest.raises(BoardUnavailable):object_region((480,640,3),[128,96,512,480])


@pytest.mark.parametrize('box',[[None],[],[1,2,3],[1,2,3,4,5],[True,2,3,4],[10,10,5,5]])
def test_bad_grounding_is_retryable_not_accepted(box):
    with pytest.raises(gpt.DetectorFormatError):
        gpt.decode_grounding_response(response({'bbox_2d':box}),1280,720)


def test_percent_scale_retries_full_image_before_front_crop():
    d=gpt.DeepSeekDetector('洗衣机',api_key='test-only',max_calls=3)
    replies=iter([response({'bbox_2d':[30,43,40,63]}),response({'bbox_2d':[300,432,400,638]}),
                  response({'corners':[[196,196],[646,196],[646,792],[196,792]],'door_bbox':[196,283,646,683]})])
    prompts=[]
    def post(*args,**kwargs):
        prompts.append(kwargs['json']['input'][0]['content'][0]['text'])
        return NS(status_code=200,json=lambda:next(replies))
    d.session=NS(post=post)
    d.detect_region(np.zeros((720,1280,3),np.uint8))
    assert 'bbox_2d' in prompts[0] and 'bbox_2d' in prompts[1] and 'door_bbox' in prompts[2]
    assert d.calls==3 and len(d.last_response['grounding_attempts'])==1
    assert d.last_response['crop_geometry']['origin'][0]>300


def test_small_true_top_left_target_is_not_multiplied_or_refined():
    d=gpt.DeepSeekDetector('洗衣机',api_key='test-only',max_calls=2)
    d.session=NS(post=lambda *a,**k:NS(status_code=200,json=lambda:response({'bbox_2d':[30,43,40,63]})))
    with pytest.raises(gpt.GroundingResolutionError):d.detect_region(np.zeros((720,1280,3),np.uint8))
    assert d.calls==2 and d.last_metadata is None
    assert len(d.last_response['grounding_attempts'])==2


def test_door_guides_repair_without_accepting_bad_face():
    d=gpt.DeepSeekDetector('洗衣机',api_key='test-only',max_calls=3)
    replies=iter([response({'bbox_2d':[385,432,442,612]}),
        response({'corners':[[0,222],[667,222],[667,778],[0,778]],'door_bbox':[400,300,850,700]}),
        response({'corners':[[400,220],[845,220],[845,800],[400,800]],'door_bbox':[400,300,850,700]})])
    prompts=[]
    def post(*a,**kw):
        prompts.append(kw['json']['input'][0]['content'][0]['text'])
        return NS(status_code=200,json=lambda:next(replies))
    d.session=NS(post=post)
    region=d.detect_region(np.zeros((720,1280,3),np.uint8))
    assert d.calls==3 and '舱门候选框' in prompts[-1]
    assert d.last_metadata['refinement']=='door_guided_front_corners'
    assert min(region.polygon[:,0])>510


@pytest.mark.parametrize('door',[None,[0,0,1,1],[400,700,850,300],[0,0,1000,1000]])
def test_unusable_door_cannot_anchor_repair(door):
    assert gpt.door_hint(response({'corners':None,'door_bbox':door}),np.array([470,272]),np.array([117,207]),[492.8,311.04,565.76,440.64]) is None


@pytest.mark.parametrize('decoder',[lambda p:gpt.decode_grounding_response(p,1280,720),
    lambda p:gpt.decode_front_corners(p,np.array([470,272]),np.array([117,207]),[492,311,566,441],(720,1280,3),True),
    lambda p:gpt.door_hint(p,np.array([470,272]),np.array([117,207]),[492,311,566,441])])
def test_incomplete_output_is_retryable_never_parsed(decoder):
    payload=response({'corners':[[0,0],[1000,0],[1000,1000],[0,1000]],'door_bbox':[200,200,800,800]})
    payload['status']='incomplete'
    with pytest.raises(gpt.DetectorFormatError,match='incomplete'):decoder(payload)


def test_door_repair_truncation_preserves_evidence_and_is_retryable():
    d=gpt.DeepSeekDetector('洗衣机',api_key='test-only',max_calls=3)
    truncated=response({'corners':None,'door_bbox':None});truncated['status']='incomplete'
    truncated['incomplete_details']={'reason':'max_output_tokens'}
    replies=iter([response({'bbox_2d':[385,432,442,612]}),
        response({'corners':[[0,222],[667,222],[667,778],[0,778]],'door_bbox':[400,300,850,700]}),truncated])
    d.session=NS(post=lambda *a,**k:NS(status_code=200,json=lambda:next(replies)))
    with pytest.raises(gpt.DetectorFormatError,match='incomplete'):d.detect_region(np.zeros((720,1280,3),np.uint8))
    assert d.last_metadata is None and d.calls==3
    assert d.last_response['door_guided_refinement']['incomplete_details']['reason']=='max_output_tokens'
