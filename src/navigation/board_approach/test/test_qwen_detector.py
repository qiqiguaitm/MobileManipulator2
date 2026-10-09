import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
from board_approach.qwen_detector import QwenDetector, QwenTransport, ENDPOINTS
from board_approach.gpt_detector import DetectorFormatError


def test_qwen_full_grounding_pipeline_and_key_isolation():
    detector = QwenDetector('洗衣机', api_key='test-qwen-key')
    assert detector.session.session.trust_env is False
    replies = iter([
        {'bbox_2d': [250,200,650,800]},
        {'corners': [[188,188],[812,188],[812,812],[188,812]],
         'door_bbox': [300,350,700,700]},
    ])
    calls=[]
    def post(url, **kwargs):
        assert url == ENDPOINTS['beijing']
        assert kwargs['headers']['Authorization'] == 'Bearer test-qwen-key'
        assert kwargs['allow_redirects'] is False
        body=kwargs['json']
        assert body['model']=='qwen3-vl-plus'
        assert body['enable_thinking'] is False
        assert 'input' not in body and 'reasoning' not in body
        content=body['messages'][0]['content']
        assert content[0]['type']=='text'
        assert content[1]['image_url']['url'].startswith('data:image/jpeg;base64,')
        calls.append(body)
        payload={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(next(replies))}}]}
        return NS(status_code=200,json=lambda:payload)
    detector.session=QwenTransport(NS(post=post))
    region=detector.detect_region(np.zeros((480,640,3),np.uint8))
    assert len(calls)==2 and detector.calls==2
    assert len(region.exclusion_polygons)==1


def test_truncated_qwen_reply_is_retryable():
    payload={'choices':[{'finish_reason':'length','message':{'content':'{"bbox_2d":['}}]}
    session=NS(post=lambda *a,**k:NS(status_code=200,json=lambda:payload))
    transport=QwenTransport(session)
    response=transport.post(ENDPOINTS['beijing'],json={'model':'qwen3-vl-plus','input':[]})
    with pytest.raises(DetectorFormatError, match='incomplete'):
        QwenDetector('洗衣机',api_key='test').decode_payload(response.json(),640,480)


def test_http_error_is_preserved():
    response=NS(status_code=401)
    transport=QwenTransport(NS(post=lambda *a,**k:response))
    assert transport.post(ENDPOINTS['beijing'],json={'model':'test','input':[]}) is response
