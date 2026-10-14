import json
from types import SimpleNamespace as S
import pytest
from robot_tasks.ledger import Ledger
from robot_tasks.vlm import decode


def obj(name, x):
    return S(name=name, instance_id="", position=S(point=S(x=x, y=0, z=0.3)))


def test_retry_dedup_and_refresh():
    l = Ledger(0.12, 2)
    o, r = l.select([obj("杯子", 1)])
    l.finish(r, False)
    _, same = l.select([obj("杯子", 1.02)])
    assert r is same
    l.finish(r, False)
    assert l.select([obj("杯子", 1.03)]) == (None, None)
    _, other = l.select([obj("杯子", 2)])
    l.finish(other, True)
    assert l.counts() == (1, 1)
    assert l.select([]) == (None, None)


def payload(text):
    return {
        "status": "completed",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": text}]}
        ],
    }


def test_detect_multi_and_empty():
    data = {
        "objects": [
            {"name": "杯子", "bbox_2d": [100, 200, 500, 600]},
            {"name": "杯子", "bbox_2d": [600, 200, 900, 600]},
        ]
    }
    r = decode(payload(json.dumps(data)), 1280, 720, ["杯子"])
    assert r[0]["bbox"] == [128, 144, 640, 432]
    assert len(r) == 2
    assert decode(payload('{"objects":[]}'), 1280, 720, ["杯子"]) == []


@pytest.mark.parametrize(
    "text",
    [
        "null",
        "{}",
        '{"objects":null}',
        '{"objects":[{"name":"unknown","bbox_2d":[0,0,200,200]}]}',
        '{"objects":[{"name":"杯子","bbox_2d":[800,0,200,200]}]}',
    ],
)
def test_error_is_not_empty(text):
    with pytest.raises(ValueError):
        decode(payload(text), 1280, 720, ["杯子"])
