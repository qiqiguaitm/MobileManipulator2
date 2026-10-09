import json
from types import SimpleNamespace as NS

import cv2
import numpy as np

from board_approach.diagnostics import save_failure


def test_failure_keeps_original_frames_boxes_and_error(tmp_path):
    stage = tmp_path / 'stage_2'
    stage.mkdir()
    for i in range(3):
        cv2.imwrite(str(stage / f'rgb_{i}.png'), np.full((8, 8, 3), i, np.uint8))
        cv2.imwrite(str(stage / f'detection_{i}.png'), np.full((8, 8, 3), 20+i, np.uint8))
        np.save(stage / f'depth_{i}.npy', np.ones((8, 8)))
    (stage / 'observations.json').write_text('[{"distance":1.4}]')
    save_failure(tmp_path, stage, 'inconsistent centres', NS(data={}, received={}))
    report = json.loads((tmp_path / 'failure/failure.json').read_text())
    assert report['error'] == 'inconsistent centres'
    assert len(report['images']) == 6
    assert report['camera_image'] is None
    for source in stage.iterdir():
        assert (tmp_path / 'failure' / source.name).read_bytes() == source.read_bytes()


def test_failure_without_observation_saves_padded_rgb_and_timestamp(tmp_path):
    rgb = NS(encoding='rgb8', height=1, width=1, step=4, data=bytes([255, 0, 0, 99]),
             header=NS(stamp=NS(sec=123, nanosec=456)))
    save_failure(tmp_path, None, 'sensor stale', NS(data={'rgb': rgb}, received={'rgb': 0}))
    report = json.loads((tmp_path / 'failure/failure.json').read_text())
    assert report['camera_image']['timestamp_ns'] == 123000000456
    assert report['camera_image']['age_seconds'] > 0
    assert cv2.imread(str(tmp_path / 'failure/last_received_camera.png'))[0, 0].tolist() == [0, 0, 255]
