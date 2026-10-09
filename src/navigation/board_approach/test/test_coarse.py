import cv2
import numpy as np
import pytest
import board_approach.coarse as coarse
from board_approach.perception import BoardUnavailable, DetectionError, arrived


@pytest.fixture
def scene(monkeypatch):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    texture = cv2.aruco.CharucoBoard((7, 5), 1., .737, dictionary).generateImage((70, 50))
    polygon = np.float32([[0, 0], [70, 0], [70, 50], [0, 50]])
    monkeypatch.setattr(coarse, 'templates', lambda: [(texture, polygon)])
    image = np.full((240, 400, 3), 180, np.uint8)
    image[100:150, 200:270] = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    depth = np.full((240, 400), 2.)
    k = np.array([[640., 0, 200], [0, 640, 120], [0, 0, 1.]])
    return image, depth, k, texture


def test_appearance_uses_measured_depth_without_decoded_identity(scene):
    image, depth, k, _ = scene
    detector = coarse.AppearanceDetector()
    observation = detector.detect(image, depth, k)
    assert observation.source == 'appearance_depth'
    assert observation.ids == []
    assert observation.center[2] == pytest.approx(2., abs=.001)
    assert observation.score > .99
    # No old pixel ROI is assumed on the next image: full-frame matching follows a moved image.
    shifted = np.roll(image, 30, axis=1)
    moved = detector.detect(shifted, depth, k)
    assert moved.center[0]-observation.center[0] == pytest.approx(30*2/640, abs=.002)


def test_two_candidates_stop_instead_of_picking_one(scene):
    image, depth, k, texture = scene
    image[20:70, 30:100] = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    with pytest.raises(DetectionError, match='ambiguous'):
        coarse.AppearanceDetector().detect(image, depth, k)


def test_missing_target_and_invalid_depth_stop(scene):
    image, depth, k, _ = scene
    with pytest.raises(BoardUnavailable):
        coarse.AppearanceDetector().detect(image, np.zeros_like(depth), k)
    image[:] = 180
    with pytest.raises(BoardUnavailable):
        coarse.AppearanceDetector().detect(image, depth, k)


def test_appearance_alone_cannot_report_arrival():
    assert not arrived(dict(distance=1., lateral=0., heading_error=0., identity_verified=False), 1.)
