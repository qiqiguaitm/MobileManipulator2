import cv2
import numpy as np
import pytest
from board_approach.perception import (DetectionError, Observation, arrived,
                                      detect_board, pose_matrix, target_geometry)


def scene(duplicate=False):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    board = cv2.aruco.CharucoBoard((7, 5), 1., .737, dictionary)
    texture = board.generateImage((350, 250))
    image = np.full((720, 1280, 3), 210, np.uint8)
    image[300:550, 450:800] = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    if duplicate:
        image[40:290, 50:400] = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    k = np.array([[640., 0, 640], [0, 640, 360], [0, 0, 1.]])
    depth = np.full((720, 1280), 2.)
    return image, depth, k


def test_full_frame_detection_and_depth_plane():
    image, depth, k = scene()
    observation = detect_board(image, depth, k)
    assert {8, 15} <= set(observation.ids)
    assert observation.center[2] == pytest.approx(2., abs=.005)
    assert observation.normal[2] < -.99
    assert observation.rmse < .001


def test_slanted_plane_is_recovered():
    image, depth, k = scene()
    yy, xx = np.mgrid[:720, :1280]
    normal = np.array([.20, .05, -1.])
    normal /= np.linalg.norm(normal)
    ray_dot = normal[0]*(xx-640)/640 + normal[1]*(yy-360)/640 + normal[2]
    depth = -2./ray_dot
    observation = detect_board(image, depth, k)
    assert observation.normal @ normal > .999


def test_two_identical_boards_are_rejected():
    with pytest.raises(DetectionError, match='Multiple boards'):
        detect_board(*scene(duplicate=True))


def test_no_board_is_rejected():
    image, depth, k = scene()
    image[:] = 210
    with pytest.raises(DetectionError, match='Need at least'):
        detect_board(image, depth, k)


def test_no_depth_is_rejected():
    image, depth, k = scene()
    depth[:] = 0
    with pytest.raises(DetectionError, match='depth'):
        detect_board(image, depth, k)


def test_nonplanar_depth_is_rejected():
    image, depth, k = scene()
    depth[:] = np.random.default_rng(1).uniform(1, 3, depth.shape)
    with pytest.raises(DetectionError, match='consistent plane'):
        detect_board(image, depth, k)


def test_geometry_uses_base_frame_and_plane_normal():
    obs = Observation(np.array([.1, .2, 2.]), np.array([0., 0., -1.]),
                      [8, 15], np.empty((0, 2)), 0., 1.)
    base_camera = np.array([[0., 0., 1., -.2], [-1., 0., 0., 0.],
                            [0., -1., 0., .6], [0., 0., 0., 1.]])
    target = target_geometry(obs, base_camera, np.eye(4), 1.)
    assert target['distance'] == pytest.approx(1.8)
    assert target['goal_map'] == pytest.approx([.8, -.1, 0.])
    assert target['goal_yaw'] == pytest.approx(0.)
    assert not arrived(target, 1.)


def test_arrival_requires_distance_lateral_and_heading():
    good = dict(distance=1.04, lateral=.02, heading_error=.01, identity_verified=True)
    assert arrived(good, 1.)
    for key, value in [('distance', 1.2), ('lateral', .2), ('heading_error', .2)]:
        assert not arrived(dict(good, **{key: value}), 1.)


def test_distributed_ids_can_recover_an_unreadable_center_marker():
    from board_approach.perception import sufficient_layout
    assert sufficient_layout([i for i in range(17) if i != 8])
    assert not sufficient_layout(list(range(8)))
    assert not sufficient_layout([0, 1, 2])


def test_center_marker_occlusion_still_fits_whole_board():
    image, depth, k = scene()
    # ID 8 is the centre cell (x=3, y=2) of this synthetic 7x5 board.
    image[400:450, 600:650] = 220
    observation = detect_board(image, depth, k)
    assert 8 not in observation.ids
    assert len(observation.ids) >= 8
    assert observation.center[2] == pytest.approx(2., abs=.005)


def test_object_arrival_requires_explicit_opt_out_and_geometry():
    target = dict(distance=1., lateral=0., heading_error=0., identity_verified=False,
                  detection_source='object_depth')
    assert not arrived(target, 1.)
    assert arrived(target, 1., require_aruco=False)
    assert not arrived(dict(target, detection_source='unknown'), 1., require_aruco=False)
    for key, value in [('distance', 1.2), ('lateral', .2), ('heading_error', .2)]:
        assert not arrived(dict(target, **{key:value}), 1., require_aruco=False)
    assert not arrived(dict(distance=1., lateral=0., heading_error=0.), 1.)
