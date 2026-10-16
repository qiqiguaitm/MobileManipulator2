"""Catalog checks without requiring ROS, hardware, or the robot data disk."""
from pathlib import Path
import sys

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from prepare_test import parse_args


def catalog():
    return yaml.safe_load((SCRIPTS / 'test_set.yaml').read_text())


def test_membership_and_relative_paths():
    data = catalog()
    ids = [scene['id'] for scene in data['scenes']]
    assert len(ids) == len(set(ids))
    for scene in data['scenes']:
        assert scene['enabled'] == (scene['inclusion_decision'] == 'accepted')
        for key in ('scene_dir', 'original_scene_dir', 'map_config'):
            if key in scene:
                path = Path(scene[key])
                assert not path.is_absolute() and '..' not in path.parts


def test_existing_bin_location_unchanged():
    data = catalog()
    scene = next(s for s in data['scenes'] if s['id'] == 'bin_approach_20261015_01')
    assert Path(data['data_root']) / scene['scene_dir'] == Path(
        '/data/junbin.zhang/navigation_scenarios/scenes/bin_approach_20261015_01')


def test_fire_is_accepted_failure_regression_not_success_baseline():
    data = catalog()
    fire = next(s for s in data['scenes'] if s['id'] == 'fire_extinguisher_approach_20261016_01')
    assert fire['enabled'] is True
    assert fire['inclusion_decision'] == 'accepted'
    assert 'planning_failure_regression' in fire['tags']
    assert fire['data_quality']['verification_status'] == 'passed_data_checks'
    assert fire['data_quality']['scan_tf_coverage'] == 1.0
    assert str(Path(data['data_root']) / fire['scene_dir']).startswith('/data/workspace/wangyibo/')


def test_target_is_explicit_and_whitespace_trimmed():
    options = parse_args(['scene', 'output', '--target', ' 灭火器 ', '--snapshot-offset', '1.5'])
    assert options.target == '灭火器'
    assert options.snapshot_offset == 1.5


def test_default_target_backward_compatible():
    assert parse_args(['scene', 'output']).target == '垃圾桶'


@pytest.mark.parametrize('options', [
    ['--target', ' '], ['--snapshot-offset', '-1'],
    ['--snapshot-offset', 'nan'], ['--snapshot-offset', 'inf'],
])
def test_bad_options_rejected_before_loading_ros_or_calling_api(options):
    with pytest.raises(SystemExit) as error:
        parse_args(['scene', 'output', *options])
    assert error.value.code == 2
