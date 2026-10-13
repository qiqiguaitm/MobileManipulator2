"""Startup failures must identify the gate without dispatching motion."""
import time
from types import SimpleNamespace as NS

import pytest
import rclpy

from auto_explore.explorer import Explorer


@pytest.fixture
def node(tmp_path):
    rclpy.init()
    n=Explorer(NS(output=str(tmp_path),execute=False,bounds=[-5,-5,5,5],
                        max_seconds=0,max_distance=0))
    n.navigation_ready=lambda:True
    n.input_problem=lambda:None
    n.pose3=lambda:(0.,0.,0.)
    n.mapper_ready=lambda:True
    yield n
    n.destroy_node()
    rclpy.shutdown()


def test_estop_is_reported_before_timeout_and_never_starts_exploration(node):
    node.args.execute=True
    node.can_state=[1,1,10,70,0,152,0,0]
    node.can_time=time.monotonic()
    node.coordinate()
    assert not node.enabled and node.process is None and not node.done
    assert 'vehicle=0x01' in node.startup_blockers['chassis']
    events=(node.out/'events.jsonl').read_text()
    assert 'startup_wait' in events and '01 01 0a 46 00 98 00 00' in events
    node.started=time.monotonic()-91
    node.can_time=time.monotonic()
    node.coordinate()
    assert node.done and node.final_reason=='startup_inputs_not_ready: chassis'
    assert node.save()['startup_blockers']==node.startup_blockers
    assert node.process is None and not node.enabled


def test_startup_reports_all_unready_inputs(node):
    node.navigation_ready=lambda:False
    node.input_problem=lambda:dict(source='scan',check='valid_points',value=4,minimum=20)
    node.pose3=lambda:(_ for _ in ()).throw(RuntimeError('stale_map_transform'))
    node.watchdog_ok=False
    node.mapper_ready=lambda:False
    node.coordinate()
    assert set(node.startup_blockers)=={'inputs','pose','navigation','driver_watchdog','mapper'}
    assert node.startup_blockers['pose']=='stale_map_transform'
    assert not node.enabled and node.process is None


def test_recovered_preview_clears_old_blockers_without_motion(node):
    node.startup_blockers={'chassis':'old fault'}
    node.coordinate()
    assert node.done and node.status=='preview'
    assert node.startup_blockers=={} and node.process is None and not node.enabled
