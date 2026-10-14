from robot_tasks.motion_lease import MotionLease


def test_missing_and_expired_permits_stop_motion():
    lease = MotionLease()
    assert not lease.permits(1)
    lease.update(True, 1)
    assert lease.permits(1.4)
    assert lease.expired(1.51) and not lease.permits(1.51)
    assert not lease.permits(0.9)


def test_idle_revokes_permit_and_next_task_can_reuse_it():
    lease = MotionLease()
    lease.update(True, 1)
    lease.update(False, 1.1)
    assert not lease.permits(1.1) and not lease.expired(2)
    lease.update(True, 3)
    assert lease.permits(3.1)
