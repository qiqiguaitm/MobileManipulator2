from threading import Event
import time

import numpy as np

from auto_explore.archive import ObservationArchive


def test_slow_disk_is_bounded_and_submission_never_waits(tmp_path):
    entered, release = Event(), Event()
    saved = []

    def blocked_write(path, values):
        entered.set()
        assert release.wait(3.)
        saved.append(values['sequence'])

    archive = ObservationArchive(capacity=2, writer=blocked_write)
    try:
        assert archive.submit(tmp_path/'first.npz', sequence=1)
        assert entered.wait(1.)
        start = time.monotonic()
        assert archive.submit(tmp_path/'second.npz', sequence=2)
        assert not archive.submit(tmp_path/'third.npz', sequence=3)
        assert time.monotonic()-start < .2
        assert len(archive.pending) == 2
        assert archive.error == 'archive_backlog_full'
    finally:
        release.set()
        archive.close()
    assert saved == [1, 2]
    assert archive.written == 2
    assert archive.error == 'archive_backlog_full'


def test_disk_failure_is_reported_and_stays_failed(tmp_path):
    def fail(*args):
        raise OSError('disk full')

    archive = ObservationArchive(writer=fail)
    archive.submit(tmp_path/'frame.npz', points=np.zeros((1, 3)))
    archive.close()
    assert archive.poll() == 'archive_write:disk full'
    assert not archive.submit(tmp_path/'next.npz', points=np.zeros((1, 3)))
    assert archive.written == 0


def test_saved_points_and_anchor_replacement_are_complete(tmp_path):
    archive = ObservationArchive()
    points = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    archive.submit(tmp_path/'observation_000000.npz', stamp=1., points=points,
                   pose=np.eye(4), sensor_origin=np.array([.2, 0., .1]))
    archive.submit(tmp_path/'anchors.npz', times=np.array([1.]))
    archive.submit(tmp_path/'anchors.npz', times=np.array([1., 2.]))
    archive.close()
    assert archive.error is None and archive.written == 3
    with np.load(tmp_path/'observation_000000.npz') as data:
        np.testing.assert_array_equal(data['points'], points)
        np.testing.assert_array_equal(data['pose'], np.eye(4))
    with np.load(tmp_path/'anchors.npz') as data:
        np.testing.assert_array_equal(data['times'], [1., 2.])
    assert not list(tmp_path.glob('*.tmp'))
