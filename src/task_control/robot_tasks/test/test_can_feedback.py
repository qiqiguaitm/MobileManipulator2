import socket, struct
import pytest
from robot_tasks.can_feedback import CanFeedbackGate, CanStatusReceiver, DrainedSocket

NORMAL = [0, 1, 10, 80, 0, 0, 0, 0]


def ready(g):
    assert g.update(0, NORMAL, 0) == (False, None)
    assert g.update(0.2, NORMAL, 0.2) == (False, None)
    assert g.update(0.4, NORMAL, 0.4) == (True, None)


def test_short_outage_stops_then_needs_three_distinct_frames():
    g = CanFeedbackGate()
    ready(g)
    assert g.update(0.801, NORMAL, 0.4) == (False, None)
    assert g.update(0.81, NORMAL, 0.81) == (False, None)
    for t in (0.82, 0.83, 0.84):
        assert g.update(t, NORMAL, 0.81) == (False, None)
    assert g.update(1.01, NORMAL, 1.01) == (False, None)
    assert g.update(1.21, NORMAL, 1.21) == (True, None)
    assert g.pause_since is None


def test_intermittent_recovery_resets_consecutive_count():
    g = CanFeedbackGate()
    ready(g)
    g.update(0.81, NORMAL, 0.4)
    g.update(0.9, NORMAL, 0.9)
    assert g.update(1.25, NORMAL, 1.25) == (False, None)
    assert g.count == 1
    assert g.update(1.45, NORMAL, 1.45) == (False, None)
    assert g.update(1.65, NORMAL, 1.65) == (True, None)


def test_sustained_outage_latches_even_if_frames_return():
    g = CanFeedbackGate()
    ready(g)
    g.update(0.81, NORMAL, 0.4)
    enabled, error = g.update(2.82, NORMAL, 0.4)
    assert not enabled and "within 2.0s" in error
    assert g.update(3, NORMAL, 3) == (False, error)
    g.reset()
    ready(g)


@pytest.mark.parametrize("index,value", [(0, 1), (1, 3), (4, 1), (5, 128)])
def test_fault_or_mode_change_is_immediate_and_latched(index, value):
    g = CanFeedbackGate()
    ready(g)
    s = NORMAL.copy()
    s[index] = value
    enabled, error = g.update(0.6, s, 0.6)
    assert not enabled and "fault/control mode" in error
    assert g.update(0.8, NORMAL, 0.8) == (False, error)


def test_delayed_or_missing_feedback_cannot_resume():
    g = CanFeedbackGate()
    for now, stamp in [(1, 0), (1.2, None), (1.4, float("nan")), (1.6, 2)]:
        assert g.update(now, NORMAL, stamp) == (False, None)
    assert g.update(1.8, [0, 1], 1.8) == (False, None)


class FakeSocket:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.options = []
        self.closed = False

    def setsockopt(self, *args):
        self.options.append(args)

    def recvmsg(self, *args):
        try:
            return next(self.frames)
        except StopIteration:
            raise BlockingIOError

    def close(self):
        self.closed = True


def frame(state, sec, nsec=0):
    return (
        struct.pack("=IB3x8s", 0x211, 8, bytes(state)),
        [(socket.SOL_SOCKET, 35, struct.pack("=qq", sec, nsec))],
        0,
        None,
    )


def test_receiver_uses_kernel_age_and_keeps_transient_fault(monkeypatch):
    monkeypatch.setattr("robot_tasks.can_feedback.time.monotonic", lambda: 20.0)
    monkeypatch.setattr("robot_tasks.can_feedback.time.time", lambda: 100.0)
    bad = NORMAL.copy()
    bad[4] = 1
    sock = FakeSocket([frame(bad, 99), frame(NORMAL, 99, 500000000)])
    r = CanStatusReceiver(sock)
    r.poll()
    assert r.stamp == 19.5 and r.state == NORMAL and r.fault_in_batch == bad
    assert r.recent[-1]["reader_delay_s"] == 0.5 and r.recent[-1]["kernel_gap_s"] == 0.5
    proxy = DrainedSocket(sock)
    with pytest.raises(BlockingIOError):
        proxy.recv(16)
    proxy.close()
    assert sock.closed
