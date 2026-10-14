"""Timestamped CAN status reception and bounded stop/recovery policy."""

import math
import socket
import struct
import time
from collections import deque


class CanFeedbackGate:
    def __init__(
        self, stale_after=0.4, recovery_frames=3, recovery_gap=0.3, abort_after=2.0
    ):
        self.stale_after = stale_after
        self.recovery_frames = recovery_frames
        self.recovery_gap = recovery_gap
        self.abort_after = abort_after
        self.reset()

    def reset(self):
        self.paused = True
        self.pause_since = None
        self.count = 0
        self.last_stamp = None
        self.failure = None

    def update(self, now, state, stamp):
        if self.failure:
            return False, self.failure
        fresh = (
            stamp is not None
            and math.isfinite(stamp)
            and 0 <= now - stamp <= self.stale_after
        )
        complete = state is not None and len(state) >= 6
        # A received emergency/error/control-mode change is never retried away.
        if complete and (state[0] != 0 or state[4] or state[5] or state[1] != 1):
            self.failure = f"CAN chassis fault/control mode abnormal: raw={state}"
            self.paused = True
            return False, self.failure
        if self.pause_since is not None and now - self.pause_since >= self.abort_after:
            self.failure = (
                f"CAN feedback did not recover within {self.abort_after:.1f}s"
            )
            self.paused = True
            return False, self.failure
        if not fresh or not complete:
            if self.pause_since is None:
                self.pause_since = now
            self.paused = True
            self.count = 0
            return False, None
        if stamp != self.last_stamp:
            consecutive = (
                self.last_stamp is not None
                and 0 < stamp - self.last_stamp <= self.recovery_gap
            )
            self.count = self.count + 1 if consecutive else 1
            self.last_stamp = stamp
        if self.paused:
            if self.pause_since is None:
                self.pause_since = now
            if self.count >= self.recovery_frames:
                self.paused = False
                self.pause_since = None
        return not self.paused, None


class DrainedSocket:
    """Legacy guard must not consume status frames outside the timestamped reader."""

    def __init__(self, original):
        self.original = original

    def recv(self, size):
        raise BlockingIOError()

    def close(self):
        self.original.close()


class CanStatusReceiver:
    def __init__(self, sock):
        self.sock = sock
        self.state = None
        self.stamp = None
        self.sequence = 0
        self.recent = deque(maxlen=300)
        self.pending = deque(maxlen=3000)
        self.last_kernel = None
        self.last_overflow = 0
        # Linux robot02: standard data status frame only, excluding EFF/RTR.
        sock.setsockopt(
            socket.SOL_CAN_RAW,
            socket.CAN_RAW_FILTER,
            struct.pack("=II", 0x211, 0xC00007FF),
        )
        sock.setsockopt(socket.SOL_SOCKET, 35, 1)  # SO_TIMESTAMPNS
        sock.setsockopt(socket.SOL_SOCKET, 40, 1)  # SO_RXQ_OVFL
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)

    def poll(self):
        self.fault_in_batch = None
        for _ in range(256):
            try:
                raw, anc, flags, _ = self.sock.recvmsg(16, 128)
            except BlockingIOError:
                break
            now = time.monotonic()
            wall = time.time()
            kernel = None
            overflow = self.last_overflow
            for level, kind, value in anc:
                if level == socket.SOL_SOCKET and kind == 35:
                    sec, nsec = struct.unpack("=qq", value[:16])
                    kernel = sec + nsec / 1e9
                elif level == socket.SOL_SOCKET and kind == 40:
                    overflow = struct.unpack("=I", value[:4])[0]
            if len(raw) != 16:
                continue
            ident, size, payload = struct.unpack("=IB3x8s", raw)
            if ident != 0x211:
                continue
            self.sequence += 1
            self.state = list(payload[:size])
            if len(self.state) >= 6 and (
                self.state[0] != 0
                or self.state[4]
                or self.state[5]
                or self.state[1] != 1
            ):
                self.fault_in_batch = self.state.copy()
            delay = wall - kernel if kernel is not None else None
            # Missing/future timestamps fail closed; delayed queued data is not made fresh on read.
            self.stamp = now - delay if delay is not None and delay >= 0 else None
            record = dict(
                sequence=self.sequence,
                read_monotonic=now,
                kernel_wall=kernel,
                kernel_gap_s=(
                    kernel - self.last_kernel
                    if kernel is not None and self.last_kernel is not None
                    else None
                ),
                reader_delay_s=delay,
                rx_queue_drops=overflow,
                raw=self.state,
                timestamp_missing=kernel is None,
                truncated=bool(flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)),
            )
            if record["truncated"]:
                self.stamp = None
            self.last_kernel = kernel
            self.last_overflow = overflow
            self.recent.append(record)
            self.pending.append(record)
