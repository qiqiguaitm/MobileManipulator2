"""Bounded, ordered NPZ storage outside the ROS callback executor."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np


def write_npz(path, values):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        # Passing a handle prevents numpy from appending another .npz suffix.
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **values)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ObservationArchive:
    """Never wait for disk in submit/poll, and never silently drop a record.

    Saturation or a write failure is sticky: the mapper must become not-ready.
    Pending records are immutable after submission and are drained on close.
    """
    def __init__(self, capacity=32, writer=write_npz):
        if capacity < 1:
            raise ValueError('archive capacity must be positive')
        self.capacity = capacity
        self.writer = writer
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = deque()
        self.error = None
        self.written = 0
        self.max_write_seconds = 0.

    def _write(self, path, values):
        started = time.monotonic()
        self.writer(path, values)
        return time.monotonic() - started

    def poll(self):
        while self.pending and self.pending[0].done():
            future = self.pending.popleft()
            try:
                self.max_write_seconds = max(self.max_write_seconds, future.result())
                self.written += 1
            except Exception as exc:
                self.error = self.error or f'archive_write:{exc}'
        return self.error

    def submit(self, path, **values):
        self.poll()
        if self.error:
            return False
        if len(self.pending) >= self.capacity:
            self.error = 'archive_backlog_full'
            return False
        self.pending.append(self.executor.submit(self._write, path, values))
        return True

    def close(self):
        self.executor.shutdown(wait=True)
        self.poll()

