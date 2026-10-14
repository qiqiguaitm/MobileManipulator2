"""An expiring NC motion permit. Explicit idle never authorizes movement."""


class MotionLease:
    def __init__(self, timeout=0.5):
        self.timeout = timeout
        self.requested = False
        self.received = None

    def update(self, requested, now):
        self.requested = bool(requested)
        self.received = now

    def expired(self, now):
        return self.requested and (
            self.received is None or not 0 <= now - self.received <= self.timeout
        )

    def permits(self, now):
        return self.requested and not self.expired(now)
