"""Per-task target identity and bounded retries across refreshed observations."""

import math
from dataclasses import dataclass


@dataclass
class Record:
    name: str
    xyz: tuple
    attempts: int = 0
    state: str = "pending"
    instance_id: str = ""


class Ledger:
    def __init__(self, radius=0.15, attempts=2):
        self.radius, self.max_attempts = radius, attempts
        self.records = []

    def match(self, name, xyz):
        matches = [(math.dist(r.xyz, xyz), r) for r in self.records if r.name == name]
        return (
            min(matches, key=lambda v: v[0])[1]
            if matches and min(v[0] for v in matches) <= self.radius
            else None
        )

    def select(self, objects):
        # Caller supplies a fresh scene. Never reuse a target absent from that scene.
        for obj in objects:
            p = obj.position.point
            xyz = (p.x, p.y, p.z)
            r = self.match(obj.name, xyz)
            if r is None:
                r = Record(obj.name, xyz, instance_id=f"target_{len(self.records)+1}")
                self.records.append(r)
            if r.state == "pending":
                r.xyz = xyz
                obj.instance_id = r.instance_id
                return obj, r
        return None, None

    def finish(self, record, success):
        record.attempts += 1
        if success:
            record.state = "completed"
        elif record.attempts >= self.max_attempts:
            record.state = "failed"

    def counts(self):
        return (
            sum(r.state == "completed" for r in self.records),
            sum(r.state == "failed" for r in self.records),
        )
