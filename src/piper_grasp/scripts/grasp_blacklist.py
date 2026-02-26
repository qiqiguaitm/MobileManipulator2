#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grasp Blacklist System - Dual-Layer Implementation

Purpose: Avoid repeatedly trying to grasp unreachable objects
Architecture: Two-layer system (Spatial + Object ID) for robust tracking
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple
import json


class SpatialBlacklistEntry:
    """Spatial blacklist entry - precise location matching"""

    def __init__(self,
                 center: List[float],
                 radius: float = 30.0,
                 reason: str = "TARGET_LIMIT",
                 category: Optional[str] = None):
        """
        Args:
            center: Object center in base frame [x, y, z] (mm)
            radius: Matching radius in mm (default 30mm for precision)
            reason: Failure reason ("TARGET_LIMIT", etc.)
            category: Optional category for debugging
        """
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.reason = reason
        self.category = category
        self.timestamp = time.time()
        self.attempts = 1

    def matches(self, center: List[float]) -> bool:
        """Check if position matches (distance-based)"""
        distance = np.linalg.norm(np.array(center) - self.center)
        return distance < self.radius

    def is_expired(self, max_age: float) -> bool:
        """Check if entry is expired"""
        return (time.time() - self.timestamp) > max_age

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'type': 'spatial',
            'center': self.center.tolist(),
            'radius': self.radius,
            'reason': self.reason,
            'category': self.category,
            'timestamp': self.timestamp,
            'attempts': self.attempts
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SpatialBlacklistEntry':
        """Create from dictionary"""
        entry = cls(
            center=data['center'],
            radius=data.get('radius', 30.0),
            reason=data.get('reason', 'TARGET_LIMIT'),
            category=data.get('category')
        )
        entry.timestamp = data.get('timestamp', time.time())
        entry.attempts = data.get('attempts', 1)
        return entry


class ObjectBlacklistEntry:
    """Object ID blacklist entry - tracks specific objects with larger tolerance"""

    def __init__(self,
                 object_id: str,
                 category: str,
                 initial_center: List[float],
                 radius: float = 60.0,
                 reason: str = "TARGET_LIMIT"):
        """
        Args:
            object_id: Unique object identifier
            category: Object category
            initial_center: Initial center position [x, y, z] (mm)
            radius: Matching radius in mm (default 60mm, allows movement)
            reason: Failure reason
        """
        self.object_id = object_id
        self.category = category
        self.current_center = np.array(initial_center, dtype=float)
        self.radius = radius
        self.reason = reason
        self.timestamp = time.time()
        self.attempts = 1
        self.positions = [self.current_center.copy()]  # Track movement history

    def matches(self, object_id: Optional[str], category: str, center: List[float]) -> bool:
        """Check if object matches (ID + distance-based)"""
        # ID must match (if provided)
        if object_id and object_id != self.object_id:
            return False

        # Category must match
        if category != self.category:
            return False

        # Position must be within radius (allows object movement)
        distance = np.linalg.norm(np.array(center) - self.current_center)
        return distance < self.radius

    def update_position(self, center: List[float]) -> None:
        """Update current position (object moved)"""
        self.current_center = np.array(center, dtype=float)
        self.positions.append(self.current_center.copy())
        self.timestamp = time.time()
        self.attempts += 1

    def is_expired(self, max_age: float) -> bool:
        """Check if entry is expired"""
        return (time.time() - self.timestamp) > max_age

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'type': 'object',
            'object_id': self.object_id,
            'category': self.category,
            'current_center': self.current_center.tolist(),
            'radius': self.radius,
            'reason': self.reason,
            'timestamp': self.timestamp,
            'attempts': self.attempts,
            'positions': [p.tolist() for p in self.positions]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ObjectBlacklistEntry':
        """Create from dictionary"""
        entry = cls(
            object_id=data['object_id'],
            category=data['category'],
            initial_center=data['current_center'],
            radius=data.get('radius', 60.0),
            reason=data.get('reason', 'TARGET_LIMIT')
        )
        entry.timestamp = data.get('timestamp', time.time())
        entry.attempts = data.get('attempts', 1)
        entry.positions = [np.array(p) for p in data.get('positions', [])]
        return entry


class DualLayerBlacklist:
    """
    Dual-Layer Blacklist System

    Layer 1: Spatial - Precise location matching (30mm radius)
    Layer 2: Object ID - Track specific objects (60mm radius, allows movement)

    Design Philosophy (Linus-style):
    - Two simple mechanisms, not one complex one
    - Spatial for precision, Object ID for persistence
    - Data structure drives the logic, not clever algorithms
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Configuration dict with structure:
                {
                    'spatial': {'radius': 30, 'max_age': 600, 'max_entries': 100},
                    'object': {'radius': 60, 'max_age': 300, 'max_entries': 50, 'require_id': True},
                    'persist_file': '/tmp/piper_grasp_blacklist.json'
                }
        """
        config = config or {}

        # Spatial blacklist config
        spatial_cfg = config.get('spatial', {})
        self.spatial_radius = spatial_cfg.get('radius', 30.0)
        self.spatial_max_age = spatial_cfg.get('max_age', 600.0)  # 10 min
        self.spatial_max_entries = spatial_cfg.get('max_entries', 100)
        self.spatial_entries: List[SpatialBlacklistEntry] = []

        # Object blacklist config
        object_cfg = config.get('object', {})
        self.object_radius = object_cfg.get('radius', 60.0)
        self.object_max_age = object_cfg.get('max_age', 300.0)  # 5 min
        self.object_max_entries = object_cfg.get('max_entries', 50)
        self.object_require_id = object_cfg.get('require_id', True)
        self.object_entries: List[ObjectBlacklistEntry] = []

        # Persistence
        self.persist_file = config.get('persist_file')

        # Statistics
        self.stats = {
            'spatial': {'added': 0, 'matched': 0, 'expired': 0},
            'object': {'added': 0, 'matched': 0, 'expired': 0}
        }

        # Load from file if exists
        if self.persist_file:
            self.load()

    def add(self,
            center: List[float],
            reason: str = "TARGET_LIMIT",
            object_id: Optional[str] = None,
            category: Optional[str] = None) -> None:
        """
        Add object to blacklist (both layers)

        Args:
            center: Object center in base frame [x,y,z] (mm)
            reason: Failure reason
            object_id: Object ID (if available)
            category: Object category (if available)
        """
        # Layer 1: Always add to spatial blacklist (precise location)
        self._add_spatial(center, reason, category)

        # Layer 2: Add to object blacklist if ID provided
        if object_id and category:
            self._add_object(object_id, category, center, reason)
        elif not self.object_require_id and category:
            # Fallback: use category + position as pseudo-ID
            pseudo_id = f"{category}_{int(center[0])}_{int(center[1])}"
            self._add_object(pseudo_id, category, center, reason)

        # Persist if enabled
        if self.persist_file:
            self.save()

    def _add_spatial(self, center: List[float], reason: str, category: Optional[str]) -> None:
        """Add to spatial blacklist"""
        # Check if position already exists (update attempts)
        for entry in self.spatial_entries:
            if entry.matches(center):
                entry.attempts += 1
                entry.timestamp = time.time()
                print(f"[Blacklist:Spatial] Updated at {center}, attempts={entry.attempts}")
                return

        # Add new entry
        entry = SpatialBlacklistEntry(center, self.spatial_radius, reason, category)
        self.spatial_entries.append(entry)
        self.stats['spatial']['added'] += 1
        print(f"[Blacklist:Spatial] Added {category or '?'} at {center}, reason={reason}")

        # Enforce max entries (remove oldest)
        if len(self.spatial_entries) > self.spatial_max_entries:
            removed = self.spatial_entries.pop(0)
            print(f"[Blacklist:Spatial] Removed oldest entry")

    def _add_object(self, object_id: str, category: str, center: List[float], reason: str) -> None:
        """Add to object blacklist"""
        # Check if object ID already exists (update position)
        for entry in self.object_entries:
            if entry.object_id == object_id:
                entry.update_position(center)
                print(f"[Blacklist:Object] Updated {category}#{object_id} at {center}, attempts={entry.attempts}")
                return

        # Add new entry
        entry = ObjectBlacklistEntry(object_id, category, center, self.object_radius, reason)
        self.object_entries.append(entry)
        self.stats['object']['added'] += 1
        print(f"[Blacklist:Object] Added {category}#{object_id} at {center}, reason={reason}")

        # Enforce max entries (remove oldest)
        if len(self.object_entries) > self.object_max_entries:
            removed = self.object_entries.pop(0)
            print(f"[Blacklist:Object] Removed oldest: {removed.category}#{removed.object_id}")

    def is_blacklisted(self,
                       center: List[float],
                       object_id: Optional[str] = None,
                       category: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Check if object is blacklisted (check both layers)

        Args:
            center: Object center in base frame [x,y,z] (mm)
            object_id: Object ID (if available)
            category: Object category (if available)

        Returns:
            (is_blacklisted, reason):
                - (True, reason): Object is blacklisted
                - (False, None): Object is not blacklisted

        Priority: Object ID match > Spatial match
        """
        # Clean expired entries first
        self._clean_expired()

        # Layer 2: Check object blacklist first (if ID provided)
        if object_id and category:
            for entry in self.object_entries:
                if entry.matches(object_id, category, center):
                    self.stats['object']['matched'] += 1
                    distance = np.linalg.norm(np.array(center) - entry.current_center)
                    print(f"[Blacklist:Object] Matched {category}#{object_id}, "
                          f"distance={distance:.0f}mm, attempts={entry.attempts}")
                    return (True, f"OBJECT:{entry.reason}")

        # Layer 1: Check spatial blacklist
        for entry in self.spatial_entries:
            if entry.matches(center):
                self.stats['spatial']['matched'] += 1
                distance = np.linalg.norm(np.array(center) - entry.center)
                print(f"[Blacklist:Spatial] Matched at {center}, "
                      f"distance={distance:.0f}mm, attempts={entry.attempts}")
                return (True, f"SPATIAL:{entry.reason}")

        return (False, None)

    def _clean_expired(self) -> None:
        """Remove expired entries from both layers"""
        # Spatial layer
        original_count = len(self.spatial_entries)
        self.spatial_entries = [e for e in self.spatial_entries
                                if not e.is_expired(self.spatial_max_age)]
        removed = original_count - len(self.spatial_entries)
        if removed > 0:
            self.stats['spatial']['expired'] += removed
            print(f"[Blacklist:Spatial] Removed {removed} expired entries")

        # Object layer
        original_count = len(self.object_entries)
        self.object_entries = [e for e in self.object_entries
                               if not e.is_expired(self.object_max_age)]
        removed = original_count - len(self.object_entries)
        if removed > 0:
            self.stats['object']['expired'] += removed
            print(f"[Blacklist:Object] Removed {removed} expired entries")

    def clear(self) -> None:
        """Clear all entries"""
        spatial_count = len(self.spatial_entries)
        object_count = len(self.object_entries)
        self.spatial_entries = []
        self.object_entries = []
        print(f"[Blacklist] Cleared {spatial_count} spatial + {object_count} object entries")

        if self.persist_file:
            self.save()

    def get_stats(self) -> dict:
        """Get statistics"""
        return {
            'spatial': {
                'current_count': len(self.spatial_entries),
                'radius': self.spatial_radius,
                'max_age': self.spatial_max_age,
                **self.stats['spatial']
            },
            'object': {
                'current_count': len(self.object_entries),
                'radius': self.object_radius,
                'max_age': self.object_max_age,
                **self.stats['object']
            }
        }

    def save(self) -> None:
        """Save to file"""
        if not self.persist_file:
            return

        try:
            data = {
                'spatial_entries': [e.to_dict() for e in self.spatial_entries],
                'object_entries': [e.to_dict() for e in self.object_entries],
                'stats': self.stats
            }
            with open(self.persist_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[Blacklist] Saved to {self.persist_file}")
        except Exception as e:
            print(f"[Blacklist] Failed to save: {e}")

    def load(self) -> None:
        """Load from file"""
        if not self.persist_file:
            return

        try:
            with open(self.persist_file, 'r') as f:
                data = json.load(f)

            self.spatial_entries = [SpatialBlacklistEntry.from_dict(e)
                                    for e in data.get('spatial_entries', [])]
            self.object_entries = [ObjectBlacklistEntry.from_dict(e)
                                   for e in data.get('object_entries', [])]
            self.stats = data.get('stats', self.stats)

            # Clean expired on load
            self._clean_expired()

            print(f"[Blacklist] Loaded {len(self.spatial_entries)} spatial + "
                  f"{len(self.object_entries)} object entries")
        except FileNotFoundError:
            print(f"[Blacklist] No existing file at {self.persist_file}")
        except Exception as e:
            print(f"[Blacklist] Failed to load: {e}")


if __name__ == '__main__':
    # Test dual-layer blacklist
    config = {
        'spatial': {'radius': 30, 'max_age': 600, 'max_entries': 100},
        'object': {'radius': 60, 'max_age': 300, 'max_entries': 50, 'require_id': True},
        'persist_file': '/tmp/dual_blacklist_test.json'
    }

    blacklist = DualLayerBlacklist(config)

    # Test 1: Add with object ID
    print("\n=== Test 1: Add with Object ID ===")
    blacklist.add(
        center=[800, 500, -300],
        reason="TARGET_LIMIT",
        object_id="bottle_123",
        category="bottle"
    )

    # Test 2: Check exact match
    print("\n=== Test 2: Exact match ===")
    is_black, reason = blacklist.is_blacklisted([800, 500, -300], "bottle_123", "bottle")
    print(f"Result: blacklisted={is_black}, reason={reason}")

    # Test 3: Object moved (within object radius 60mm)
    print("\n=== Test 3: Object moved 50mm ===")
    is_black, reason = blacklist.is_blacklisted([850, 500, -300], "bottle_123", "bottle")
    print(f"Result: blacklisted={is_black}, reason={reason}")

    # Test 4: Different object, same location (spatial match)
    print("\n=== Test 4: Different object, same location ===")
    is_black, reason = blacklist.is_blacklisted([805, 500, -300], "cup_456", "cup")
    print(f"Result: blacklisted={is_black}, reason={reason}")

    # Test 5: Stats
    print("\n=== Test 5: Statistics ===")
    print(json.dumps(blacklist.get_stats(), indent=2))
