"""Routing contract: recorded commands must never become simulation commands."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene_topics import resolve_topics, REFERENCE_TOPICS


class RoutingTests(unittest.TestCase):
    def test_native_topics_take_precedence(self):
        names = ["/scan", "/reference/scan", "/current_pose", "/reference/trajectory"]
        resolved = resolve_topics(names)
        self.assertEqual(resolved["/scan"], "/scan")
        self.assertEqual(resolved["/current_pose"], "/current_pose")
        self.assertEqual(resolved, resolve_topics(reversed(names)))

    def test_legacy_pose_and_sensors(self):
        resolved = resolve_topics(
            [
                "/reference/trajectory",
                "/reference/tf",
                "/reference/camera/top/color/image_raw",
            ]
        )
        self.assertEqual(resolved["/current_pose"], "/reference/trajectory")
        self.assertEqual(resolved["/tf"], "/reference/tf")
        self.assertIn("/camera/top/color/image_raw", resolved)

    def test_reference_whitelist_excludes_active_sensor_and_tf_topics(self):
        for topic in ["/cmd_vel", "/goal_pose", "/plan", "/current_pose"]:
            self.assertIn(topic, REFERENCE_TOPICS)
            self.assertNotEqual("/reference" + topic, topic)
        for topic in ["/map", "/scan", "/tf", "/tf_static", "/arbitrary_command"]:
            self.assertNotIn(topic, REFERENCE_TOPICS)


if __name__ == "__main__":
    unittest.main()
