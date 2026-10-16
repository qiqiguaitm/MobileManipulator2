"""Routing contract: recorded commands must never become simulation commands."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene_topics import resolve_topics, REFERENCE_TOPICS


class RoutingTests(unittest.TestCase):
    def test_native_topics_are_unchanged(self):
        names = ["/scan", "/current_pose", "/tf", "/camera/top/color/image_raw"]
        self.assertEqual(resolve_topics(names), {name: name for name in names})

    def test_prefixed_input_rejected(self):
        for names in [["/reference/scan"], ["/scan", "/reference/scan"]]:
            with self.assertRaisesRegex(ValueError, "without /reference/"):
                resolve_topics(names)

    def test_reference_whitelist_excludes_active_sensor_and_tf_topics(self):
        for topic in ["/cmd_vel", "/goal_pose", "/plan", "/current_pose"]:
            self.assertIn(topic, REFERENCE_TOPICS)
            self.assertNotEqual("/reference" + topic, topic)
        for topic in ["/map", "/scan", "/tf", "/tf_static", "/arbitrary_command"]:
            self.assertNotIn(topic, REFERENCE_TOPICS)


if __name__ == "__main__":
    unittest.main()
