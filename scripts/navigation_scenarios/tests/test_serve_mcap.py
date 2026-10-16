"""File service allows in-tree MCAP links without exposing outside files."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve_mcap import resolve_mcap


class LinkTests(unittest.TestCase):
    def test_resolved_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            allowed = base / "scenarios"
            root = allowed / "remote_mcap"
            root.mkdir(parents=True)
            target = allowed / "scene.mcap"
            target.write_bytes(b"test")
            (root / "valid.mcap").symlink_to("../scene.mcap")
            outside = base / "outside.mcap"
            outside.write_bytes(b"private")
            (root / "outside.mcap").symlink_to(outside)
            (root / "broken.mcap").symlink_to(allowed / "missing.mcap")
            (root / "loop.mcap").symlink_to("loop.mcap")
            text = allowed / "private.txt"
            text.write_text("private")
            (root / "disguised.mcap").symlink_to(text)
            self.assertEqual(resolve_mcap(root, allowed, "valid.mcap"), target)
            for name in [
                "outside.mcap",
                "broken.mcap",
                "loop.mcap",
                "disguised.mcap",
                "../scene.mcap",
            ]:
                self.assertIsNone(resolve_mcap(root, allowed, name), name)
            self.assertIsNone(resolve_mcap(root, root, "valid.mcap"))
            (root / "plain.mcap").write_bytes(b"plain")
            self.assertEqual(
                resolve_mcap(root, allowed, "plain.mcap"), root / "plain.mcap"
            )


if __name__ == "__main__":
    unittest.main()
