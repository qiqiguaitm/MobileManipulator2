import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preview import select_frames


class PreviewTests(unittest.TestCase):
    def test_three_rgb_depth_pairs(self):
        color = "/camera/top/color/image_raw"
        depth = "/camera/top/aligned_depth_to_color/image_raw"
        stamps = [i * 100_000_000 for i in range(11)]
        selected = select_frames(
            {color: stamps, depth: [s + 1_000_000 for s in stamps]}
        )
        self.assertEqual(selected[color], {0, 5, 10})
        self.assertEqual(selected[depth], {0, 5, 10})

    def test_unmatched_depth_not_fabricated(self):
        selected = select_frames(
            {
                "/camera/top/color/image_raw": [0, 1_000_000_000],
                "/camera/top/depth/image_raw": [500_000_000],
            }
        )
        self.assertEqual(selected["/camera/top/depth/image_raw"], set())

    def test_short_and_multiple_cameras(self):
        streams = {
            "/camera/top/color/image_raw": [0, 10],
            "/camera/hand/color/image_raw": [50, 60, 70, 80, 90],
        }
        selected = select_frames(streams)
        self.assertEqual(selected["/camera/top/color/image_raw"], {0, 1})
        self.assertEqual(selected["/camera/hand/color/image_raw"], {0, 2, 4})
        self.assertEqual(select_frames({}), {})


if __name__ == "__main__":
    unittest.main()
