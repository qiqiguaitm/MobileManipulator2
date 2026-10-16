import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catalog import resolve_artifact, catalog


class CatalogTests(unittest.TestCase):
    def test_paths_and_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'
            scene = root / 'scenes/accepted'
            scene.mkdir(parents=True)
            (scene / 'test.mcap').write_bytes(b'test')
            (root / 'pending/new').mkdir(parents=True)
            (root / 'scenes/unlisted').mkdir()
            manifest = Path(tmp) / 'test_set.yaml'
            manifest.write_text('scenes:\n- id: accepted\n  name: Accepted\n  enabled: true\n  scene_dir: scenes/accepted\n')
            page = catalog(root, manifest, 'http://localhost:18770')
            self.assertIn('<h2>场景库</h2><span class="count">2</span>', page)
            self.assertIn('已启用', page)
            self.assertIn('未登记', page)
            self.assertIn('<h2>待入集场景</h2><span class="count">1</span>', page)
            self.assertIn('unlisted', page)
            self.assertEqual(resolve_artifact(root, 'scenes/accepted/test.mcap'), scene / 'test.mcap')
            secret = Path(tmp) / 'secret.json'
            secret.write_text('{}')
            (scene / 'outside.json').symlink_to(secret)
            for path in ['scenes/../secret.json', 'scenes/accepted/outside.json', '.viewer_deps/file.json', '/etc/passwd', 'scenes/missing.mcap']:
                self.assertIsNone(resolve_artifact(root, path))
