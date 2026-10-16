import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_report import report


class ReportTest(unittest.TestCase):
    def test_report_metrics_links_and_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / 'simulation_runs/run_20261016T000000Z'
            output = run / 'example'
            output.mkdir(parents=True)
            (output / 'metrics.json').write_text(json.dumps({'success': False, 'failure_stage': 'planning'}))
            for folder in [root / 'scenes/example/visualization_bag', output / 'result_bag']:
                folder.mkdir(parents=True)
                (folder / 'data.mcap').write_bytes(b'test')
            cfg = {'scenes': [{'id': 'example', 'name': '<unsafe>', 'scene_dir': 'scenes/example'}]}
            result = report(run, root, cfg)
            report(run, root, cfg)
            self.assertEqual(result['success_rate'], 0)
            page = (run / 'report.html').read_text()
            self.assertIn('&lt;unsafe&gt;', page)
            self.assertEqual(page.count('ds=remote-file'), 2)
            self.assertFalse((root / 'remote_mcap').exists())
            self.assertIn('simulation_runs%2Frun_20261016T000000Z', page)
