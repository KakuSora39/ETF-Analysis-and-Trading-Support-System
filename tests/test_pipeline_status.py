import json
import tempfile
import unittest
from pathlib import Path

from pipeline import STEPS
import pipeline
from pipeline_status import PipelineStatus


class PipelineStatusTests(unittest.TestCase):
    def test_pipeline_console_uses_utf8(self):
        self.assertEqual("utf-8", pipeline.sys.stdout.encoding.lower())
        self.assertEqual("utf-8", pipeline.sys.stderr.encoding.lower())

    def test_all_pipeline_step_names_round_trip_as_utf8(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "pipeline_status.json"
            status = PipelineStatus(path)
            status.reset()
            for step in STEPS:
                status.add_step(step["id"], step["name"])
            raw = path.read_bytes()
            self.assertIn("黄金避险轮动 🥇".encode("utf-8"), raw)
            saved = json.loads(raw.decode("utf-8"))
            self.assertEqual("黄金避险轮动 🥇", saved["steps"]["gold_safe_haven"]["name"])
            self.assertEqual(saved, PipelineStatus(path).load())


if __name__ == "__main__":
    unittest.main()
