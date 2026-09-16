import unittest
from unittest.mock import patch

from simulation.framework import backtest_align


class BacktestAlignEncodingTests(unittest.TestCase):
    def test_child_output_is_decoded_as_utf8(self):
        completed = type("Completed", (), {"returncode": 0, "stdout": None, "stderr": None})()
        with patch.object(backtest_align.subprocess, "run", return_value=completed) as run:
            self.assertIsNone(backtest_align.run_backtest("momentum_rotation", "2026-07-01", "2026-09-16", "test"))
        kwargs = run.call_args.kwargs
        self.assertEqual("utf-8", kwargs["encoding"])
        self.assertEqual("replace", kwargs["errors"])
        self.assertEqual("utf-8", kwargs["env"]["PYTHONIOENCODING"])

    def test_nonzero_return_with_missing_stderr_does_not_mask_failure(self):
        completed = type("Completed", (), {"returncode": 1, "stdout": None, "stderr": None})()
        with patch.object(backtest_align.subprocess, "run", return_value=completed):
            self.assertIsNone(backtest_align.run_backtest("momentum_rotation", "2026-07-01", "2026-09-16", "test"))


if __name__ == "__main__":
    unittest.main()
