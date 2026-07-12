"""Execute the browser-independent result-summary regression contract."""

from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from pathlib import Path


class ResultSummaryJavaScriptTests(unittest.TestCase):
    def test_production_and_mixed_payload_copy(self) -> None:
        node_binary = os.getenv("NODE_BINARY") or shutil.which("node")
        if not node_binary:
            self.skipTest("Node.js is not available; run tests/result_summary.test.cjs directly")

        test_file = Path(__file__).with_name("result_summary.test.cjs")
        completed = subprocess.run(
            [node_binary, str(test_file)],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=(completed.stdout + completed.stderr).strip(),
        )
        self.assertIn("Result summary regression cases passed", completed.stdout)


if __name__ == "__main__":
    unittest.main()
