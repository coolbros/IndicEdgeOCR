import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from phase36_evaluate import edit_distance
from trace_parity import BASELINE, metrics


class Phase36ToolsTest(unittest.TestCase):
    def test_tolerance_reports_outlier_count_and_preserves_dtype(self):
        expected = np.array([0, 1, 2], dtype=np.float32)
        actual = expected + np.array([2e-6, 0, 1e-5], dtype=np.float32)
        row = metrics(expected, actual, "probability")
        self.assertFalse(row["pass"])
        self.assertEqual((row["pass_count"], row["elements"]), (2, 3))
        self.assertFalse(metrics(expected, expected.astype(np.float64), "probability")["pass"])

    def test_experiment_rejects_unpinned_archive_before_creating_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path = Path(temp) / "fixture"
            path.write_bytes(b"changed")
            output = path.with_name("output")
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools/phase36_experiment.py"), "torch", "cpu",
                 "--source", temp, "--archive", str(path), "--original", str(path),
                 "--checkpoint", str(path), "--dataset", temp, "--output", str(output)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Input digest mismatch: archive", result.stderr)
            self.assertFalse(output.exists())

    def test_kaggle_staging_is_reproducible_and_private(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            audit = root / "audit.json"
            audit.write_text(json.dumps({"passed": True,
                                         "manifest_sha256": {"train.jsonl": BASELINE["train_manifest"]}}))
            for name in ("one", "two"):
                subprocess.run(
                    [sys.executable, str(ROOT / "tools/prepare_kaggle_phase36.py"),
                     "--owner", "owner", "--directory", str(root / name), "--audit", str(audit),
                     "--reference", "owner/fixture/4"], check=True,
                )
            first, second = (root / name / "phase36.py" for name in ("one", "two"))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            ast.parse(first.read_text(encoding="utf8"))
            metadata = json.loads(first.with_name("kernel-metadata.json").read_text())
            self.assertTrue(metadata["is_private"])
            self.assertEqual(metadata["kernel_sources"], ["owner/fixture/4"])

    def test_word_and_codepoint_distances(self):
        self.assertEqual(edit_distance("cafe\u0301", "caf\u00e9"), 2)
        self.assertEqual(edit_distance("three words now".split(), "two words".split()), 2)
