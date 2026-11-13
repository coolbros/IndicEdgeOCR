import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_dataset_splits import audit


class SplitAuditTest(unittest.TestCase):
    def audit_pair(self, case):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            img = np.random.default_rng(2026).integers(16, 239, (16, 18, 3), dtype=np.uint8)
            other = np.random.default_rng(2027).integers(16, 239, img.shape, dtype=np.uint8)
            rows = []
            for i, split in enumerate(("train", "test")):
                image = img if i == 0 or case == "exact" else img + 4 if case == "near" else other
                content = cv2.imencode(".png", image)[1].tobytes()
                path = root / f"{i}.png"
                path.write_bytes(content)
                rows.append({"sample_id": str(0 if case == "id" else i), "image_path": path.name,
                             "transcription": "same word", "dataset_source": "fixture",
                             "source_document_id": str(0 if case == "group" else i),
                             "split": split, "original_split": split, "license": "fixture",
                             "image_sha256": hashlib.sha256(content).hexdigest(),
                             "retrieved_on": "2026-11-13", "source_url": "fixture"})
            mutations = {"label": ("transcription", ""), "escape": ("image_path", "../outside"),
                         "split": ("split", "other"), "partition": ("original_split", "train"),
                         "unreadable": ("image_path", "manifest.jsonl")}
            if case in mutations:
                key, value = mutations[case]
                rows[1][key] = value
            path = root / "manifest.jsonl"
            path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            return audit([path], root)

    def test_repeated_transcription_is_allowed(self):
        self.assertTrue(self.audit_pair("clean")["passed"])

    def test_hard_failures(self):
        for case in ("id", "exact", "group", "label", "escape", "split", "partition", "unreadable"):
            with self.subTest(case=case):
                self.assertTrue(self.audit_pair(case)["errors"])

    def test_near_duplicate(self):
        result = self.audit_pair("near")
        self.assertFalse(result["passed"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["near_duplicates"][0]["distance"], 0)
