import json
import os
from pathlib import Path
import unittest

import numpy as np


class PaddleReferenceArtifactsTest(unittest.TestCase):
    def test_reference(self):
        root = Path(os.environ.get("PADDLE_REFERENCE_OUTPUT", "artifacts/paddle"))
        info = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(info["ctc_classes"], 18710)
        self.assertEqual(info["nrtr_classes"], 18714)
        self.assertEqual(info["checkpoint"]["missing"], [])
        self.assertEqual(info["checkpoint"]["unexpected"], [])
        self.assertEqual(info["checkpoint"]["mismatched"], [])
        self.assertEqual(info["samples"]["official"]["text"], "绿洲仕格维花园公寓")
        self.assertGreaterEqual(len(info["samples"]), 3)
        with np.load(root / "reference.npz", allow_pickle=False) as arrays:
            self.assertEqual(arrays["train/input"].shape, (2, 3, 48, 320))
            self.assertEqual(arrays["train/ctc_logits"].shape, (2, 40, 18710))
            self.assertEqual(arrays["train/nrtr_logits"].shape, (2, 6, 18714))
            self.assertEqual(arrays["eval/synthetic_0/input"].shape, (1, 3, 48, 320))
            self.assertGreater(arrays["eval/synthetic_1/input"].shape[-1], 320)
            self.assertIn("eval/official/backbone.blocks4.6", arrays.files)
            self.assertIn("eval/official/head.ctc_encoder.encoder.local_conv", arrays.files)
            self.assertIn("train/head.gtc_head.decoder.3", arrays.files)
            np.testing.assert_allclose(
                arrays["train/loss/loss"],
                arrays["train/loss/CTCLoss"] + arrays["train/loss/NRTRLoss"],
            )
            for name in arrays.files:
                self.assertTrue(np.isfinite(arrays[name]).all(), name)
