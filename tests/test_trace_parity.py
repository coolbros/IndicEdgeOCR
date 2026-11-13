from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from trace_parity import attention_steps, metrics


class TraceParityTest(unittest.TestCase):
    def test_metrics_count_outliers_and_agree_on_causal_mask(self):
        a = np.array([0, 1, -np.inf], dtype=np.float32)
        b = np.array([2e-5, 1.00001, -np.inf], dtype=np.float32)
        result = metrics(a, b, "activation")
        self.assertEqual(result["pass_count"], 2)
        self.assertEqual(result["elements"], 3)
        self.assertIs(result["pass"], False)
        self.assertIs(metrics(a, a, "activation")["pass"], True)
        self.assertEqual(set(result["percentile_abs"]), {"50", "90", "99", "99.9"})

    def test_attention_mask_and_layout(self):
        q = torch.ones(1, 16, 2, 32)
        v = q.clone()
        v[:, :, 1, :] *= 3
        mask = torch.full((2, 2), -torch.inf).triu(1)
        steps = attention_steps(q, q, v, mask, lambda x: x.softmax(-1))
        self.assertEqual(tuple(steps["merge"][1].shape), (1, 2, 512))
        torch.testing.assert_close(steps["merge"][1][0, :, 0], torch.tensor([1.0, 2.0]))
