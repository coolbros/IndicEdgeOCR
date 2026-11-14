import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from trace_backward import Trace, array_hash, gradient_metrics, unique_targets, vjp


class TraceBackwardTest(unittest.TestCase):
    def test_vjp_preserves_named_aliases_and_requires_affine_gradients(self):
        norm = torch.nn.LayerNorm(3)
        x = torch.tensor([[1., 2., 4.]], requires_grad=True)
        upstream = torch.tensor([[0.25, 0.5, -1.]])
        targets = {"input": x, "alias": x, "weight": norm.weight, "bias": norm.bias}
        values, indices = unique_targets(targets)
        self.assertEqual(len(values), 3)
        self.assertEqual(indices["input"], indices["alias"])
        result = vjp(torch.autograd.grad, norm(x), upstream, targets)
        self.assertEqual(set(result), set(targets))
        torch.testing.assert_close(result["bias"], upstream.flatten())
        self.assertIs(result["input"], result["alias"])

    def test_gradient_metrics_keep_original_threshold_and_localize_outliers(self):
        a = np.zeros((1, 2, 4), np.float32)
        b = a.copy()
        b[0, 1, 2] = 2e-5
        row = gradient_metrics(a, b, {"token": 1, "channel": 2})
        self.assertFalse(row["pass"])
        self.assertEqual(row["outliers"], 1)
        self.assertEqual(row["outlier_fraction"], 1 / 8)
        self.assertEqual(row["outlier_axes"]["token"], [0, 1])
        self.assertEqual(row["outlier_axes"]["channel"], [0, 0, 1, 0])
        self.assertIsNone(row["relative_l2"])

    def test_missing_gradient_names_the_disconnected_parameter(self):
        x = torch.tensor([2.], requires_grad=True)
        weight = torch.tensor([3.], requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "unused_weight"):
            vjp(torch.autograd.grad, x.square(), torch.ones_like(x),
                {"input": x, "unused_weight": weight})

    def test_reference_rejects_array_with_wrong_content(self):
        sample = np.ones((1, 3), np.float32)
        checksum = array_hash(sample)
        trace = Trace(None, None, {"input": {"sha256": checksum}})
        trace.reference_arrays = {checksum: sample.copy()}
        np.testing.assert_array_equal(trace.exact("input"), sample)
        trace.reference_arrays[checksum][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "Corrupt reference tensor: input"):
            trace.exact("input")
