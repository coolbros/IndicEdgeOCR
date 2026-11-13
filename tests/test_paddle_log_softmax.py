import json
import unittest

import torch

from ppocrv6_torch.losses import multi_loss
from ppocrv6_torch.ops.paddle_log_softmax import paddle_compatible_log_softmax


class PaddleLogSoftmaxTest(unittest.TestCase):
    def test_strided_axes_and_ties(self):
        x = torch.tensor([[2., 2., -1.], [3., 1., -5.]]).transpose(0, 1)
        for implementation in ("torch", "pairwise", "sequential"):
            torch.testing.assert_close(
                paddle_compatible_log_softmax(x, 0, implementation),
                x.log_softmax(0),
            )

    def test_clip_and_autograd(self):
        x = torch.tensor([[700., 699., 636.01, 635.99]], dtype=torch.float64,
                         requires_grad=True)
        y = paddle_compatible_log_softmax(x)
        self.assertAlmostEqual((y[0, 2] - y[0, 3]).item(), 0.01, places=12)
        for implementation in ("torch", "pairwise", "sequential"):
            self.assertTrue(torch.autograd.gradcheck(
                lambda values: paddle_compatible_log_softmax(values, implementation=implementation), (x,),
            ))

    def test_loss_selection_round_trips_and_backpropagates(self):
        config = json.loads(json.dumps({"ctc_log_softmax": "torch",
                                        "nrtr_log_softmax": "pairwise"}))
        a = torch.randn(1, 4, 5, requires_grad=True)
        b = torch.randn(1, 3, 6, requires_grad=True)
        batch = (torch.tensor([[2, 1]]), torch.tensor([[2, 4, 5, 3]]), torch.tensor([2]))
        standard = multi_loss({"ctc": a, "gtc": b}, *batch)
        alternate = multi_loss({"ctc": a, "gtc": b}, *batch, **config)
        torch.testing.assert_close(standard["loss"], alternate["loss"])
        for first, second in zip(torch.autograd.grad(standard["loss"], (a, b)),
                                 torch.autograd.grad(alternate["loss"], (a, b))):
            torch.testing.assert_close(first, second)
