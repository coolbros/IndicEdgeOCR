import unittest

import torch

from ppocrv6_torch import PPOCRv6Medium


class RecognizerTest(unittest.TestCase):
    def test_inventory_and_modes(self):
        torch.set_num_threads(2)
        model = PPOCRv6Medium()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 55_527_798)
        self.assertEqual(sum(b.numel() for b in model.buffers()), 2_622_784)
        self.assertEqual(len(model.state_dict()), 441)
        images = torch.randn(2, 3, 48, 320)
        targets = torch.tensor([[2, 7, 3, 0], [2, 9, 8, 3]])
        with torch.no_grad():
            train = model(images, (None, targets, torch.tensor([1, 2]), None))
            self.assertEqual(train["ctc"].shape, (2, 40, 18710))
            self.assertEqual(train["gtc"].shape, (2, 3, 18714))
            model.eval()
            probabilities = model(images)
            torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 40))

    def test_rep_branch_fusion(self):
        from ppocrv6_torch.backbone import RepDWConv

        layer = RepDWConv(12).eval()
        x = torch.randn(2, 12, 5, 13)
        with torch.no_grad():
            layer.bn.running_mean.uniform_(-1, 1)
            layer.conv.bn.running_var.uniform_(0.5, 2)
            torch.testing.assert_close(layer(x), layer.fuse()(x), atol=3e-6, rtol=1e-5)
