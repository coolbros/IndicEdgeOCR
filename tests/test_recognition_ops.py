from pathlib import Path
import unittest

import cv2
import numpy as np
import torch

from ppocrv6_torch.backbone import BatchNorm2d, EvenSameMaxPool
from ppocrv6_torch.data import Vocabulary, preprocess
from ppocrv6_torch.losses import multi_loss


DICTIONARY = Path("references/PaddleOCR/ppocr/utils/dict/ppocrv6_dict.txt")


class RecognitionOpsTest(unittest.TestCase):
    def test_population_variance_cpu(self):
        norm = BatchNorm2d(1)
        x = torch.tensor([[[[1.0, 3.0]]]])
        torch.testing.assert_close(norm(x), (x - 2) / (1 + 1e-5)**0.5)
        torch.testing.assert_close(norm.running_mean, torch.tensor([0.2]))
        torch.testing.assert_close(norm.running_var, torch.tensor([1.0]))

    def test_asymmetric_max_pool_keeps_negative_border(self):
        value = torch.tensor([[[[-4.0, -3.0], [-2.0, -1.0]]]])
        torch.testing.assert_close(EvenSameMaxPool()(value), torch.full_like(value, -1))

    def test_preprocess_and_label_indices(self):
        image = np.full((8, 65, 3), (0, 128, 255), dtype=np.uint8)
        content = cv2.imencode(".png", image)[1].tobytes()
        x, ratio = preprocess(content, inference=True)
        self.assertEqual(x.shape, (3, 48, 390))
        self.assertEqual(ratio, 1)
        np.testing.assert_allclose(x[:, 0, 0], [-1, 128 / 127.5 - 1, 1], atol=1e-7)
        vocab = Vocabulary(DICTIONARY)
        ctc, nrtr, length = vocab.encode("AA")
        self.assertEqual(length, 2)
        np.testing.assert_array_equal(nrtr[:4], [2, ctc[0] + 3, ctc[1] + 3, 3])
        probabilities = np.zeros((1, 4, len(vocab.characters)), dtype=np.float32)
        probabilities[0, np.arange(4), [ctc[0], ctc[0], 0, ctc[0]]] = 1
        self.assertEqual(vocab.decode(probabilities), [("AA", 1.0)])

    def test_losses_have_the_expected_analytic_reductions(self):
        ctc = torch.zeros(1, 2, 2, requires_grad=True)
        nrtr = torch.zeros(1, 2, 4, requires_grad=True)
        loss = multi_loss({"ctc": ctc, "gtc": nrtr}, torch.tensor([[1]]),
                          torch.tensor([[2, 1, 3]]), torch.tensor([1]))
        torch.testing.assert_close(loss["CTCLoss"], -torch.log(torch.tensor(0.75)))
        torch.testing.assert_close(loss["NRTRLoss"], torch.log(torch.tensor(4.0)))
        loss["loss"].backward()
        self.assertTrue(torch.isfinite(ctc.grad).all())
        self.assertTrue(torch.isfinite(nrtr.grad).all())
