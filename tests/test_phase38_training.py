from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from accelerate import Accelerator

from ppocrv6_torch import PPOCRv6Medium, RecognitionTokens
from ppocrv6_torch.data import Vocabulary, resize_image
from ppocrv6_torch.losses import multi_loss
from ppocrv6_torch.train_data import augment, concatenate, multiscale_batches, scale_pairs
from ppocrv6_torch.training import cosine_schedule, optimizer_for

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from train_smoke import restore_checkpoint, save_checkpoint


class Phase38TrainingTest(unittest.TestCase):
    def test_training_resize_and_official_scale_sizes(self):
        self.assertEqual(scale_pairs(), [(320, 32, 64), (320, 48, 42), (320, 64, 32)])
        image = np.full((32, 20, 3), (0, 128, 255), np.uint8)
        values, ratio = resize_image(image, height=32, width=320)
        self.assertEqual(values.shape, (3, 32, 320))
        np.testing.assert_allclose(values[:, 0, 0], [-1, 128/127.5-1, 1], atol=1e-7)
        self.assertEqual(ratio, 20/320)
        self.assertFalse(values[:, :, 20:].any())

    def test_sampler_epoch_and_padded_tail(self):
        batches = multiscale_batches(140)
        ids = [i for _, _, batch in batches for i in batch]
        self.assertEqual(set(ids), set(range(140)))
        self.assertEqual(len(ids), 202)
        self.assertEqual(batches, multiscale_batches(140))
        self.assertNotEqual(batches, multiscale_batches(140, epoch=1))

    def test_crop_and_concat_label_geometry(self):
        image = np.full((24, 20, 3), 190, np.uint8)
        with patch("random.random", return_value=.5):
            joined, text = concatenate(image, "ab", [(image, "c"), (image, "de")], probability=1)
            rejected, old_text = concatenate(image, "abc", [(image, "def")], probability=1, max_length=5)
        self.assertEqual((joined.shape, text), ((48, 120, 3), "abcde"))
        self.assertIs(rejected, image)
        self.assertEqual(old_text, "abc")
        off = dict(tia=0, crop=0, blur=0, hsv=0, jitter=0, noise=0, reverse=0)
        with patch("random.random", return_value=.5), patch("random.randint", return_value=2):
            np.testing.assert_array_equal(augment(image, **off), image)
            np.testing.assert_array_equal(augment(image, **(off | {"reverse": 1})), 255-image)
            self.assertEqual(augment(image, **(off | {"crop": 1})).shape, (22, 20, 3))
        for name in ("tia", "blur", "hsv", "jitter", "noise"):
            random.seed(3808)
            np.random.seed(3808)
            value = augment(image.copy(), **(off | {name: 1}))
            self.assertEqual(value.shape, image.shape)
            self.assertEqual(value.dtype, np.uint8)

    def test_independent_nondefault_indices_and_head_sizes(self):
        model = PPOCRv6Medium(3, 8, blank=1, pad=7, bos=5, eos=6)
        vocab = Vocabulary.from_symbols({"A": 0, "B": 2}, {"A": 1, "B": 3}, model.tokens)
        c, n, length = vocab.encode("AB")
        np.testing.assert_array_equal(c[:4], [0, 2, 1, 1])
        np.testing.assert_array_equal(n[:5], [5, 1, 3, 6, 7])
        ctc = model.head.ctc_head(torch.zeros(1, 4, 192))
        nrtr = model.head.gtc_head(torch.zeros(1, 4, 512), torch.tensor(n)[None], torch.tensor([length]))
        self.assertEqual((ctc.shape[-1], nrtr.shape[-1]), (3, 8))
        values = multi_loss({"ctc": ctc, "gtc": nrtr}, torch.tensor(c)[None],
                            torch.tensor(n)[None], torch.tensor([length]), blank=1, pad=7)
        self.assertTrue(torch.isfinite(values["loss"]))
        self.assertEqual(vocab.decode(np.eye(3, dtype=np.float32)[[0, 1, 2]][None]), [("AB", 1.0)])
        with self.assertRaises(ValueError):
            vocab.encode("")
        with self.assertRaises(ValueError):
            vocab.encode("A" * 24)
        with self.assertRaises(ValueError):
            RecognitionTokens(3, 8, blank=1.0)
        with self.assertRaises(ValueError):
            Vocabulary.from_symbols({"A": False, "B": 2}, {"A": 1, "B": 3}, model.tokens)

    def test_midrun_resume_matches_uninterrupted_update(self):
        torch.manual_seed(3801)
        accelerator = Accelerator(cpu=True)
        model = torch.nn.Linear(3, 1)
        optimizer = optimizer_for(model)
        schedule = cosine_schedule(optimizer, 3, epochs=4, warmup_epochs=1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0)

        def update():
            optimizer.zero_grad()
            model(torch.rand(3)).sum().backward()
            optimizer.step()
            schedule.step()

        for _ in range(3):
            update()
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0005)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "latest.pt"
            save_checkpoint(path, model, optimizer, schedule, accelerator,
                            {"config": {"seed": 3801}, "epoch": 1, "global_step": 3})
            update()
            expected = model.weight.detach().clone()
            moment = optimizer.state[model.weight]["exp_avg"].clone()
            restore_checkpoint(path, model, optimizer, schedule, accelerator, {"seed": 3801})
            update()
            torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)
            torch.testing.assert_close(optimizer.state[model.weight]["exp_avg"], moment, rtol=0, atol=0)

    def test_impossible_ctc_alignment_is_visible(self):
        logits = {"ctc": torch.zeros(1, 2, 3), "gtc": torch.zeros(1, 3, 6)}
        result = multi_loss(logits, torch.tensor([[1, 1]]), torch.tensor([[2, 4, 4, 3]]),
                            torch.tensor([2]))
        self.assertTrue(torch.isinf(result["CTCLoss"]))
        self.assertTrue(torch.isfinite(result["NRTRLoss"]))

    def test_real_checkpoint_loads_with_default_head_shapes(self):
        from safetensors.torch import load_file
        checkpoint = Path("weights/converted/PP-OCRv6_medium_rec.safetensors")
        if not checkpoint.is_file():
            self.skipTest("Official checkpoint is an external fixture")
        model = PPOCRv6Medium()
        model.load_state_dict(load_file(checkpoint), strict=True)
        self.assertEqual(model.head.ctc_head.fc.out_features, 18710)
        self.assertEqual(model.head.gtc_head.tgt_word_prj.out_features, 18714)

    def test_shared_gradient_is_additive_and_order_independent(self):
        torch.manual_seed(3801)
        model = PPOCRv6Medium(9, 13)
        model.eval()
        pooled = torch.randn(2, 768, 1, 4, requires_grad=True)
        target = torch.tensor([[2, 1, 3], [2, 4, 3]])
        lengths = torch.tensor([1, 1])
        model.head.gtc_head.train()
        for m in model.head.gtc_head.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()
        ctc = model.head.ctc_head.fc(model.head.ctc_encoder(pooled)).sum()
        nrtr = model.head.gtc_head(model.head.before_gtc(pooled), target, lengths).sum()
        c, = torch.autograd.grad(ctc, pooled, retain_graph=True)
        n, = torch.autograd.grad(nrtr, pooled, retain_graph=True)
        joint, = torch.autograd.grad((ctc, nrtr), pooled, retain_graph=True)
        reverse, = torch.autograd.grad((nrtr, ctc), pooled)
        torch.testing.assert_close(joint, c+n, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(joint, reverse, rtol=1e-4, atol=1e-5)
