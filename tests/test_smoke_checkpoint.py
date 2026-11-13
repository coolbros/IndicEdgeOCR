from pathlib import Path
import sys
import tempfile
import unittest

from accelerate import Accelerator
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from train_smoke import restore_checkpoint, save_checkpoint


class SmokeCheckpointTest(unittest.TestCase):
    def test_save_restore_and_checksum(self):
        accelerator = Accelerator(cpu=True)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        config = {"manifest_sha256": {"train.jsonl": "fixture"}}
        expected = model.weight.detach().clone()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "latest.pt"
            save_checkpoint(path, model, optimizer, scheduler, accelerator,
                            {"config": config, "epoch": 1, "global_step": 1})
            with torch.no_grad():
                model.weight.zero_()
            checkpoint = restore_checkpoint(path, model, optimizer, scheduler, accelerator, config)
            self.assertEqual(checkpoint["global_step"], 1)
            self.assertEqual(scheduler.last_epoch, 1)
            torch.testing.assert_close(model.weight, expected)
            with path.open("ab") as stream:
                stream.write(b"bad")
            with self.assertRaisesRegex(ValueError, "checksum"):
                restore_checkpoint(path, model, optimizer, scheduler, accelerator, config)
