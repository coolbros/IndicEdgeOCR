from collections import Counter
from pathlib import Path
import sys
import unittest

import torch

from ppocrv6_torch import PPOCRv6Medium

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from convert_paddle_checkpoint import mapping


class ConversionMappingTest(unittest.TestCase):
    def test_complete_bijective_mapping(self):
        torch.set_num_threads(2)
        model = PPOCRv6Medium()
        rows = list(mapping(model))
        self.assertEqual(len({source for source, _, _ in rows}), 441)
        self.assertEqual({target for _, target, _ in rows}, set(model.state_dict()))
        self.assertEqual(sum(transpose for _, _, transpose in rows), 39)
        self.assertEqual(Counter(source.rsplit(".", 1)[1] for source, _, _ in rows)["_mean"], 61)
