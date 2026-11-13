"""Evaluate a frozen best checkpoint against the verified test manifest."""

import argparse
import json
from pathlib import Path

from accelerate import Accelerator
import torch

from convert_paddle_checkpoint import digest
from ppocrv6_torch import PPOCRv6Medium
from ppocrv6_torch.data import Vocabulary
from train_smoke import prepared_records, validate
from verify_dataset_splits import SPLITS, audit


def run(args):
    audited = audit([args.dataset / f"{s}.jsonl" for s in SPLITS], args.dataset)
    metadata = json.loads(args.checkpoint.with_suffix(".json").read_text())
    if (args.checkpoint.name != "best.pt" or not audited["passed"]
            or audited["manifest_sha256"] != metadata["config"]["manifest_sha256"]
            or digest(args.checkpoint) != metadata["sha256"]
            or digest(args.dictionary) != metadata["config"]["dictionary_sha256"]):
        raise ValueError("Frozen checkpoint or split checksum failed verification")
    accelerator = Accelerator()
    model = PPOCRv6Medium()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                     weights_only=False)["model"])
    model.to(accelerator.device)
    vocab = Vocabulary(args.dictionary)
    cache = args.dataset.parent / "preprocessed-cache"
    cache.mkdir(exist_ok=True)
    records = list(prepared_records(args.dataset, "test", vocab, cache))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"split": "test", "samples": len(records),
                   "exact_match": validate(model, records, vocab, accelerator.device),
                   "sha256": metadata["sha256"],
                   "test_manifest_sha256": audited["manifest_sha256"]["test.jsonl"]}, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "dictionary", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    torch.set_num_threads(2)
    run(parser.parse_args())
