"""Convert the verified official PP-OCRv6 medium checkpoint to safetensors."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from safetensors.torch import save_file
import torch
from torch import nn

from ppocrv6_torch import PPOCRv6Medium


OFFICIAL_SHA256 = "c4a110423c23739f157ff870e27f41649d8c89e4c8df5230c2cd2e9fd90a7e92"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_official(path):
    if digest(path) != OFFICIAL_SHA256:
        raise ValueError("Only the SHA-256-pinned official checkpoint can be deserialized")
    with Path(path).open("rb") as stream:
        state = pickle.load(stream)
    del state["StructuredToParameterName@@"]
    return state


def mapping(model):
    modules = dict(model.named_modules())
    for target in model.state_dict():
        parent, leaf = target.rsplit(".", 1)
        source_leaf = {"running_mean": "_mean", "running_var": "_variance"}.get(leaf, leaf)
        transpose = leaf == "weight" and isinstance(modules[parent], nn.Linear)
        yield f"{parent}.{source_leaf}", target, transpose


def convert(path, output):
    original = load_official(path)
    model = PPOCRv6Medium()
    target = model.state_dict()
    state = {}
    rows = []
    used = set()
    for source, destination, transpose in mapping(model):
        used.add(source)
        array = original.get(source)
        expected = target[destination]
        status = "matched"
        if array is None:
            status = "missing"
        else:
            converted = array.T if transpose else array
            if converted.shape != tuple(expected.shape) or converted.dtype != np.float32:
                status = "shape_or_dtype_mismatch"
            else:
                state[destination] = torch.from_numpy(np.ascontiguousarray(converted))
        rows.append({
            "source": source, "target": destination, "status": status,
            "source_shape": list(array.shape) if array is not None else None,
            "target_shape": list(expected.shape),
            "transform": "transpose_2d" if transpose else "identity",
        })
    rows.extend({"source": key, "status": "unexpected"} for key in sorted(original.keys() - used))
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"source_sha256": OFFICIAL_SHA256, "rows": rows}
    if all(row["status"] == "matched" for row in rows):
        model.load_state_dict(state, strict=True)
        save_file(state, output)
        manifest["checkpoint_sha256"] = digest(output)
    output.with_suffix(".conversion.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if "checkpoint_sha256" not in manifest:
        raise ValueError("Checkpoint mapping failed. See the conversion manifest.")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = convert(args.checkpoint, args.output)
    print(f"Converted {len(result['rows'])} tensors: {result['checkpoint_sha256']}")
