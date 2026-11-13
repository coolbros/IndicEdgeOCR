"""Capture PP-OCRv6 medium fixtures using an unmodified PaddleOCR checkout."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import cv2
import numpy as np
import paddle
import yaml


CONFIG = "configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml"
PIN = "dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf"
CHECKPOINT_SHA256 = "c4a110423c23739f157ff870e27f41649d8c89e4c8df5230c2cd2e9fd90a7e92"


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_inventory(model: paddle.nn.Layer) -> dict:
    return {
        path or "<model>": {
            "type": type(layer).__name__,
            "parameters": sum(math.prod(p.shape) for p in layer.parameters()),
            "trainable": sum(
                math.prod(p.shape) for p in layer.parameters() if not p.stop_gradient
            ),
            "own_parameters": {
                name: list(p.shape)
                for name, p in layer.named_parameters(include_sublayers=False)
            },
        }
        for path, layer in [("", model), *model.named_sublayers()]
    }


def load_verified(config: dict, model: paddle.nn.Layer, path: Path) -> dict:
    from ppocr.utils.save_load import load_model

    digest = sha256(path)
    if digest != CHECKPOINT_SHA256:
        raise ValueError(f"Unexpected official checkpoint SHA-256: {digest}")
    original = paddle.load(str(path))
    state = model.state_dict()
    missing = sorted(state.keys() - original.keys())
    unexpected = sorted(original.keys() - state.keys())
    mismatched = sorted(
        key for key in state.keys() & original.keys()
        if list(state[key].shape) != list(original[key].shape)
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}, "
            f"shape_mismatch={mismatched}"
        )
    config["Global"]["pretrained_model"] = str(path)
    load_model(config, model, model_type="rec")
    for key, value in model.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), original[key].astype(value.dtype).numpy())
    return {
        "sha256": digest,
        "tensors": len(state),
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
    }


@contextmanager
def capture(model, arrays, prefix, shapes):
    handles = []

    def hook(path):
        def save(layer, inputs, output):
            if not isinstance(output, paddle.Tensor):
                return
            shapes[path] = {
                "inputs": [list(t.shape) for t in inputs if isinstance(t, paddle.Tensor)],
                "output": list(output.shape),
            }
            if path.startswith("backbone") or path.startswith("head.ctc_encoder"):
                # Capture the full blocks and their main paths, plus the detailed shape trace.
                if path.count(".") <= 2 or path.startswith("head.ctc_encoder.encoder."):
                    arrays[f"{prefix}/{path}"] = output.numpy()
            if path in {"head.before_gtc", "head.ctc_head.fc", "head.gtc_head"} or (
                path.startswith("head.gtc_head.decoder.") and path.count(".") == 3
            ):
                arrays[f"{prefix}/{path}"] = output.numpy()
        return save

    try:
        for name, layer in model.named_sublayers():
            handles.append(layer.register_forward_post_hook(hook(name)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def make_operators(config: dict, *, inference: bool):
    from ppocr.data import create_operators

    operators = deepcopy(config["Eval"]["dataset"]["transforms"])
    options = deepcopy(config["Global"])
    if inference:
        operators = [op for op in operators if "Label" not in next(iter(op))]
        for op in operators:
            if "KeepKeys" in op:
                op["KeepKeys"]["keep_keys"] = ["image"]
        options["infer_mode"] = True
    return create_operators(operators, options)


def synthetic_images(seed: int) -> dict[str, tuple[bytes, str]]:
    rng = np.random.default_rng(seed)
    images = {}
    for index, (height, width, text) in enumerate([(64, 250, "AB123"), (56, 426, "XYZ")]):
        image = rng.integers(235, 256, (height, width, 3), dtype=np.uint8)
        cv2.putText(image, text, (10, height - 13), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (40, 25, 15), 2, cv2.LINE_AA)
        ok, buffer = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Could not encode synthetic PNG")
        images[f"synthetic_{index}"] = (buffer.tobytes(), text)
    return images


def run(source: Path, checkpoint: Path, sample: Path, output: Path, device: str, seed: int):
    if git(source, "rev-parse", "HEAD") != PIN or git(source, "status", "--porcelain"):
        raise ValueError(f"Expected a clean PaddleOCR checkout at {PIN}")
    sys.path.insert(0, str(source))
    os.chdir(source)
    from ppocr.data import transform
    from ppocr.losses import build_loss
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process

    paddle.set_device(device)
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    config = yaml.safe_load(Path(CONFIG).read_text(encoding="utf-8"))
    config["Global"]["use_gpu"] = device.startswith("gpu")
    decoder = build_post_process(config["PostProcess"], config["Global"])
    classes = len(decoder.character)
    config["Architecture"]["Head"]["out_channels_list"] = {
        "CTCLabelDecode": classes,
        "SARLabelDecode": classes + 2,
        "NRTRLabelDecode": classes + 3,
    }
    model = build_model(config["Architecture"])
    checkpoint_info = load_verified(config, model, checkpoint)
    inventory = model_inventory(model)
    images = synthetic_images(seed)
    official = sample.read_bytes()
    arrays = {}
    shapes = {}
    samples = {}
    model.eval()
    infer_ops = make_operators(config, inference=True)
    with paddle.no_grad():
        for name, content in {"official": official, **{
            name: image for name, (image, _) in images.items()
        }}.items():
            prefix = f"eval/{name}"
            inputs = transform({"image": content}, infer_ops)[0][None, ...]
            arrays[f"{prefix}/input"] = inputs
            with capture(model, arrays, prefix, shapes.setdefault(prefix, {})):
                probabilities = model(paddle.to_tensor(inputs))
            arrays[f"{prefix}/ctc_logits"] = arrays.pop(f"{prefix}/head.ctc_head.fc")
            arrays[f"{prefix}/ctc_probabilities"] = probabilities.numpy()
            text, confidence = decoder(probabilities)[0]
            samples[name] = {"text": text, "confidence": confidence,
                             "image_sha256": hashlib.sha256(content).hexdigest()}

        labeled_ops = make_operators(config, inference=False)
        rows = [transform({"image": image, "label": text}, labeled_ops)
                for image, text in images.values()]
        keys = ["input", "label_ctc", "label_gtc", "length", "valid_ratio"]
        batch = [paddle.to_tensor(np.stack(items)) for items in zip(*rows)]
        arrays.update({f"train/{key}": value.numpy() for key, value in zip(keys, batch)})
        model.train()
        with capture(model, arrays, "train", shapes.setdefault("train", {})):
            outputs = model(batch[0], data=batch[1:])
        arrays["train/ctc_logits"] = outputs["ctc"].numpy()
        arrays["train/nrtr_logits"] = outputs["gtc"].numpy()
        loss = build_loss(config["Loss"])(outputs, batch)
        arrays.update({f"train/loss/{key}": value.numpy() for key, value in loss.items()})

    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite fixture: {name}")
    if not samples["official"]["text"]:
        raise ValueError("Official example produced an empty transcription")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "reference.npz", **arrays)
    manifest = {
        "commit": PIN, "paddle": paddle.__version__, "opencv": cv2.__version__,
        "numpy": np.__version__, "device": device, "seed": seed,
        "gpu_count": paddle.device.cuda.device_count() if device.startswith("gpu") else 0,
        "ctc_classes": classes, "nrtr_classes": model.head.gtc_head.out_channels,
        "checkpoint": checkpoint_info, "samples": samples,
        "losses": {key: float(value.item()) for key, value in loss.items()},
        "shapes": shapes, "layers": inventory,
        "fixture_sha256": sha256(output / "reference.npz"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("device", "gpu_count", "samples", "losses")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("references/PaddleOCR"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/paddle"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    run(args.source.resolve(), args.checkpoint.resolve(), args.sample.resolve(),
        args.output.resolve(), args.device, args.seed)
