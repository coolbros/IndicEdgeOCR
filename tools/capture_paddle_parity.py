"""Capture a new deterministic, labeled Paddle parity fixture."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
import paddle
import yaml

from paddle_reference import CONFIG, PIN, capture, git, load_verified, make_operators, sha256


def run(args):
    source, output = args.source.resolve(), args.output.resolve()
    checkpoint, dataset = args.checkpoint.resolve(), args.dataset.resolve()
    manifest = dataset / "train.jsonl"
    audit = json.loads((dataset / "audit.json").read_text(encoding="utf-8"))
    if not audit["passed"] or sha256(manifest) != audit["manifest_sha256"][manifest.name]:
        raise ValueError("Expected the audited training manifest")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    if git(source, "rev-parse", "HEAD") != PIN or git(source, "status", "--porcelain"):
        raise ValueError("Unexpected upstream checkout")
    os.chdir(source)
    sys.path.insert(0, str(source))
    from ppocr.data import transform
    from ppocr.losses import build_loss
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process

    paddle.set_device(args.device)
    paddle.seed(2026)
    np.random.seed(2026)
    config = yaml.safe_load(Path(CONFIG).read_text(encoding="utf-8"))
    decoder = build_post_process(config["PostProcess"], config["Global"])
    classes = len(decoder.character)
    config["Architecture"]["Head"]["out_channels_list"] = {
        "CTCLabelDecode": classes, "SARLabelDecode": classes + 2, "NRTRLabelDecode": classes + 3
    }
    model = build_model(config["Architecture"])
    load_verified(config, model, checkpoint)
    arrays, shapes = {}, {}
    traces = {}

    def save(name):
        def hook(layer, inputs, output):
            if isinstance(output, paddle.Tensor):
                if ".rep_dw" in name and name.endswith(".bn") and ".conv." not in name:
                    traces[name + "/sum_before_bn"] = inputs[0].numpy()
                traces[name] = output.numpy()
        return hook

    def selected(name):
        return ".rep_dw" in name or name.startswith("head.gtc_head.")

    handles = [layer.register_forward_post_hook(save(name))
               for name, layer in model.named_sublayers() if selected(name)]
    train_ops = make_operators(config, inference=False)
    infer_ops = make_operators(config, inference=True)
    batch = []
    samples = {}
    model.eval()
    with paddle.no_grad():
        for index, row in enumerate(rows):
            path = dataset / row["image_path"]
            if row["split"] != "train" or sha256(path) != row["image_sha256"]:
                raise ValueError("Invalid training source or checksum")
            content = path.read_bytes()
            batch.append(transform({"image": content, "label": row["transcription"]}, train_ops))
            key = f"real_{index}"
            prefix = f"eval/{key}"
            inputs = transform({"image": content}, infer_ops)[0][None]
            arrays[f"{prefix}/input"] = inputs
            traces.clear()
            with capture(model, arrays, prefix, shapes.setdefault(prefix, {})):
                probabilities = model(paddle.to_tensor(inputs))
            arrays.update({f"{prefix}/{name}": value for name, value in traces.items()})
            arrays[f"{prefix}/ctc_logits"] = arrays.pop(f"{prefix}/head.ctc_head.fc")
            arrays[f"{prefix}/ctc_probabilities"] = probabilities.numpy()
            text, confidence = decoder(probabilities)[0]
            samples[key] = {**row, "text": text, "confidence": confidence}

        tensors = [paddle.to_tensor(np.stack(items)) for items in zip(*batch)]
        keys = ("input", "label_ctc", "label_gtc", "length", "valid_ratio")
        arrays.update({f"train/{key}": tensor.numpy() for key, tensor in zip(keys, tensors)})
        model.train()
        for layer in model.sublayers():
            if isinstance(layer, paddle.nn.Dropout):
                layer.eval()
        bn_layers = [layer for layer in model.sublayers() if isinstance(layer, paddle.nn.BatchNorm2D)]
        assert all(layer.training for layer in bn_layers)
        traces.clear()
        with capture(model, arrays, "train", shapes.setdefault("train", {})):
            outputs = model(tensors[0], data=tensors[1:])
        arrays.update({f"train/{name}": value for name, value in traces.items()})
        arrays["train/ctc_logits"] = outputs["ctc"].numpy()
        arrays["train/nrtr_logits"] = outputs["gtc"].numpy()
        losses = build_loss(deepcopy(config["Loss"]))(outputs, tensors)
        arrays.update({f"train/loss/{key}": value.numpy() for key, value in losses.items()})
        arrays.update({f"train/state/{name}": value.numpy()
                       for name, value in model.state_dict().items()
                       if name.endswith(("._mean", "._variance"))})
    for handle in handles:
        handle.remove()
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "real_parity.npz"
    np.savez_compressed(archive, **arrays)
    metadata = {"commit": PIN, "paddle": paddle.__version__, "device": args.device,
                "fixture_sha256": sha256(archive), "samples": samples,
                "train_manifest_sha256": sha256(manifest), "dropout": "disabled",
                "batch_norm": "training", "shapes": shapes,
                "preprocessing": {"eval": "RecResizeImg infer_mode=true",
                                  "train": "Eval YAML RecResizeImg plus MultiLabelEncode",
                                  "image_shape": [3, 48, 320], "channel_order": "BGR"}}
    (output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                         encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "checkpoint", "dataset", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())
