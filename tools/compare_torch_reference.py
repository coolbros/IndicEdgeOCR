"""Compare immutable upstream Paddle fixtures with the native PyTorch model."""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch

from convert_paddle_checkpoint import digest, load_official, mapping
from ppocrv6_torch import PPOCRv6Medium
from ppocrv6_torch.data import Vocabulary, preprocess
from ppocrv6_torch.losses import multi_loss


TOLERANCE = {
    "input": (0.0, 0.0),
    "activation": (1e-4, 1e-5),
    "logits": (1e-4, 1e-5),
    "probability": (1e-4, 1e-6),
    "loss": (1e-4, 1e-5),
}


def compare(expected, actual, category):
    rtol, atol = TOLERANCE[category]
    result = {"expected_shape": list(expected.shape), "actual_shape": list(actual.shape),
              "expected_dtype": str(expected.dtype), "actual_dtype": str(actual.dtype),
              "rtol": rtol, "atol": atol, "pass": False}
    if expected.shape != actual.shape:
        return result
    a, b = expected.astype(np.float64).ravel(), actual.astype(np.float64).ravel()
    delta = np.abs(a - b)
    a_norm = np.sqrt(np.sum(a * a))
    norm = a_norm * np.sqrt(np.sum(b * b))
    result.update(max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                  relative_l2=float(np.sqrt(np.sum(delta * delta)) / a_norm) if a_norm else None,
                  cosine=float(np.sum(a * b) / norm) if norm else None,
                  pass_count=int((delta <= atol + rtol * np.abs(a)).sum()),
                  elements=int(a.size))
    result["pass"] = expected.dtype == actual.dtype and result["pass_count"] == a.size
    return result


@torch.no_grad()
def run(args):
    original = load_official(args.original)
    manifest = json.loads(args.archive.with_name("manifest.json").read_text(encoding="utf-8"))
    if digest(args.archive) != manifest["fixture_sha256"]:
        raise ValueError("Reference archive checksum mismatch")
    model = PPOCRv6Medium().eval()
    state = load_file(args.checkpoint)
    model.load_state_dict(state, strict=True)
    for source, target, transpose in mapping(model):
        np.testing.assert_array_equal(
            original[source].T if transpose else original[source], state[target].numpy(),
            err_msg=target,
        )
    del original, state
    model.to(args.device)
    vocab = Vocabulary(args.dictionary)
    report = {"archive_sha256": digest(args.archive), "checkpoint_sha256": digest(args.checkpoint),
              "exact_checkpoint_tensors": 441, "torch": torch.__version__,
              "cuda": torch.version.cuda, "device": args.device, "layers": {}, "decoding": {}}
    captured = {}
    dataset = args.dataset

    def record(name):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                captured[name] = output.detach().cpu().numpy()
                if ".rep_dw" in name and name.endswith(".bn") and ".conv." not in name:
                    captured[name + "/sum_before_bn"] = _inputs[0].detach().cpu().numpy()
        return hook

    with np.load(args.archive, allow_pickle=False) as archive:
        for prefix, info in manifest["samples"].items():
            base = "eval/" + prefix
            keys = {name.removeprefix(base + "/") for name in archive.files
                    if name.startswith(base + "/")}
            handles = [module.register_forward_hook(record(name))
                       for name, module in model.named_modules() if name in keys
                       or name == "head.ctc_head.fc"]
            captured.clear()
            x = archive[f"{base}/input"]
            p = model(torch.from_numpy(x).to(args.device)).cpu().numpy()
            for handle in handles:
                handle.remove()
            captured["ctc_logits"] = captured.pop("head.ctc_head.fc")
            captured["ctc_probabilities"] = p
            if prefix == "official":
                captured["input"] = preprocess(args.sample.read_bytes(), inference=True)[0][None]
            elif dataset is not None:
                captured["input"] = preprocess(
                    (dataset / info["image_path"]).read_bytes(), inference=True
                )[0][None]
            for name in keys - captured.keys() - ({"input"} if dataset is None else set()):
                report["layers"][f"{base}/{name}"] = {"pass": False, "error": "missing capture"}
            for name in sorted(keys & captured.keys()):
                category = {"input": "input", "ctc_logits": "logits",
                            "ctc_probabilities": "probability"}.get(name, "activation")
                report["layers"][f"{base}/{name}"] = compare(
                    archive[f"{base}/{name}"], captured[name], category)
            text, probability = vocab.decode(p)[0]
            error = abs(info["confidence"] - probability)
            report["decoding"][prefix] = {"expected": info["text"], "actual": text,
                                          "confidence_error": error,
                                          "pass": text == info["text"] and error <= 1e-5}
        if dataset is not None:
            captured.clear()
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.eval()
            from ppocrv6_torch.backbone import BatchNorm2d
            assert all(module.training for module in model.modules() if isinstance(module, BatchNorm2d))
            keys = {name[6:] for name in archive.files if name.startswith("train/")}
            handles = [module.register_forward_hook(record(name))
                       for name, module in model.named_modules() if name in keys]
            preprocessed = [preprocess((dataset / row["image_path"]).read_bytes())
                            for row in manifest["samples"].values()]
            encoded = [vocab.encode(row["transcription"]) for row in manifest["samples"].values()]
            for key, values in (
                ("input", [x[0] for x in preprocessed]),
                ("valid_ratio", [x[1] for x in preprocessed]),
                *zip(("label_ctc", "label_gtc", "length"), zip(*encoded)),
            ):
                captured[key] = np.stack(values)
            batch = [torch.from_numpy(archive[f"train/{key}"]).to(args.device)
                     for key in ("input", "label_ctc", "label_gtc", "length", "valid_ratio")]
            outputs = model(batch[0], batch[1:])
            for handle in handles:
                handle.remove()
            captured.update(ctc_logits=outputs["ctc"].cpu().numpy(),
                            nrtr_logits=outputs["gtc"].cpu().numpy())
            for name, loss in multi_loss(outputs, *batch[1:4]).items():
                captured[f"loss/{name}"] = loss.cpu().numpy()
            for source, target, _ in mapping(model):
                if source.endswith(("._mean", "._variance")):
                    captured[f"state/{source}"] = model.state_dict()[target].cpu().numpy()
            for name in sorted(keys):
                if name not in captured:
                    report["layers"][f"train/{name}"] = {"pass": False, "error": "missing capture"}
                    continue
                kind = ("loss" if name.startswith("loss/") else "logits" if name.endswith("_logits")
                        else "input" if name in ("input", "valid_ratio", "length", "label_ctc", "label_gtc")
                        else "activation")
                report["layers"][f"train/{name}"] = compare(
                    archive[f"train/{name}"], captured[name], kind)
            modules = dict(model.named_modules())
            for key in sorted(archive.files):
                if key.startswith("probes/") and key.endswith("/output"):
                    path = key.removeprefix("probes/").removesuffix("/output")
                    source = torch.from_numpy(archive[f"probes/{path}/input"]).to(args.device)
                    report["layers"][key] = compare(archive[key], modules[path](source).cpu().numpy(),
                                                   "activation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    all_rows = [*report["layers"].values(), *report["decoding"].values()]
    failures = sum(not row["pass"] for row in all_rows)
    print(f"{len(all_rows) - failures}/{len(all_rows)} comparisons passed")
    return failures == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=Path("artifacts/paddle/reference.npz"))
    parser.add_argument("--sample", type=Path, default=Path("references/samples/general_ocr_rec_001.png"))
    parser.add_argument("--dictionary", type=Path,
                        default=Path("references/PaddleOCR/ppocr/utils/dict/ppocrv6_dict.txt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    raise SystemExit(0 if run(args) else 1)
