"""Capture and replay official Paddle operation boundaries with the fixed checkpoint."""

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import os
import re
import sys
import zipfile

import numpy as np

from convert_paddle_checkpoint import digest
from compare_torch_reference import TOLERANCE


BASELINE = {
    "archive": "e861cbf75fe498d05e4edc0fa568a09cabe88f6224b2e38427ee557df65af8fc",
    "original": "c4a110423c23739f157ff870e27f41649d8c89e4c8df5230c2cd2e9fd90a7e92",
    "checkpoint": "0afc351e31cea4f6d670e68652bd22a11ab44273584e156159dbdb83fafb02ef",
    "train_manifest": "4fc83fe5c90366051e75d2f10ef61624a68dbec3785e263a1bcd14042cdfed73",
}


def metrics(expected, actual, category):
    rtol, atol = TOLERANCE[category]
    row = {"expected_shape": list(expected.shape), "actual_shape": list(actual.shape),
           "expected_dtype": str(expected.dtype), "actual_dtype": str(actual.dtype),
           "rtol": rtol, "atol": atol, "pass": False}
    if expected.shape != actual.shape:
        return row
    a, b = expected.astype(np.float64), actual.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    same_infinity = np.isinf(a) & (a == b)
    errors = np.abs(a[finite] - b[finite])
    counts = (errors <= atol + rtol * np.abs(a[finite])).sum() + same_infinity.sum()
    a_finite, b_finite = a[finite], b[finite]
    denominator = np.sqrt(np.sum(a_finite * a_finite))
    norm = denominator * np.sqrt(np.sum(b_finite * b_finite))
    percentiles = np.percentile(errors, [50, 90, 99, 99.9]) if errors.size else [0.] * 4
    row.update(pass_count=int(counts), elements=int(a.size),
               nonfinite_mismatch=int((~(finite | same_infinity)).sum()),
               max_abs=float(errors.max()) if errors.size else 0.,
               mean_abs=float(errors.mean()) if errors.size else 0.,
               relative_l2=float(np.sqrt(np.sum(errors * errors)) / denominator) if denominator else None,
               cosine=float(np.sum(a_finite * b_finite) / norm) if norm else None,
               percentile_abs=dict(zip(("50", "90", "99", "99.9"), map(float, percentiles))))
    row["pass"] = bool(counts == a.size and expected.dtype == actual.dtype)
    return row


def attention_op(name, inputs, softmax, paddle_backend=False):
    if name == "qk":
        q, k = inputs
        return q @ (k.transpose([0, 1, 3, 2]) if paddle_backend else k.transpose(-1, -2))
    if name == "scale":
        return inputs[0] * 32**-0.5
    if name == "mask":
        return inputs[0] + inputs[1]
    if name == "softmax":
        return softmax(inputs[0])
    if name == "av":
        return inputs[0] @ inputs[1]
    x = inputs[0]
    x = x.transpose([0, 2, 1, 3]) if paddle_backend else x.transpose(1, 2)
    return x.reshape([x.shape[0], -1, 512])


def attention_steps(q, k, v, mask, softmax, paddle_backend=False):
    steps = {}

    def step(name, *values):
        result = attention_op(name, values, softmax, paddle_backend)
        steps[name] = (values, result)
        return result

    scores = step("scale", step("qk", q, k))
    if mask is not None:
        scores = step("mask", scores, mask)
    return_value = step("av", step("softmax", scores), v)
    step("merge", return_value)
    return steps


def selected(name):
    return (name.startswith(("backbone", "head.ctc", "head.before_gtc", "head.gtc_head"))
            and name != "head.gtc_head.decoder")


def unpack_attention(module, values, kwargs, paddle_backend):
    q = values[0]
    memory = values[1] if len(values) > 1 else kwargs.get("key" if paddle_backend else "memory")
    mask = kwargs.get("attn_mask" if paddle_backend else "mask")

    def split(x, count):
        x = x.reshape([x.shape[0], x.shape[1], count, 16, 32])
        x = x.transpose([2, 0, 3, 1, 4]) if paddle_backend else x.permute(2, 0, 3, 1, 4)
        return tuple(x[i] for i in range(count))

    if module.self_attn:
        q, k, v = split(module.qkv(q), 3)
    else:
        q, = split(module.q(q), 1)
        k, v = split(module.kv(memory), 2)
    return q, k, v, mask


class Recorder:
    """Write named calls in execution order. Each array is independently readable."""

    def __init__(self, archive):
        self.archive = archive
        self.keys = set()
        self.rows = []
        self.counts = Counter()

    def save(self, mode, name, inputs, kwargs, output, category="activation"):
        key = f"{mode}/{name}"
        index = self.counts[key]
        self.counts[key] += 1
        prefix = f"{key}/{index}"
        arguments = []
        for label, value in [*(enumerate(inputs)), *kwargs.items()]:
            if value is None:
                arguments.append([label, None])
            else:
                path = f"{prefix}/arg_{label}"
                self.array(path, value)
                arguments.append([label, path])
        self.array(prefix + "/output", output)
        self.rows.append({"mode": mode, "name": name, "index": index, "arguments": arguments,
                          "output": prefix + "/output", "category": category})

    def array(self, key, value):
        if self.archive is None:
            return
        if key in self.keys:
            raise ValueError(f"Duplicate boundary array: {key}")
        self.keys.add(key)
        with self.archive.open(key + ".npy", "w", force_zip64=True) as stream:
            np.lib.format.write_array(stream, value.numpy(), allow_pickle=False)


def attention_boundaries(named, first, last, save, softmax, paddle_backend=False):
    for idx in range(4):
        root = f"head.gtc_head.decoder.{idx}"
        for kind in ("self_attn", "cross_attn"):
            name = root + "." + kind
            q = first[name][0]
            if paddle_backend:
                mask = named["head.gtc_head"].generate_square_subsequent_mask(q.shape[1])
            else:
                import torch
                mask = torch.full((q.shape[1],) * 2, -torch.inf, dtype=q.dtype).triu(1)[None, None]
            if kind == "self_attn":
                kwargs = {"attn_mask" if paddle_backend else "mask": mask}
            else:
                kwargs = {"key" if paddle_backend else "memory": last["head.before_gtc"]}
            steps = attention_steps(*unpack_attention(named[name], (q,), kwargs, paddle_backend),
                                    softmax, paddle_backend=paddle_backend)
            np.testing.assert_array_equal(steps["merge"][1].numpy(),
                                          first[name + ".out_proj"][0].numpy())
            for operation, (inputs, value) in steps.items():
                save(name + "::" + operation, inputs, value,
                     "probability" if operation == "softmax" else "activation")


def functional_boundaries(mode, named, first, last, record, log_softmax, paddle_backend=False):
    """Record functional inputs/outputs using adjacent observed module boundaries."""
    stem = "backbone.conv1"
    left, right = last[stem + ".pool"], last[stem + ".stem2b"]
    record(stem + "::concat", (left, right), first[stem + ".stem3"][0])
    for name in named:
        if re.fullmatch(r"backbone\.blocks[2-5]\.\d+", name):
            rep = name + ".token_mixer.rep_dw"
            if rep + ".bn" in first:
                identity = first[rep][0]
                record(rep + "::identity", (identity,), identity)
                record(rep + "::sum", (last[rep + ".conv"], last[rep + ".conv1"], identity),
                       first[rep + ".bn"][0])
                record(name + "::residual", (last[name + ".token_mixer"],
                                               last[name + ".channel_mixer"]), last[name])
    features = last["backbone.blocks6"]
    record("backbone::" + ("adaptive" if mode == "train" else "average"),
           (features,), last["backbone"])
    neck = "head.ctc_encoder.encoder"
    reduced, local = last[neck + ".conv_reduce"], last[neck + ".local_conv"]
    shape = tuple(reduced.shape)

    def to_grid(value):
        transposed = value.transpose([0, 2, 1]) if paddle_backend else value.transpose(1, 2)
        return transposed.reshape(shape)

    record(neck + "::sum", (reduced, local), to_grid(first[neck + ".svtr_block.0"][0]))
    record(neck + "::residual", (to_grid(last[neck + ".norm"]), last[neck + ".skip_conv"]),
           last[neck])
    logits = last["head.ctc_head.fc"]
    if mode == "eval":
        record("ctc::softmax", (logits,), last["head.ctc_head"], "probability")
        return
    record("ctc::log_softmax", (logits,), log_softmax(logits), "probability")
    logits = last["head.gtc_head"]
    record("nrtr::log_softmax", (logits,), log_softmax(logits), "probability")
    x = last["head.before_gtc.0"]
    record("head.before_gtc::transpose", (x,), first["head.before_gtc.1.fc"][0], "input")
    for idx in range(4):
        name = f"head.gtc_head.decoder.{idx}"
        for stage, source in (
            (1, first[name][0]), (2, last[name + ".norm1"]), (3, last[name + ".norm2"]),
        ):
            record(f"{name}::residual{stage}", (source, last[f"{name}.dropout{stage}"]),
                   first[f"{name}.norm{stage}"][0])


def register_paddle(model, recorder, mode):
    import paddle

    first, last, keywords = {}, {}, {}
    named = dict(model.named_sublayers())

    def hook(name):
        def save(layer, values, result):
            if not isinstance(result, paddle.Tensor):
                return
            first[name], last[name] = values, result
            kwargs = keywords.get(name, {})
            if name == "head.gtc_head":
                values = (values[0], *values[1][:2])
            elif name == "head.ctc_head":
                values = values[:1]
            category = "logits" if name in {"head.gtc_head", "head.gtc_head.tgt_word_prj",
                                         "head.ctc_head.fc"} else "activation"
            recorder.save(mode, name, values, kwargs, result, category)
            if mode == "train" and isinstance(layer, paddle.nn.BatchNorm2D):
                prefix = recorder.rows[-1]["output"]
                recorder.array(prefix + "/running_mean", layer._mean)
                recorder.array(prefix + "/running_var", layer._variance)
        return save

    def pre(name):
        def remember(_layer, _values, kwargs):
            keywords[name] = dict(kwargs)
        return remember

    handles = []
    for name, layer in named.items():
        if selected(name):
            handles.append(layer.register_forward_pre_hook(pre(name), with_kwargs=True))
            handles.append(layer.register_forward_post_hook(hook(name)))
    return handles, named, first, last


def capture(args):
    import paddle
    import yaml
    from paddle_reference import CONFIG, PIN, git, load_verified

    args.output.mkdir(parents=True, exist_ok=False)
    if git(args.source, "rev-parse", "HEAD") != PIN or git(args.source, "status", "--porcelain"):
        raise ValueError("Expected the unmodified pinned PaddleOCR checkout")
    sys.path.insert(0, str(args.source))
    os.chdir(args.source)
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process

    config = yaml.safe_load(Path(CONFIG).read_text(encoding="utf-8"))
    vocab = len(build_post_process(config["PostProcess"], config["Global"]).character)
    config["Architecture"]["Head"]["out_channels_list"] = {
        "CTCLabelDecode": vocab, "SARLabelDecode": vocab + 2, "NRTRLabelDecode": vocab + 3,
    }
    paddle.set_device("cpu")
    model = build_model(config["Architecture"])
    load_verified(config, model, args.original)
    output = args.output / "boundaries.npz"
    reproducibility = {}
    with np.load(args.archive, allow_pickle=False) as baseline, paddle.no_grad(), \
            zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as archive:
        recorder = Recorder(archive)
        batch = [paddle.to_tensor(baseline[f"train/{key}"]) for key in
                 ("input", "label_ctc", "label_gtc", "length", "valid_ratio")]
        for mode in ("eval", "train"):
            model.train() if mode == "train" else model.eval()
            for layer in model.sublayers():
                if isinstance(layer, paddle.nn.Dropout):
                    layer.eval()
            handles, named, first, last = register_paddle(model, recorder, mode)
            if mode == "train":
                result = model(batch[0], data=batch[1:])["gtc"]
                expected = baseline["train/nrtr_logits"]
            else:
                result = model(paddle.to_tensor(baseline["eval/real_0/input"]))
                expected = baseline["eval/real_0/ctc_probabilities"]
            for handle in handles:
                handle.remove()
            reproducibility[mode] = metrics(expected, result.numpy(),
                                            "logits" if mode == "train" else "probability")

            def record(name, inputs, result, category="activation"):
                recorder.save(mode, name, inputs, {}, result, category)

            f = paddle.nn.functional
            functional_boundaries(mode, named, first, last, record, f.log_softmax, True)
            if mode == "train":
                attention_boundaries(named, first, last, record, f.softmax, True)
        rows = recorder.rows
    (args.output / "boundaries.json").write_text(json.dumps({
        "archive_sha256": digest(output), "baseline": BASELINE, "paddle": paddle.__version__,
        "paddle_commit": PIN, "original_output_reproduction": reproducibility, "rows": rows,
    }, indent=2), encoding="utf-8")


def functional_torch(name, values):
    import torch
    from torch.nn import functional as f

    if name == "concat":
        return torch.cat(values, dim=1)
    if name == "identity":
        return values[0]
    if name.startswith("residual") or name == "sum":
        result = values[0]
        for value in values[1:]:
            result = result + value
        return result
    if name == "transpose":
        return values[0].transpose(1, 2)
    if name == "average":
        return f.avg_pool2d(values[0], (3, 2))
    if name == "adaptive":
        return f.adaptive_avg_pool2d(values[0], (1, 40))
    if name == "log_softmax":
        return values[0].log_softmax(-1)
    return attention_op(name, values, lambda value: value.softmax(-1))


@contextmanager
def capture_torch(model, calls):
    import torch

    handles = []
    counts = Counter()

    def hook(name):
        def save(_layer, values, result):
            if isinstance(result, torch.Tensor):
                index = counts[name]
                counts[name] += 1
                calls[name, index] = (values, result)
        return save

    for name, module in model.named_modules():
        if selected(name):
            handles.append(module.register_forward_hook(hook(name)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def normalize_kwargs(kwargs):
    for previous, current in (("key", "memory"), ("attn_mask", "mask"), ("self_mask", "mask")):
        if previous in kwargs:
            kwargs[current] = kwargs.pop(previous)
    if kwargs.pop("cross_mask", None) is not None:
        raise ValueError("The medium recognizer's cross-attention does not use a mask")
    return kwargs


def replay(args):
    import torch
    from safetensors.torch import load_file
    from ppocrv6_torch import PPOCRv6Medium
    from ppocrv6_torch.backbone import BatchNorm2d, ConvBNAct, EvenSameConv, fuse_conv_bn

    torch.set_num_threads(args.threads)
    torch.backends.mkldnn.enabled = args.mkldnn == "on"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = PPOCRv6Medium()
    state = load_file(args.checkpoint)
    model.load_state_dict(state, strict=True)
    named = dict(model.named_modules())
    path = args.boundaries / "boundaries.npz"
    metadata = json.loads((args.boundaries / "boundaries.json").read_text(encoding="utf-8"))
    if digest(path) != metadata["archive_sha256"] or metadata["baseline"] != BASELINE:
        raise ValueError("Boundary archive or source provenance checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    with torch.no_grad(), np.load(path, allow_pickle=False) as captured, \
            np.load(args.archive, allow_pickle=False) as baseline:
        x = torch.from_numpy(baseline["train/input"])
        targets = tuple(torch.from_numpy(baseline[f"train/{key}"]) for key in
                        ("label_ctc", "label_gtc", "length", "valid_ratio"))
        replay_archive = args.output / "torch_replay.npz"
        archive_context = nullcontext() if args.metrics_only else zipfile.ZipFile(replay_archive, "x")
        with archive_context as results:
            writer = Recorder(results)
            for mode in ("eval", "train"):
                model.load_state_dict(state, strict=True)
                model.train(mode == "train")
                for module in model.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.eval()
                natural = {}
                with capture_torch(model, natural):
                    if mode == "train":
                        model(x, targets)
                    else:
                        model(torch.from_numpy(baseline["eval/real_0/input"]))
                natural_functions = {}
                first = {name: inputs for (name, idx), (inputs, _) in natural.items() if idx == 0}
                last = {name: result for (name, _), (_, result) in natural.items()}

                def remember(name, _inputs, result, _category="activation"):
                    natural_functions[name] = result

                functional_boundaries(mode, named, first, last, remember, lambda v: v.log_softmax(-1))
                if mode == "train":
                    attention_boundaries(named, first, last, remember, lambda v: v.softmax(-1))
                model.load_state_dict(state, strict=True)
                injected = {}
                if mode == "train":
                    with capture_torch(model, injected):
                        model.head.gtc_head(torch.from_numpy(baseline["train/head.before_gtc"]),
                                            targets[1], targets[2])
                for row in (item for item in metadata["rows"] if item["mode"] == mode):
                    name, idx = row["name"], row["index"]
                    expected = captured[row["output"]]
                    values, kwargs = [], {}
                    for label, key in row["arguments"]:
                        value = torch.from_numpy(captured[key]) if key is not None else None
                        if isinstance(label, int):
                            values.append(value)
                        else:
                            kwargs[label] = value
                    category = row["category"]
                    if "::" in name:
                        result = functional_torch(name.rsplit("::", 1)[1], values)
                    else:
                        result = named[name](*values, **normalize_kwargs(kwargs))
                    key = row["output"]
                    writer.array(key, result)
                    entry = {"mode": mode, "name": name, "index": idx, "output": key,
                             "local": metrics(expected, result.numpy(), category)}
                    observed = natural.get((name, idx))
                    if observed is not None:
                        entry["natural"] = metrics(expected, observed[1].numpy(), category)
                        writer.array(key + "/natural", observed[1])
                    elif name in natural_functions:
                        entry["natural"] = metrics(expected, natural_functions[name].numpy(), category)
                        writer.array(key + "/natural", natural_functions[name])
                    if (name, idx) in injected:
                        exact = injected[name, idx][1]
                        entry["injected_memory"] = metrics(expected, exact.numpy(), category)
                        writer.array(key + "/injected_memory", exact)
                    if mode == "train" and isinstance(named.get(name), BatchNorm2d):
                        # Full replay buffers use the original state plus exactly one update.
                        bn = named[name]
                        bn.running_mean.copy_(state[name + ".running_mean"])
                        bn.running_var.copy_(state[name + ".running_var"])
                        bn(*values)
                        suffixes = (("running_var", "_variance"), ("running_mean", "_mean"))
                        for label, source in suffixes:
                            captured_key = key + "/" + label
                            reference = (captured[captured_key] if captured_key in captured.files
                                         else baseline[f"train/state/{name}.{source}"])
                            entry[label] = metrics(reference, getattr(bn, label).numpy(), category)
                    if mode == "eval" and isinstance(named.get(name), ConvBNAct):
                        block = named[name]
                        # Apply BN-folded weights to the same input, including even SAME padding.
                        fused = block.act(fuse_conv_bn(block.conv, block.bn)(values[0]))
                        entry["fused_diagnostic"] = metrics(expected, fused.numpy(), category)
                    if name == "head.before_gtc.1.fc":
                        weight = named[name].weight
                        source = values[0]
                        gemm = source.double() @ weight.double().t()
                        entry["fp64_accumulation_diagnostic"] = metrics(
                            expected, gemm.float().numpy(), category)
                        entry["gemm_input"] = metrics(source.numpy(), observed[0][0].numpy(), "input")
                        entry["gemm_input_strides"] = {"replay": list(source.stride()),
                                                       "natural": list(observed[0][0].stride())}
                        entry["converted_linear_weight_sha256"] = hashlib.sha256(
                            weight.numpy().tobytes()).hexdigest()
                    layer = named.get(name)
                    if isinstance(layer, torch.nn.Conv2d) and (not entry["local"]["pass"]
                                                               or name.startswith("backbone.conv1.")):
                        f = torch.nn.functional
                        value = values[0].double()
                        if isinstance(layer, EvenSameConv):
                            value = f.pad(value, (0, 1, 0, 1))
                        bias = None if layer.bias is None else layer.bias.double()
                        high = f.conv2d(value, layer.weight.double(), bias, layer.stride,
                                        layer.padding, layer.dilation, layer.groups).float()
                        entry["fp64_accumulation_diagnostic"] = metrics(expected, high.numpy(), category)
                        entry["fp32_vs_fp64_rounding"] = metrics(high.numpy(), result.numpy(), category)
                    if isinstance(layer, BatchNorm2d) and (not entry["local"]["pass"]
                                                          or name.startswith("backbone.conv1.")):
                        value = values[0].double()
                        if mode == "train":
                            variance, mean = torch.var_mean(value, (0, 2, 3), correction=0)
                        else:
                            variance = layer.running_var.double()
                            mean = layer.running_mean.double()
                        high = ((value - mean[None, :, None, None])
                                * torch.rsqrt(variance + layer.eps)[None, :, None, None]
                                * layer.weight.double()[None, :, None, None]
                                + layer.bias.double()[None, :, None, None]).float()
                        entry["fp64_reduction_diagnostic"] = metrics(expected, high.numpy(), category)
                    rows.append(entry)
    report = {"baseline": BASELINE, "boundaries_sha256": metadata["archive_sha256"],
              "replay_sha256": None if args.metrics_only else digest(replay_archive),
              "torch": torch.__version__, "threads": args.threads,
              "mkldnn": torch.backends.mkldnn.enabled, "rows": rows}
    (args.output / "comparisons.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                                  encoding="utf-8")
    print(f"{sum(row['local']['pass'] for row in rows)}/{len(rows)} local replays passed",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "replay"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--boundaries", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--mkldnn", choices=("on", "off"), default="on")
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()
    for field, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, field, value.resolve())
    for field, expected in BASELINE.items():
        path = args.dataset / "train.jsonl" if field == "train_manifest" else getattr(args, field)
        if digest(path) != expected:
            raise ValueError(f"Unexpected SHA-256 for {field}: {path}")
    if args.action == "capture":
        if args.source is None:
            parser.error("--source is required for capture")
        capture(args)
    else:
        if args.boundaries is None:
            parser.error("--boundaries is required for replay")
        replay(args)


if __name__ == "__main__":
    main()
