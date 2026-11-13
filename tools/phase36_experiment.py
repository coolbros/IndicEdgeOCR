"""Same-fixture CPU/GPU forwards, loss ablations and gradient comparisons."""

import argparse
import gc
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter

import numpy as np

from convert_paddle_checkpoint import digest, load_official, mapping
from trace_parity import BASELINE, metrics


PARAMETERS = (
    "backbone.conv1.stem1.conv.weight", "backbone.blocks2.0.token_mixer.rep_dw.conv.conv.weight",
    "backbone.blocks4.3.channel_mixer.compress.conv.weight", "head.ctc_head.fc.weight",
    "head.ctc_head.fc.bias", "head.before_gtc.1.fc.weight",
    "head.gtc_head.decoder.0.cross_attn.kv.weight", "head.gtc_head.decoder.3.mlp.fc2.weight",
    "head.gtc_head.tgt_word_prj.weight",
)
BACKBONE_BOUNDARIES = (
    "backbone.conv1.stem2a", "backbone.conv1.stem2b", "backbone.conv1",
    "backbone.blocks2.0", "backbone.blocks2", "backbone.blocks3", "backbone.blocks4",
    "backbone.blocks5", "backbone",
)
FEATURES = ("head.ctc_encoder", "head.before_gtc")
IMPLEMENTATIONS = ("native", "torch", "pairwise", "sequential")
LABELS = ("input", "label_ctc", "label_gtc", "length", "valid_ratio")


def cases(archive):
    rng = np.random.default_rng(36)
    return {
        "real_ctc": (archive["train/ctc_logits"], -1),
        "real_nrtr": (archive["train/nrtr_logits"], -1),
        "random": (rng.normal(size=(2, 19, 513)).astype("f"), -1),
        "middle_axis": (rng.normal(size=(2, 37, 7)).astype("f").transpose(2, 1, 0), 1),
        "clipping_ties": (np.array([[700, 700, 636.01, 636, 635.99, -700],
                                     [-700, -701, -763.99, -764, -764.01, -900]], "f"), -1),
        "leading_axis": (rng.uniform(-800, 800, (11, 3)).astype("f"), 0),
        "special_values": (np.array([[3, -np.inf], [-np.inf, -np.inf],
                                      [np.inf, 0], [np.nan, 2]], "f"), -1),
    }


def log_probes(backend, archive, refs, report):
    rng = np.random.default_rng(360)
    for name, (values, axis) in cases(archive).items():
        scale = np.full(values.shape, 0.01, dtype=np.float32)
        upstream = (rng.standard_normal(values.shape).astype("f") * scale)
        x = backend.tensor(values, True)
        if backend.paddle:
            y = backend.log(x, axis, "native")
            refs[f"log/{name}/forward"] = backend.array(y)
            if name != "special_values":
                refs[f"log/{name}/grad"] = backend.array(backend.grad(y, [x], backend.tensor(upstream))[0])
            continue
        for implementation in IMPLEMENTATIONS:
            key = f"log/{name}/{implementation}"
            begin = perf_counter()
            y = backend.log(x, axis, implementation)
            backend.sync()
            elapsed = perf_counter() - begin
            expected = refs[f"log/{name}/forward"]
            actual = backend.array(y)
            if name == "special_values":
                finite = np.isfinite(expected) & np.isfinite(actual)
                report[key] = {"shape": list(actual.shape), "dtype": str(actual.dtype),
                               "actual_nan": np.isnan(actual).tolist(),
                               "expected_nan": np.isnan(expected).tolist(),
                               "special_value_mismatch": int((~(finite | (actual == expected)
                                                                | (np.isnan(actual) & np.isnan(expected)))).sum()),
                               "finite_values": metrics(expected[finite], actual[finite], "probability")}
                continue
            grad = backend.array(backend.grad(y, [x], backend.tensor(upstream))[0])
            timings = []
            with backend.fw.no_grad():
                for _ in range(5):
                    begin = perf_counter()
                    backend.log(x, axis, implementation)
                    timings.append(perf_counter() - begin)
            report[key] = {"forward": metrics(expected, actual, "probability"),
                           "gradient": metrics(refs[f"log/{name}/grad"], grad, "activation"),
                           "median_forward_ms": median(timings) * 1000,
                           "cold_autograd_forward_ms": elapsed * 1000, "input_bytes": values.nbytes,
                           "working_tensor_bytes_lower_bound": 0 if implementation == "native"
                           else 2 * values.nbytes,
                           "input_contiguous": bool(x.is_contiguous())}


class Backend:
    def __init__(self, name, device, original, checkpoint, source):
        self.paddle = name == "paddle"
        self.device = device
        gpu = device == "gpu"
        if self.paddle:
            import paddle as fw
            import yaml
            from paddle_reference import CONFIG, PIN, git, load_verified
            if git(source, "rev-parse", "HEAD") != PIN or git(source, "status", "--porcelain"):
                raise ValueError("Expected clean pinned Paddle source")
            sys.path.insert(0, str(source))
            cwd = Path.cwd()
            try:
                os.chdir(source)
                from ppocr.modeling.architectures import build_model
                from ppocr.losses import build_loss
                fw.set_device("gpu:0" if gpu else "cpu")
                config = yaml.safe_load(Path(CONFIG).read_text(encoding="utf8"))
                config["Architecture"]["Head"]["out_channels_list"] = {
                    "CTCLabelDecode": 18710, "SARLabelDecode": 18712, "NRTRLabelDecode": 18713,
                }
                self.model = build_model(config["Architecture"])
                load_verified(config, self.model, original)
                self.loss_fn = build_loss(config["Loss"])
            finally:
                os.chdir(cwd)
            self.named = dict(self.model.named_sublayers())
            self.params = dict(self.model.named_parameters())
            self.initial = fw.load(str(original))
        else:
            import torch as fw
            from safetensors.torch import load_file
            from ppocrv6_torch import PPOCRv6Medium
            fw.set_num_threads(2)
            fw.set_float32_matmul_precision("highest")
            fw.backends.cuda.matmul.allow_tf32 = False
            fw.backends.cudnn.allow_tf32 = False
            fw.backends.cudnn.deterministic = True
            self.model = PPOCRv6Medium().to("cuda" if gpu else "cpu")
            self.initial = load_file(checkpoint)
            src = load_official(original)
            self.map = {target: (name, transpose) for name, target, transpose in mapping(self.model)}
            assert len(self.map) == len(src) == len(self.initial) == 441
            for target, (name, transpose) in self.map.items():
                np.testing.assert_array_equal(src[name].T if transpose else src[name],
                                              self.initial[target].numpy())
            self.named = dict(self.model.named_modules())
            self.params = dict(self.model.named_parameters())
        self.fw = fw
        self.reset(False)
        self.assert_device(next(iter(self.params.values())))

    def assert_device(self, value):
        actual = value.place.is_gpu_place() if self.paddle else value.is_cuda
        if actual != (self.device == "gpu"):
            raise RuntimeError(f"Unexpected device for {self.device} experiment")
        if str(value.dtype) not in ("paddle.float32", "torch.float32"):
            raise RuntimeError(f"Expected FP32 computation, got {value.dtype}")

    def reset(self, train):
        if self.paddle:
            self.model.set_state_dict(self.initial)
            self.model.train() if train else self.model.eval()
        else:
            self.model.load_state_dict(self.initial)
            self.model.train(train)
        for layer in self.named.values():
            if isinstance(layer, self.fw.nn.Dropout):
                layer.eval()
            if type(layer).__name__ in ("BatchNorm", "BatchNorm2D", "BatchNorm2d") and layer.training != train:
                raise RuntimeError("Unexpected BatchNorm mode")

    def tensor(self, data, grad=False):
        if self.paddle:
            return self.fw.to_tensor(data, stop_gradient=not grad)
        # Keep strides on CPU to exercise non-contiguous axes.
        return self.fw.from_numpy(data).to("cuda" if self.device == "gpu" else "cpu").requires_grad_(grad)

    @staticmethod
    def array(tensor):
        return tensor.detach().cpu().numpy()

    def sync(self):
        if self.device == "gpu":
            self.fw.device.cuda.synchronize() if self.paddle else self.fw.cuda.synchronize()

    def log(self, x, axis, implementation):
        if self.paddle:
            return self.fw.nn.functional.log_softmax(x, axis=axis)
        if implementation == "native":
            return x.log_softmax(axis)
        from ppocrv6_torch.ops.paddle_log_softmax import paddle_compatible_log_softmax
        return paddle_compatible_log_softmax(x, axis, implementation)

    def grad(self, out, values, upstream=None, retain=False):
        function = self.fw.grad if self.paddle else self.fw.autograd.grad
        gradients = function(out, values, grad_outputs=upstream, retain_graph=retain)
        for value in gradients:
            self.assert_device(value)
        return gradients

    def losses(self, output, batch, ctc="native", nrtr="native"):
        if self.paddle:
            return self.loss_fn(output, batch)
        from ppocrv6_torch.losses import multi_loss
        return multi_loss(output, *batch[1:4], ctc_log_softmax=ctc, nrtr_log_softmax=nrtr)

    def forward(self, batch, capture, inputs=None):
        def hook(name):
            def save(_module, _inputs, output):
                if isinstance(output, self.fw.Tensor):
                    self.assert_device(output)
                    capture[name] = output
                    if inputs is not None and name in BACKBONE_BOUNDARIES:
                        inputs[name] = _inputs[0]
            return save

        handles = []
        for name, layer in self.named.items():
            if (name.startswith(("backbone", "head.ctc", "head.before_gtc", "head.gtc_head"))
                    and not isinstance(layer, self.fw.nn.Embedding)):
                register = layer.register_forward_post_hook if self.paddle else layer.register_forward_hook
                handles.append(register(hook(name)))
        try:
            if self.model.training:
                return self.model(batch[0], batch[1:])
            return self.model(batch[0])
        finally:
            for handle in handles:
                handle.remove()

    def versions(self):
        data = {"framework": self.fw.__version__, "device": self.device, "platform": platform.platform(),
                "python": platform.python_version(), "numpy": np.__version__}
        if self.device == "gpu":
            data["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
                 "--format=csv,noheader", "-i", "0"], text=True).strip()
            if self.paddle:
                data.update(cuda=self.fw.version.cuda(), cudnn=self.fw.version.cudnn())
            else:
                data.update(cuda=self.fw.version.cuda, cudnn=self.fw.backends.cudnn.version(),
                            tf32_matmul=self.fw.backends.cuda.matmul.allow_tf32,
                            tf32_cudnn=self.fw.backends.cudnn.allow_tf32)
        return data


def experiment(backend, archive, refs, report):
    def record(key, value, category="activation"):
        backend.assert_device(value)
        actual = backend.array(value)
        if backend.paddle:
            refs[key] = actual
        else:
            report[key] = metrics(refs[key], actual, category)

    def gradients(name, loss, pairs, retain=False, reference_prefix=None):
        values = backend.grad(loss, [value for _, value in pairs], retain=retain)
        for (key, _), value in zip(pairs, values):
            if backend.paddle and key in PARAMETERS and key.endswith(".weight") and (
                key.startswith(("head.ctc_head", "head.before_gtc", "head.gtc_head"))
            ):
                value = value.transpose([1, 0])
            if reference_prefix is None:
                record(f"{name}/grad/{key}", value)
            else:
                report[f"{name}/grad/{key}"] = metrics(
                    refs[f"{reference_prefix}/grad/{key}"], backend.array(value), "activation")

    def replay(mode, inputs):
        if backend.paddle:
            for key, value in inputs.items():
                record(f"{mode}/backbone_input/{key}", value)
            return
        torch = backend.fw
        options = ("fp32", "fp64", "no_mkldnn") if backend.device == "cpu" else ("fp32", "fp64")
        with torch.no_grad():
            for option in options:
                for name in BACKBONE_BOUNDARIES:
                    backend.reset(mode == "train")
                    module = backend.named[name]
                    exact = backend.tensor(refs[f"{mode}/backbone_input/{name}"])
                    try:
                        if option == "fp64":
                            module.double()
                            exact = exact.double()
                        with torch.backends.mkldnn.flags(enabled=option != "no_mkldnn"):
                            value = module(exact).float()
                        report[f"{mode}/exact/{option}/{name}"] = metrics(
                            refs[f"{mode}/{name}"], backend.array(value), "activation")
                    finally:
                        if option == "fp64":
                            module.float()

    for name in ("real_0", "real_1", "real_2", "real_3"):
        backend.reset(False)
        captured, inputs = {}, {}
        with backend.fw.no_grad():
            probs = backend.forward([backend.tensor(archive[f"eval/{name}/input"])], captured, inputs)
        for key, value in captured.items():
            record(f"eval/{name}/{key}", value,
                   "probability" if key == "head.ctc_head" else "activation")
        record(f"eval/{name}/probabilities", probs, "probability")
        if not backend.paddle:
            expected = refs[f"eval/{name}/probabilities"]
            report[f"eval/{name}/ctc_path"] = {
                "argmax_equal": bool(np.array_equal(expected.argmax(2), backend.array(probs).argmax(2)))}
        if name == "real_0":
            replay(f"eval/{name}", inputs)
        captured.clear()
    backend.reset(True)
    batch = [backend.tensor(archive[f"train/{key}"], key == "input") for key in LABELS]
    captured, inputs = {}, {}
    outputs = backend.forward(batch, captured, inputs)
    if backend.paddle:
        replay("train", inputs)
    for key, value in captured.items():
        record(f"train/{key}", value)
    record("train/ctc_logits", outputs["ctc"], "logits")
    record("train/nrtr_logits", outputs["gtc"], "logits")
    for key, value in backend.model.state_dict().items():
        if key.endswith(("._mean", "._variance", ".running_mean", ".running_var")):
            canonical = key if backend.paddle else backend.map[key][0]
            record(f"train/bn_after/{canonical}", value)

    params = [(key, backend.params[key]) for key in PARAMETERS]
    # Paddle 3.2 CPU LayerNormGrad gates application of gamma to dX on
    # whether dScale is requested. Include each scale to measure training backward.
    norm_scales = [(f"layer_norm_scale/{name}", layer.weight)
                   for name, layer in backend.named.items()
                   if isinstance(layer, backend.fw.nn.LayerNorm)]
    groups = {
        "CTCLoss": [("logits", outputs["ctc"]), ("features", captured[FEATURES[0]]),
                    *[(k, v) for k, v in params if k.startswith("head.ctc_head")]],
        "NRTRLoss": [("logits", outputs["gtc"]), ("memory", captured[FEATURES[1]]),
                     *[(k, v) for k, v in params if k.startswith(("head.gtc_head", "head.before_gtc"))]],
        "loss": [("image", batch[0]), *params],
    }
    settings = (("native", "native"),) if backend.paddle else (
        ("native", "native"), ("torch", "native"), ("native", "torch"), ("native", "pairwise"))
    if not backend.paddle and backend.device == "cpu":
        settings += (("sequential", "native"), ("native", "sequential"))
    for index, (ctc, nrtr) in enumerate(settings):
        prefix = f"full/{ctc}-{nrtr}"
        losses = backend.losses(outputs, batch, ctc, nrtr)
        for loss_name, pairs in groups.items():
            source = f"full/native-native/{loss_name}"
            key = f"{prefix}/{loss_name}"
            if backend.paddle:
                record(key, losses[loss_name], "loss")
            else:
                report[key] = metrics(refs[source], backend.array(losses[loss_name]), "loss")
                report[key].update(paddle_value=float(refs[source]),
                                   torch_value=float(backend.array(losses[loss_name])))
            if loss_name == "CTCLoss":
                scales = [(k, v) for k, v in norm_scales if "/head.ctc_encoder." in k]
            elif loss_name == "NRTRLoss":
                scales = [(k, v) for k, v in norm_scales if "/head.gtc_head." in k]
            else:
                scales = norm_scales
            gradients(key, losses[loss_name], pairs + scales,
                      retain=index != len(settings) - 1 or loss_name != "loss",
                      reference_prefix=None if backend.paddle else source)

    # Detach the graph only to establish a separate loss-input experiment.
    for ctc, nrtr in settings:
        prefix = f"fixed_logits/{ctc}-{nrtr}"
        fixed = {"ctc": backend.tensor(archive["train/ctc_logits"], True),
                 "gtc": backend.tensor(archive["train/nrtr_logits"], True)}
        losses = backend.losses(fixed, batch, ctc, nrtr)
        for loss_name, name in (("CTCLoss", "ctc"), ("NRTRLoss", "gtc")):
            key, source = f"{prefix}/{loss_name}", f"fixed_logits/native-native/{loss_name}"
            if backend.paddle:
                record(key, losses[loss_name], "loss")
            else:
                report[key] = metrics(refs[source], backend.array(losses[loss_name]), "loss")
                report[key].update(paddle_value=float(refs[source]),
                                   torch_value=float(backend.array(losses[loss_name])))
            gradients(key, losses[loss_name], [("logits", fixed[name])],
                      reference_prefix=None if backend.paddle else source)
    if not backend.paddle:
        replay("train", inputs)
        with backend.fw.no_grad():
            for boundary in BACKBONE_BOUNDARIES:
                backend.reset(True)
                exact = backend.tensor(refs[f"train/{boundary}"])
                handle = backend.named[boundary].register_forward_hook(
                    lambda _module, _inputs, _output: exact)
                try:
                    capture = {}
                    intervened = backend.forward(batch, capture)
                finally:
                    handle.remove()
                for key, value in {
                    "backbone": capture["backbone"], "head.before_gtc": capture["head.before_gtc"],
                    "ctc_logits": intervened["ctc"], "nrtr_logits": intervened["gtc"],
                }.items():
                    report[f"intervention/{boundary}/{key}"] = metrics(
                        refs[f"train/{key}"], backend.array(value),
                        "logits" if key.endswith("logits") else "activation")


def fixed_memory_backward(backend, archive, refs, report):
    """Backpropagate through both decoders from identical memory and upstream gradients."""
    backend.reset(True)
    rng = np.random.default_rng(736)
    norm = backend.named["head.gtc_head.decoder.3.norm3"]
    x = backend.tensor(rng.standard_normal((2, 3, 512), dtype=np.float32), True)
    dy = backend.tensor(rng.standard_normal((2, 3, 512), dtype=np.float32))
    result = norm(x)
    for include_weight in (False, True):
        targets = [x, norm.weight] if include_weight else [x]
        actual = backend.array(backend.grad(result, targets, dy, retain=not include_weight)[0])
        key = f"isolated_layer_norm/grad/request_weight={include_weight}"
        if backend.paddle:
            refs[key] = actual
        else:
            report[key] = metrics(refs[key], actual, "activation")
    memory = backend.tensor(archive["train/head.before_gtc"], True)
    tokens = backend.tensor(archive["train/label_gtc"])
    lengths = backend.tensor(archive["train/length"])
    head = backend.model.head.gtc_head
    observed = {}
    handles = []
    for name, layer in backend.named.items():
        if name.startswith("head.gtc_head.decoder.") and name.rsplit(".", 1)[-1] in (
            "norm1", "norm2", "norm3", "self_attn", "cross_attn", "mlp",
        ):
            def save(_layer, _inputs, value, key=name):
                if isinstance(value, backend.fw.Tensor):
                    observed[key] = value
            hook = layer.register_forward_post_hook if backend.paddle else layer.register_forward_hook
            handles.append(hook(save))
    try:
        logits = head(memory, (tokens, lengths)) if backend.paddle else head(memory, tokens, lengths)
    finally:
        for handle in handles:
            handle.remove()
    observed["memory"] = memory
    upstream = rng.standard_normal(list(logits.shape), dtype=np.float32) * np.float32(1e-4)
    names = sorted(observed)
    norm_weights = [value for name, value in backend.params.items()
                    if name.startswith("head.gtc_head.decoder.") and ".norm" in name]
    for include_norm_weights in (False, True):
        targets = [observed[name] for name in names]
        if include_norm_weights:
            targets += norm_weights
        gradients = backend.grad(logits, targets, backend.tensor(upstream),
                                 retain=not include_norm_weights)
        mode = "with_norm_parameters" if include_norm_weights else "activations_only"
        for name, gradient in zip(names, gradients):
            key = f"fixed_memory/{mode}/grad/{name}"
            actual = backend.array(gradient)
            if backend.paddle:
                refs[key] = actual
            else:
                report[key] = metrics(refs[key], actual, "activation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=("paddle", "torch"))
    parser.add_argument("device", choices=("cpu", "gpu"))
    for name in ("source", "archive", "original", "checkpoint", "dataset", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--decoder-only", action="store_true")
    args = parser.parse_args()
    for key, expected in BASELINE.items():
        path = args.dataset / "train.jsonl" if key == "train_manifest" else getattr(args, key)
        if digest(path) != expected:
            raise ValueError(f"Input digest mismatch: {key}")
    args.output.mkdir(parents=True, exist_ok=False)
    backend = Backend(args.framework, args.device, args.original.resolve(),
                      args.checkpoint.resolve(), args.source.resolve())
    report = {}
    with np.load(args.archive, allow_pickle=False) as archive:
        if backend.paddle:
            refs = {}
            if args.device == "cpu" and not args.decoder_only:
                log_probes(backend, archive, refs, report)
            if not args.decoder_only:
                experiment(backend, archive, refs, report)
            fixed_memory_backward(backend, archive, refs, report)
            output = args.output / "reference.npz"
            np.savez_compressed(output, **refs)
            report["reference_sha256"] = digest(output)
        else:
            metadata = json.loads(args.reference.with_name("results.json").read_text())
            if (metadata["inputs"] != BASELINE or metadata["runtime"]["device"] != args.device
                    or digest(args.reference) != metadata["metrics"]["reference_sha256"]):
                raise ValueError("Paddle reference digest mismatch")
            with np.load(args.reference, allow_pickle=False) as refs:
                if args.device == "cpu" and not args.decoder_only:
                    log_probes(backend, archive, refs, report)
                    gc.collect()
                if not args.decoder_only:
                    experiment(backend, archive, refs, report)
                fixed_memory_backward(backend, archive, refs, report)
    (args.output / "results.json").write_text(json.dumps({
        "inputs": BASELINE, "runtime": backend.versions(), "metrics": report,
    }, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Completed {args.framework} {args.device} with {len(report)} metrics", flush=True)


if __name__ == "__main__":
    main()
