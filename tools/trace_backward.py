"""Capture Paddle VJPs and replay them with identical coefficients in Torch."""

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import zipfile

import numpy as np

from convert_paddle_checkpoint import digest
from phase36_experiment import Backend, LABELS, PARAMETERS
from trace_parity import BASELINE, attention_steps, metrics, unpack_attention

HEAD = "head.gtc_head"
CTC = "head.ctc_head.fc"
MEMORY = "head.before_gtc"
ENCODER = "head.ctc_encoder"
STAGES = ("backbone.conv1", *(f"backbone.blocks{i}" for i in range(2, 6)))
SEED = 3701


def unique_targets(targets):
    values, indices, identities = [], {}, {}
    for name, value in targets.items():
        if id(value) not in identities:
            identities[id(value)] = len(values)
            values.append(value)
        indices[name] = identities[id(value)]
    return values, indices


def vjp(grad, outputs, upstream, targets, retain=False):
    values, indices = unique_targets(targets)
    outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
    if upstream is not None:
        upstream = list(upstream) if isinstance(upstream, (tuple, list)) else [upstream]
    gradients = grad(outputs, values, grad_outputs=upstream, retain_graph=retain, allow_unused=True)
    if len(gradients) != len(values):
        raise RuntimeError("Autograd returned an unexpected number of gradients")
    missing = [name for name, index in indices.items() if gradients[index] is None]
    if missing:
        raise RuntimeError(f"Missing expected gradients: {missing}")
    return {name: gradients[index] for name, index in indices.items()}


def gradient_metrics(expected, actual, axes=None):
    row = metrics(expected, actual, "activation")
    if expected.shape != actual.shape:
        return row
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Nonfinite gradient")
    absolute = np.abs(expected.astype("d") - actual)
    magnitude = np.abs(expected)
    bad = absolute > row["atol"] + row["rtol"] * magnitude
    count = int(bad.sum())
    row.update(outliers=count, outlier_fraction=count / max(1, bad.size))
    limits = (0., 1e-7, 1e-5, 1e-3, 1., np.inf)
    row["reference_magnitude_bands"] = [
        {"lower": lo, "upper": hi if np.isfinite(hi) else None,
         "count": int(((magnitude >= lo) & (magnitude < hi)).sum()),
         "outliers": int((bad & (magnitude >= lo) & (magnitude < hi)).sum())}
        for lo, hi in zip(limits, limits[1:])
    ]
    row["outlier_axes"] = {
        name: bad.sum(axis=tuple(i for i in range(bad.ndim) if i != axis)).tolist()
        for name, axis in (axes or {}).items()
    }
    if count:
        row["max_error_index"] = list(map(int, np.unravel_index(absolute.argmax(), absolute.shape)))
    return row


def array_hash(array):
    value = np.ascontiguousarray(array)
    sha = hashlib.sha256(f"{value.dtype}:{value.shape}:".encode())
    sha.update(memoryview(value).cast("B"))
    return sha.hexdigest()


def axes_for(array, layout):
    names = {"NCHW": ("sample", "channel", "height", "width"),
             "NTC": ("sample", "token", "channel"),
             "NHTD": ("sample", "head", "token", "feature"),
             "OI": ("output", "input")}
    return {name: axis for axis, name in enumerate(names.get(layout, ()))
            if array.shape[axis] <= 1024}


class Trace:
    def __init__(self, backend, archive, reference=None, diagnostics=True):
        self.b = backend
        self.archive = archive
        self.reference = reference
        self.rows = {}
        self.blobs = set()
        self.rng = np.random.default_rng(SEED)
        self.diagnostics = diagnostics
        self.requests = {}
        self.internal = {}
        self.probe_order = []

    def array(self, key, value, role="forward", layout=None):
        self.b.assert_device(value)
        array = self.b.array(value)
        strides = value.strides if self.b.paddle else value.stride()
        row = {"sha256": array_hash(array), "shape": list(array.shape),
               "dtype": str(array.dtype), "strides_elements": list(strides), "layout": layout}
        if self.reference is not None:
            expected = self.exact(key)
            if role == "exact" and row["sha256"] != self.reference[key]["sha256"]:
                raise ValueError(f"Exact operand changed: {key}")
            if role == "gradient":
                row["comparison"] = gradient_metrics(expected, array, axes_for(array, layout))
            elif role == "forward":
                row["comparison"] = metrics(expected, array, "activation")
        if self.archive is not None and row["sha256"] not in self.blobs:
            with self.archive.open(row["sha256"] + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, array, allow_pickle=False)
            self.blobs.add(row["sha256"])
        if key in self.rows:
            raise ValueError(f"Duplicate capture key: {key}")
        self.rows[key] = row
        return array

    def exact(self, key):
        return self.reference_arrays[self.reference[key]["sha256"]]

    def select(self, key, value, grad=False):
        array = self.exact(key) if self.reference is not None else self.b.array(value)
        return self.b.tensor(array, grad)

    def upstream(self, key, y, value=None):
        if self.reference is not None:
            value = self.b.tensor(self.exact(key))
        elif value is None:
            value = self.b.tensor(self.rng.standard_normal(tuple(y.shape), dtype=np.float32) * np.float32(0.01))
        self.array(key, value, "exact")
        return value

    def parameters(self, roots):
        return {key: value for key, value in self.b.params.items()
                if any(key.startswith(root + ".") for root in roots)
                and (not value.stop_gradient if self.b.paddle else value.requires_grad)}

    def grads(self, key, out, upstream, tensors, params, retain=False, save_all=True):
        for value in params.values():
            self.b.assert_device(value)
        targets = {**tensors, **params}
        function = self.b.fw.grad if self.b.paddle else self.b.fw.autograd.grad
        result = vjp(function, out, upstream, targets, retain)
        for name, value in result.items():
            if name in params and self.b.paddle:
                module = self.b.named[name.rsplit(".", 1)[0]]
                if name.endswith(".weight") and isinstance(module, self.b.fw.nn.Linear):
                    value = value.transpose([1, 0])
            self.b.assert_device(value)
            array = self.b.array(value)
            if not np.isfinite(array).all():
                raise ValueError(f"Nonfinite requested gradient {key}/{name}")
            if save_all or name in tensors or name in PARAMETERS:
                layout = layout_for(key if "::" in key else name, array.ndim)
                self.array(f"{key}/grad/{name}", value, "gradient", layout)
        self.requests[key] = sorted(targets)
        return result

    def probe(self, name, operation, arguments, params=None, frozen=None, upstream=None):
        key = "local/" + name
        # Each local forward starts from independent leaves and exact Paddle argument bytes.
        inputs = [self.select(f"{key}/input/{i}", a, True) for i, a in enumerate(arguments)]
        for i, x in enumerate(inputs):
            self.array(f"{key}/input/{i}", x, "exact")
        output = operation(*inputs)
        self.array(f"{key}/output", output)
        dy = self.upstream(f"{key}/upstream", output, upstream)
        if tuple(dy.shape) != tuple(output.shape):
            raise ValueError(f"Upstream shape mismatch: {key}")
        tensors = {f"input_{i}": value for i, value in enumerate(inputs)}
        grads = self.grads(key, output, dy, tensors, params or {})
        self.probe_order.append(key)
        if frozen:
            for label, value in frozen.items():
                self.array(f"{key}/constant/{label}", value, "exact")
        return output, grads


def layout_for(name, ndim):
    if ndim == 4 and ("::" in name):
        return "NHTD"
    if ndim == 4:
        return "NCHW"
    if ndim == 3:
        return "NTC"
    if ndim == 2 and name.endswith(".weight"):
        return "OI"
    return None


@contextmanager
def hooks(backend, names):
    calls, handles = {}, []
    for name in names:
        module = backend.named[name]

        def capture(_module, args, output, key=name):
            calls[key] = args, output

        register = module.register_forward_post_hook if backend.paddle else module.register_forward_hook
        handles.append(register(capture))
    try:
        yield calls
    finally:
        for handle in handles:
            handle.remove()


def mask_for(backend, sequence):
    length = sequence.shape[1]
    mask = np.full((length, length), -np.inf, dtype=np.float32)
    return backend.tensor(np.triu(mask, 1)[None, None])


def block_call(backend, block, x, memory, mask):
    if backend.paddle:
        return block(x, memory, self_mask=mask)
    return block(x, memory, mask)


def attention_call(backend, layer, x, memory, mask):
    if layer.self_attn:
        return layer(x, attn_mask=mask) if backend.paddle else layer(x, mask=mask)
    return layer(x, memory)


def boundary_names(backend):
    decoder = HEAD + ".decoder."
    decoder_layers = ("self_attn", "cross_attn", "norm1", "norm2", "norm3", "mlp", "q", "kv",
                      "qkv", "out_proj", "fc1", "fc2", "act")
    return [
        name for name in backend.named if name in (
            "backbone", *STAGES, CTC, MEMORY, MEMORY+".0", MEMORY+".1.fc",
            ENCODER, HEAD+".positional_encoding", HEAD+".tgt_word_prj",
        ) or name in backbone_probe_names() or name in encoder_probe_names()
        or (name.startswith(decoder) and name.rsplit(".", 1)[-1] in decoder_layers)
        or (name.startswith(decoder) and name.count(".") == 3)
    ]


def backbone_probe_names():
    return (
        *(f"backbone.conv1.stem{i}" for i in ("1", "2a", "2b", "3", "4")),
        *(f"backbone.conv1.stem2b.{name}" for name in ("conv", "bn", "act")),
        "backbone.conv1.pool",
        *(f"backbone.blocks{stage}.{i}" for stage, count in ((2, 1), (3, 3), (4, 7), (5, 3))
          for i in range(count)),
        *(f"backbone.blocks{stage}.{i}.{name}" for stage, i in ((2, 0), (4, 3), (5, 2))
          for name in ("token_mixer", "token_mixer.rep_dw", "token_mixer.rep_dw.conv.conv",
                       "token_mixer.rep_dw.conv.bn", "token_mixer.rep_dw.conv1",
                       "token_mixer.rep_dw.bn", "channel_mixer", "channel_mixer.expand.conv",
                       "channel_mixer.expand.bn", "channel_mixer.act", "channel_mixer.compress.conv",
                       "channel_mixer.compress.bn")),
    )


def encoder_probe_names():
    root = ENCODER + ".encoder."
    return tuple(root + name for name in ("skip_conv", "conv_reduce", "local_conv", "svtr_block.0",
                                           "svtr_block.1", "norm"))


def grad_operators(trace, outputs, upstream, targets):
    function = trace.b.fw.grad if trace.b.paddle else trace.b.fw.autograd.grad
    return vjp(function, outputs, upstream, targets, retain=True)


def branch_grad(trace, label, output, upstream, tensors, params, save_all=False):
    return trace.grads(label, output, upstream, tensors, params, retain=True, save_all=save_all)


def gradients_at(calls, names):
    return {name: calls[name][1] for name in names}


def run_trace(trace, fixture):
    b = trace.b
    b.reset(True)
    x = b.tensor(fixture["train/input"], True)
    batch = [x, *(b.tensor(fixture[f"train/{key}"]) for key in LABELS[1:])]
    trace.batch_hashes = {name: array_hash(fixture[f"train/{name}"]) for name in LABELS}
    names = boundary_names(b)
    with hooks(b, names) as calls:
        output = b.model(x, batch[1:])
    pooled = calls["backbone"][1]
    real_memory = calls[MEMORY][1]
    real_ctc_features = calls[ENCODER][1]

    # Preserve both loss-generated seeds and the matched-seed full-graph comparison.
    losses = b.losses(output, batch)
    c_nat = grad_operators(trace, losses["CTCLoss"], None, {"logits": output["ctc"]})["logits"]
    n_nat = grad_operators(trace, losses["NRTRLoss"], None, {"logits": output["gtc"]})["logits"]
    for branch, dy in (("ctc", c_nat), ("nrtr", n_nat)):
        trace.array(f"natural_loss/{branch}/logits_gradient", dy, "gradient", "NTC")
    c_seed = trace.upstream("seed/ctc", output["ctc"], c_nat)
    n_seed = trace.upstream("seed/nrtr", output["gtc"], n_nat)
    shared = {"image": x, **gradients_at(calls, ("backbone", *STAGES))}
    backbone_nodes = gradients_at(calls, [name for name in backbone_probe_names() if name in calls])
    encoder_nodes = gradients_at(calls, [name for name in encoder_probe_names() if name in calls])
    c_params = trace.parameters(("backbone", "head.ctc_encoder", "head.ctc_head"))
    n_params = trace.parameters(("backbone", MEMORY, HEAD))
    c_tensors = {**shared, **backbone_nodes, **encoder_nodes, "features": real_ctc_features}
    n_tensors = {**shared, **backbone_nodes, "memory": real_memory}
    c_grad = branch_grad(trace, "natural/ctc", output["ctc"], c_seed, c_tensors, c_params)
    n_grad = branch_grad(trace, "natural/nrtr", output["gtc"], n_seed, n_tensors, n_params)
    params = {**c_params, **n_params}
    joint_tensors = {**c_tensors, **n_tensors}
    joint = branch_grad(trace, "natural/joint", (output["ctc"], output["gtc"]),
                        (c_seed, n_seed), joint_tensors, params)
    reverse = branch_grad(trace, "natural/joint_reversed", (output["gtc"], output["ctc"]),
                          (n_seed, c_seed), joint_tensors, params)
    actual = branch_grad(trace, "natural/loss", losses["loss"], None, joint_tensors, params)
    for name in shared:
        total = c_grad[name] + n_grad[name]
        trace.array(f"sum/{name}", total, "gradient", layout_for(name, total.ndim))
        trace.internal[f"branch_sum/{name}"] = gradient_metrics(b.array(total), b.array(joint[name]))
        trace.internal[f"reverse/{name}"] = gradient_metrics(b.array(joint[name]), b.array(reverse[name]))
        trace.internal[f"loss_vs_matched/{name}"] = gradient_metrics(b.array(actual[name]), b.array(joint[name]))
    trace.native_outputs = {name: trace.array(f"natural/forward/{name}", value)
                            for name, (_, value) in calls.items()}
    trace.array("natural/logits/ctc", output["ctc"])
    trace.array("natural/logits/nrtr", output["gtc"])
    print("Captured native branches and shared accumulation", flush=True)
    for name, value in b.model.state_dict().items():
        if name.endswith(("._mean", "._variance", ".running_mean", ".running_var")):
            key = name if b.paddle else b.map[name][0]
            trace.array(f"natural/bn_after/{key}", value)

    # Collect propagated native gradient references at the decoder boundaries with all affine grads.
    decoder_nodes = gradients_at(calls, [name for name in names if name.startswith(HEAD+".decoder.")
                                         or name == HEAD+".positional_encoding"])
    d_out = calls[HEAD+".tgt_word_prj"][1]
    native_decoder = grad_operators(trace, d_out, n_seed, {**decoder_nodes, **trace.parameters((HEAD,))})
    # Same native graph shared by the independent calls above. Drop it before local replays.
    local_sources = {name: tuple(b.array(v) for v in args[:2] if isinstance(v, b.fw.Tensor)
                                 and str(v.dtype).endswith("float32"))
                     for name, (args, _) in calls.items()}
    d_seeds = {name: b.array(native_decoder[name]) for name in decoder_nodes}
    c_feature_grad, n_memory_grad = b.array(c_grad["features"]), b.array(n_grad["memory"])
    c_pooled_grad, n_pooled_grad = b.array(c_grad["backbone"]), b.array(n_grad["backbone"])
    stage_seeds = {label: {name: b.array(result[name]) for name in (*shared, *backbone_nodes)}
                   for label, result in (("ctc", c_grad), ("nrtr", n_grad))}
    encoder_seeds = {name: b.array(c_grad[name]) for name in encoder_nodes}
    pooled_array, memory_array = b.array(pooled), b.array(real_memory)
    natural_ctc_array = b.array(real_ctc_features)
    del calls, output, losses, c_grad, n_grad, joint, reverse, actual, native_decoder, decoder_nodes
    del pooled, real_memory, real_ctc_features, shared, c_tensors, n_tensors, joint_tensors, d_out
    del backbone_nodes, encoder_nodes

    b.reset(True)
    trace.probe("ctc/exact", b.named[CTC], (b.tensor(natural_ctc_array),), trace.parameters((CTC,)),
                upstream=c_seed)
    natural_x = b.tensor(natural_ctc_array, True)
    natural_y = b.named[CTC](natural_x)
    trace.array("ctc_natural/features", natural_x)
    trace.array("ctc_natural/logits", natural_y)
    trace.grads("ctc_natural", natural_y, c_seed, {"features": natural_x}, trace.parameters((CTC,)))
    trace.probe("nrtr/flatten", b.named[MEMORY+".0"], (b.tensor(pooled_array),))
    def projection_layout(value):
        return value.transpose([0, 2, 1]) if b.paddle else value.transpose(1, 2)
    flat = b.named[MEMORY+".0"](b.tensor(pooled_array))
    trace.probe("nrtr/transpose", projection_layout, (flat,))
    trace.probe("nrtr/projection", b.named[MEMORY], (b.tensor(pooled_array),),
                trace.parameters((MEMORY,)), upstream=b.tensor(n_memory_grad))
    trace.probe("nrtr/linear", b.named[MEMORY+".1.fc"],
                (b.tensor(local_sources[MEMORY+".1.fc"][0]),),
                trace.parameters((MEMORY+".1.fc",)), upstream=b.tensor(n_memory_grad))

    trace.probe("ctc/encoder", b.named[ENCODER], (b.tensor(pooled_array),),
                trace.parameters((ENCODER,)), upstream=b.tensor(c_feature_grad))
    for name in reversed(tuple(encoder_seeds)):
        trace.probe(name, b.named[name], (b.tensor(local_sources[name][0]),),
                    trace.parameters((name,)), upstream=b.tensor(encoder_seeds[name]))
    b.reset(True)
    decoder = b.named[HEAD]
    def decode(memory):
        return decoder(memory, (batch[2], batch[3])) if b.paddle else decoder(memory, batch[2], batch[3])
    trace.probe("nrtr/decoder", decode, (b.tensor(memory_array),), trace.parameters((HEAD,)),
                upstream=n_seed)
    trace.probe("layernorm/standalone", b.named[HEAD+".decoder.3.norm3"],
                (b.tensor(np.random.default_rng(SEED).standard_normal((2, 3, 512), dtype=np.float32)),),
                trace.parameters((HEAD+".decoder.3.norm3",)))
    trace.probe("nrtr/output_projection", b.named[HEAD+".tgt_word_prj"],
                (b.tensor(local_sources[HEAD+".tgt_word_prj"][0]),),
                trace.parameters((HEAD+".tgt_word_prj",)), upstream=n_seed)
    width = local_sources[HEAD+".decoder.0"][0].shape[1]
    causal_mask = mask_for(b, b.tensor(np.empty((1, width), "f")))
    for i in reversed(range(4)):
        root = f"{HEAD}.decoder.{i}"
        z = b.tensor(local_sources[root][0])
        memory = b.tensor(memory_array)
        trace.probe(root, lambda a, m: block_call(b, b.named[root], a, m, causal_mask),
                    (z, memory), trace.parameters((root,)), {"mask": causal_mask},
                    upstream=b.tensor(d_seeds[root]))
        for leaf in ("norm3", "mlp", "norm2", "cross_attn", "norm1", "self_attn"):
            name = f"{root}.{leaf}"
            layer = b.named[name]
            values = tuple(b.tensor(a) for a in local_sources[name][:1])
            dy = b.tensor(d_seeds[name])
            if leaf.endswith("attn"):
                m = causal_mask if layer.self_attn else None
                args = values if layer.self_attn else (*values, memory)
                trace.probe(name, lambda a, k=None: attention_call(b, layer, a, k, m),
                            args, trace.parameters((name,)), upstream=dy)
                attention_local(trace, name, layer, args, m)
            else:
                trace.probe(name, layer, values, trace.parameters((name,)), upstream=dy)
                if leaf.startswith("norm"):
                    layernorm_diagnostics(trace, name, layer, values[0], dy)
        for i_norm, leaf in reversed(tuple(enumerate(("self_attn", "cross_attn", "mlp"), 1))):
            lhs = z if i_norm == 1 else b.tensor(local_sources[root+f".{leaf}"][0])
            # Recover the residual RHS from the recorded addition operands.
            # Dropout is disabled. Re-use the native outputs from the recorded forward.
            rhs = b.tensor(trace.native_outputs[f"{root}.{leaf}"])
            trace.probe(f"{root}.residual{i_norm}", lambda a, c: a + c, (lhs, rhs))
        for name in sorted(n for n in d_seeds if n.startswith(root + ".")
                           and n.rsplit(".", 1)[-1] in ("q", "kv", "qkv", "out_proj", "fc1", "fc2", "act")):
            trace.probe(name, b.named[name], (b.tensor(local_sources[name][0]),),
                        trace.parameters((name,)), upstream=b.tensor(d_seeds[name]))
    print("Captured isolated heads and decoder operators", flush=True)

    for label, seed in (("ctc", c_pooled_grad), ("nrtr", n_pooled_grad),
                        ("sum", c_pooled_grad + n_pooled_grad)):
        b.reset(True)
        trace.probe(f"backbone/{label}", b.model.backbone, (x,),
                    trace.parameters(("backbone",)), upstream=b.tensor(seed))
        if label != "sum":
            backbone_local(trace, label, local_sources, stage_seeds[label], seed)


def attention_local(trace, name, layer, args, mask):
    from trace_parity import attention_op

    b = trace.b
    softmax = lambda value: b.fw.nn.functional.softmax(value, -1)
    kwargs = {"attn_mask" if b.paddle else "mask": mask} if layer.self_attn else {}
    q, k, v, mask = unpack_attention(layer, args, kwargs, b.paddle)
    q, k, v = (b.tensor(b.array(value), True) for value in (q, k, v))
    steps = attention_steps(q, k, v, mask, softmax, b.paddle)
    seed = trace.upstream(f"attention/{name}/seed", steps["merge"][1])
    for operation, (values, _) in reversed(list(steps.items())):
        key = name + "::" + operation
        if operation == "mask":
            if mask is not None:
                _, grads = trace.probe(key, lambda x: x + mask, values[:1],
                                       frozen={"mask": mask}, upstream=seed)
        else:
            _, grads = trace.probe(key, lambda *x: attention_op(operation, x, softmax, b.paddle),
                                   values, upstream=seed)
        seed = grads["input_0"]


def layernorm_diagnostics(trace, name, layer, value, upstream):
    b = trace.b
    key = "local/" + name
    if b.paddle:
        if name.endswith("3.norm3"):
            x = trace.select(f"{key}/input/0", value, True)
            out = layer(x)
            grads = b.fw.grad(out, [x], grad_outputs=upstream)
            trace.array("layernorm/partial/input_gradient", grads[0], "gradient", "NTC")
        return
    if name.endswith("3.norm3"):
        x = trace.select(f"{key}/input/0", value, True)
        dy = b.tensor(trace.exact(f"{key}/upstream")) if trace.reference else upstream
        dx = b.fw.autograd.grad(layer(x), x, dy)[0]
        trace.array("layernorm/partial/input_gradient", dx, "gradient", "NTC")
    if not trace.diagnostics or trace.reference is None:
        return
    fw = b.fw
    import torch.nn.functional as f
    for variant in ("fp64", "strided", "centered"):
        x = b.tensor(trace.exact(f"{key}/input/0"))
        dy = b.tensor(trace.exact(f"{key}/upstream"))
        w, bias = layer.weight.detach(), layer.bias.detach()
        if variant == "fp64":
            x, dy, w, bias = (a.double() for a in (x, dy, w, bias))
        elif variant == "strided":
            x = x.transpose(-1, -2).contiguous().transpose(-1, -2)
        x.requires_grad_()
        w.requires_grad_()
        bias.requires_grad_()
        if variant == "centered":
            centered = x - x.mean(-1, keepdim=True)
            y = centered * fw.rsqrt(centered.square().mean(-1, keepdim=True) + layer.eps) * w + bias
        else:
            y = f.layer_norm(x, layer.normalized_shape, w, bias, layer.eps)
        dx, dw, db = fw.autograd.grad(y, (x, w, bias), dy)
        trace.internal[f"layernorm/{name}/{variant}"] = {
            label: gradient_metrics(trace.exact(f"{key}/grad/{target}"), b.array(g.float()))
            for label, target, g in (("input", "input_0", dx), ("weight", name+".weight", dw),
                                     ("bias", name+".bias", db))
        }


def backbone_local(trace, label, sources, seeds, pooled_seed):
    b = trace.b
    pool = b.fw.nn.functional.adaptive_avg_pool2d
    # The recognition stage 6 is an empty sequence.
    last_input = trace.native_outputs[STAGES[-1]]
    trace.probe(f"backbone/{label}/pool", lambda z: pool(z, [1, 40]),
                (b.tensor(last_input),), upstream=b.tensor(pooled_seed))
    for name in reversed(tuple(name for name in sources if name in seeds and name != "backbone")):
        trace.probe(f"backbone/{label}/{name}", b.named[name], (b.tensor(sources[name][0]),),
                    trace.parameters((name,)), upstream=b.tensor(seeds[name]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=("paddle", "torch"))
    parser.add_argument("device", choices=("cpu", "gpu"))
    for name in ("source", "archive", "original", "checkpoint", "dataset", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--no-mkldnn", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--baseline-sha", default="b850ee7da73d180cd4296abade221d3a373cfda9")
    args = parser.parse_args()
    if (args.framework == "torch") != (args.reference is not None):
        parser.error("Torch needs a Paddle reference directory")
    if args.framework == "paddle" and args.metrics_only:
        parser.error("Paddle capture requires tensor output")
    for key, expected in BASELINE.items():
        path = args.dataset / "train.jsonl" if key == "train_manifest" else getattr(args, key)
        if digest(path) != expected:
            raise ValueError(f"Input digest mismatch: {key}")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    if args.device == "gpu":
        os.environ.update(FLAGS_cudnn_deterministic="1", FLAGS_use_tf32="0",
                          NVIDIA_TF32_OVERRIDE="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    b = Backend(args.framework, args.device, args.original.resolve(),
                args.checkpoint.resolve(), args.source.resolve())
    if not b.paddle:
        b.fw.set_num_threads(args.threads)
        b.fw.backends.mkldnn.enabled = not args.no_mkldnn
    reference, ref_path = None, None
    if args.reference:
        reference = json.loads((args.reference / "capture.json").read_text(encoding="utf8"))
        ref_path = args.reference / "tensors.npz"
        if (reference["inputs"] != BASELINE or reference["runtime"]["device"] != args.device
                or reference["baseline_commit"] != args.baseline_sha
                or reference["capture_sha256"] != digest(ref_path) or reference["seed"] != SEED):
            raise ValueError("Paddle capture or configuration checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    archive = args.output / "tensors.npz"
    with np.load(args.archive, allow_pickle=False) as fixture, \
            np.load(ref_path, allow_pickle=False) if ref_path else nullcontext(None) as arrays, \
            (nullcontext(None) if args.metrics_only else zipfile.ZipFile(
                archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1)) as writer:
        trace = Trace(b, writer, reference["rows"] if reference else None)
        trace.reference_arrays = arrays
        run_trace(trace, fixture)
    if reference and (trace.requests != reference["requests"] or trace.rows.keys() != reference["rows"].keys()):
        raise ValueError("Framework coverage or gradient requests differ")
    runtime = b.versions()
    runtime.update(threads=args.threads, torch_mkldnn=None if b.paddle else not args.no_mkldnn,
                   flags={name: os.environ.get(name) for name in (
                       "FLAGS_use_mkldnn", "FLAGS_use_tf32", "FLAGS_cudnn_deterministic",
                       "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES")})
    host = Path("/proc/cpuinfo")
    runtime["cpu"] = next((line.split(":", 1)[-1].strip() for line in host.read_text().splitlines()
                           if line.startswith("model name")), None) if host.exists() else platform.processor()
    result = {"baseline_commit": args.baseline_sha, "inputs": BASELINE, "seed": SEED,
              "mode": "train, dropout disabled", "runtime": runtime, "batch_hashes": trace.batch_hashes,
              "capture_sha256": None if args.metrics_only else digest(archive), "rows": trace.rows,
              "requests": trace.requests, "probe_order": trace.probe_order,
              "within_framework": trace.internal}
    (args.output / "capture.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    print(f"{args.framework} {args.device} wrote {len(trace.rows)} rows", flush=True)


if __name__ == "__main__":
    main()
