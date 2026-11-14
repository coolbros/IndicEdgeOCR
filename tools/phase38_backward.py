"""Phase 3.8 primitive backward probes using the Phase 3.7 matched coefficients."""

import argparse
import json
from pathlib import Path

import numpy as np

from convert_paddle_checkpoint import digest
from phase36_experiment import Backend
from trace_backward import gradient_metrics, vjp
from trace_parity import BASELINE, metrics

REP = "backbone.blocks2.0.token_mixer.rep_dw"
SE = "backbone.blocks2.0.token_mixer.se"
BLOCK = "backbone.blocks2.0"
EMB = "head.gtc_head.embedding"
KV = "head.gtc_head.decoder.2.cross_attn.kv"


def run(args):
    original, checkpoint = args.original.resolve(), args.checkpoint.resolve()
    if digest(original) != BASELINE["original"] or digest(checkpoint) != BASELINE["checkpoint"]:
        raise ValueError("Frozen checkpoint mismatch")
    b = Backend(args.framework, args.device, original, checkpoint, args.source.resolve())
    catalog = json.loads((args.prior / "capture.json").read_text())
    prior_path = args.prior / "tensors.npz"
    expected_prior = {"cpu": "1205ca14d8c01bc1126c4cc42623a6339af4edc63f33f8a1dc686748a65de0ee",
                      "gpu": "157fd928cc98b4b1efd9eda67f020793df62dc57225396df082978341e1624c5"}[args.device]
    if (digest(prior_path) != expected_prior or catalog["capture_sha256"] != expected_prior
            or catalog["inputs"] != BASELINE
            or catalog["runtime"]["device"] != args.device):
        raise ValueError("Phase 3.7 capture mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    captured, result = {}, {}

    with np.load(prior_path, allow_pickle=False) as prior, (
        np.load(args.reference / "values.npz", allow_pickle=False)
        if args.reference else np.load(prior_path, allow_pickle=False)
    ) as reference:
        if args.reference:
            reference_meta = json.loads((args.reference / "results.json").read_text())
            if (digest(args.reference / "values.npz") != reference_meta["sha256"]
                    or reference_meta["prior"] != catalog["capture_sha256"]):
                raise ValueError("Phase 3.8 reference mismatch")

        def old(key):
            return b.tensor(prior[catalog["rows"][key]["sha256"]])

        def save(name, value, gradient=False):
            b.assert_device(value)
            array = b.array(value)
            if args.reference:
                result[name] = (gradient_metrics if gradient else lambda a, c: metrics(a, c, "activation"))(
                    reference[name], array)
            else:
                captured[name] = array

        def probe(name, fn, operands, dy=None, module=None):
            b.reset(True)
            values = [b.tensor(reference[f"{name}/input/{i}"] if args.reference else b.array(value), True)
                      for i, value in enumerate(operands)]
            out = fn(*values)
            if args.reference:
                upstream = b.tensor(reference[f"{name}/upstream"])
            else:
                upstream = dy if dy is not None else b.tensor(
                    np.random.default_rng(3808).standard_normal(tuple(out.shape), dtype=np.float32) * 0.01)
                for i, value in enumerate(values):
                    save(f"{name}/input/{i}", value)
                save(f"{name}/upstream", upstream)
            save(f"{name}/output", out)
            parameters = {k: p for k, p in b.params.items() if module and k.startswith(module + ".")
                          and (not p.stop_gradient if b.paddle else p.requires_grad)}
            targets = {**{f"input/{i}": value for i, value in enumerate(values)}, **parameters}
            grad = b.fw.grad if b.paddle else b.fw.autograd.grad
            gradients = vjp(grad, out, upstream, targets)
            for k, value in gradients.items():
                if k in parameters and b.paddle and isinstance(b.named[k.rsplit(".", 1)[0]], b.fw.nn.Linear) and k.endswith(".weight"):
                    value = value.transpose([1, 0])
                save(f"{name}/grad/{k}", value, True)

        b.reset(True)
        for branch in ("ctc", "nrtr"):
            key = f"local/backbone/{branch}/"
            prefix = branch + "/"
            rep_input = old(key + REP + "/input/0")
            three = old(f"natural/forward/{REP}.conv.bn")
            one = old(f"natural/forward/{REP}.conv1")
            rep = old(f"natural/forward/{REP}")
            dy_rep = old(key + REP + "/upstream")
            dy_sum = old(key + REP + ".bn/grad/input_0")
            probe(prefix+"rep/identity", lambda x: x + 0., (rep_input,), dy_sum)
            probe(prefix+"rep/sum", lambda a, c, d: (a + c) + d, (three, one, rep_input), dy_sum)
            for leaf in (".conv", ".conv.conv", ".conv.bn", ".conv1", ".bn", ""):
                module = REP + leaf
                operand = rep_input if leaf == ".conv" else old(key + module + "/input/0")
                dy = dy_sum if leaf == ".conv" else old(key + module + "/upstream")
                probe(prefix+"rep/module"+leaf, b.named[module], (operand,), dy, module)
            se = b.named[SE]
            def pool(x):
                return x.mean([2, 3], keepdim=True) if b.paddle else x.mean((2, 3), keepdim=True)
            # Paddle uses the same explicit spatial mean and activated 1x1 SE path.
            pooled = pool(rep)
            first = se.conv1(pooled)
            activated = b.fw.nn.functional.relu(first)
            second = se.conv2(activated)
            gate = b.fw.nn.functional.hardsigmoid(second)
            probe(prefix+"se/mean", pool, (rep,))
            probe(prefix+"se/conv1", se.conv1, (pooled,), module=SE+".conv1")
            probe(prefix+"se/relu", b.fw.nn.functional.relu, (first,))
            probe(prefix+"se/conv2", se.conv2, (activated,), module=SE+".conv2")
            probe(prefix+"se/hardsigmoid", b.fw.nn.functional.hardsigmoid, (second,))
            probe(prefix+"se/multiply", lambda x, y: x * y, (rep, gate))
            probe(prefix+"se/composed", se, (rep,), module=SE)
            for leaf in ("token_mixer", "channel_mixer", "channel_mixer.expand.conv",
                         "channel_mixer.expand.bn", "channel_mixer.compress.conv", "channel_mixer.compress.bn"):
                module = BLOCK + "." + leaf
                probe(prefix+leaf, b.named[module], (old(key+module+"/input/0"),),
                      old(key+module+"/upstream"), module)
            probe(prefix+"residual", lambda x, y: x+y,
                  (old("natural/forward/"+BLOCK+".token_mixer"),
                   old("natural/forward/"+BLOCK+".channel_mixer")), old(key+BLOCK+"/upstream"))

            for index in (0, 1):
                root = f"backbone.blocks5.{index}"
                names = (root+".channel_mixer.compress.conv", root+".channel_mixer.compress.bn")
                inputs, outputs = {}, {}
                handles = []
                for name in names:
                    def capture(_layer, operands, value, key=name):
                        inputs[key], outputs[key] = operands[0], value
                    layer = b.named[name]
                    register = layer.register_forward_post_hook if b.paddle else layer.register_forward_hook
                    handles.append(register(capture))
                b.reset(True)
                try:
                    b.named[root](b.tensor(b.array(old(key+root+"/input/0")), True))
                finally:
                    for handle in handles:
                        handle.remove()
                grad = b.fw.grad if b.paddle else b.fw.autograd.grad
                block_dy = old(key+root+"/upstream")
                # The final BN is directly on the additive residual path. Clone
                # the convolution output to isolate BN and avoid Paddle alias targets.
                norm = b.named[names[1]]
                conv_out = b.tensor(b.array(outputs[names[0]]), True)
                norm_out = norm(conv_out)
                bn_grad = vjp(grad, norm_out, block_dy,
                              {"input": conv_out, "weight": norm.weight, "bias": norm.bias})
                seeds = {names[0]: bn_grad["input"], names[1]: block_dy}
                for name in names:
                    probe(prefix+name, b.named[name], (inputs[name],), seeds[name], name)

        probe("decoder/kv", b.named[KV], (old("local/"+KV+"/input/0"),),
              old("local/"+KV+"/upstream"), KV)
        indices = np.array([[80, 80, 216, 0], [216, 80, 0, 80]], dtype=np.int64)
        # Replay the actual embedding module, including padding and sqrt(512) scaling.
        layer = b.named[EMB]
        embedding_grads = {}
        for order in ("forward", "reverse"):
            ix = b.tensor(indices if order == "forward" else indices[:, ::-1].copy())
            y = layer(ix)
            dy = np.random.default_rng(3808).standard_normal(tuple(y.shape), dtype=np.float32)
            if order == "reverse":
                dy = dy[:, ::-1].copy()
            save(f"decoder/embedding/{order}/output", y)
            grad = b.fw.grad if b.paddle else b.fw.autograd.grad
            value = vjp(grad, y, b.tensor(dy), {"weight": layer.embedding.weight})["weight"]
            save(f"decoder/embedding/{order}/grad", value, True)
            embedding_grads[order] = b.array(value)
        result["embedding_order"] = gradient_metrics(embedding_grads["forward"], embedding_grads["reverse"])
    path = args.output / "values.npz"
    if not args.reference:
        np.savez_compressed(path, **captured)
    (args.output / "results.json").write_text(json.dumps({
        "runtime": b.versions(), "prior": catalog["capture_sha256"], "inputs": BASELINE,
        "seed": 3808, "sha256": digest(path) if path.exists() else None, "rows": result,
    }, indent=2, allow_nan=False), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=("paddle", "torch"))
    parser.add_argument("device", choices=("cpu", "gpu"))
    for name in ("prior", "output", "source", "original", "checkpoint"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    run(parser.parse_args())
