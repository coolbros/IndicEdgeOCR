"""Paired FP32 optimizer, learning-rate and fixed-logit loss diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np

from phase36_experiment import Backend, LABELS
from trace_parity import BASELINE, metrics
from convert_paddle_checkpoint import digest


def run(args):
    for file, key in ((args.original, "original"), (args.checkpoint, "checkpoint"), (args.archive, "archive")):
        if digest(file) != BASELINE[key]:
            raise ValueError(f"Unexpected {key} digest")
    b = Backend(args.framework, args.device, args.original.resolve(), args.checkpoint.resolve(), args.source.resolve())
    fw = b.fw
    initial = np.array([0.125, -0.25, 1e-8], "f")
    gradients = (np.array([0.1, -0.2, 1e-9], "f"), np.array([-0.4, 0.3, -2e-9], "f"))
    decays = (0., 1e-5, 3e-5)
    if b.paddle:
        parameters = [fw.create_parameter(initial.shape, "float32") for _ in decays]
        for p in parameters:
            p.set_value(initial)
        scheduler = fw.optimizer.lr.LinearWarmup(
            fw.optimizer.lr.CosineAnnealingDecay(0.0005, T_max=12), 3, 0., 0.0005)
        optimizer = fw.optimizer.Adam(learning_rate=0.0005, parameters=[
            {"params": [p], "weight_decay": (fw.regularizer.L2Decay(d) if d else None)}
            for p, d in zip(parameters, decays)])
    else:
        from ppocrv6_torch.training import cosine_schedule
        parameters = [fw.nn.Parameter(b.tensor(initial.copy())) for _ in decays]
        optimizer = fw.optim.Adam([{"params": [p], "weight_decay": d}
                                   for p, d in zip(parameters, decays)], lr=0.0005)
        schedule_optimizer = fw.optim.SGD([fw.nn.Parameter(b.tensor(initial.copy()))], lr=0.0005)
        scheduler = cosine_schedule(schedule_optimizer, 3, epochs=4, warmup_epochs=1)
    result = {"runtime": b.versions(), "optimizer": [], "schedule": []}
    for grad in gradients:
        optimizer.clear_grad() if b.paddle else optimizer.zero_grad(set_to_none=True)
        sum((p * b.tensor(grad)).sum() for p in parameters).backward()
        optimizer.step()
        state = []
        for p in parameters:
            if b.paddle:
                m = optimizer._get_accumulator("moment1", p)
                v = optimizer._get_accumulator("moment2", p)
            else:
                m, v = (optimizer.state[p][k] for k in ("exp_avg", "exp_avg_sq"))
            state.append({name: b.array(value).tolist() for name, value in (("parameter", p), ("m", m), ("v", v))})
        result["optimizer"].append(state)
    for _ in range(16):
        result["schedule"].append(float(scheduler() if b.paddle else scheduler.get_last_lr()[0]))
        if not b.paddle:
            schedule_optimizer.step()
        scheduler.step()

    with np.load(args.archive) as fixture:
        batch = [b.tensor(fixture[f"train/{name}"]) for name in LABELS]
        logits = {head: b.tensor(fixture[f"train/{name}_logits"], True)
                  for head, name in (("ctc", "ctc"), ("gtc", "nrtr"))}
    losses = b.losses(logits, batch)
    tensors = {}
    for name, head in (("CTCLoss", "ctc"), ("NRTRLoss", "gtc")):
        loss = losses[name]
        tensors[name] = b.array(loss)
        tensors[name+"_grad"] = b.array(b.grad(loss, [logits[head]])[0])
    args.output.mkdir(parents=True, exist_ok=False)
    if args.reference:
        prior = json.loads((args.reference / "optimizer-loss.json").read_text())
        with np.load(args.reference / "loss.npz") as expected:
            result["loss"] = {k: metrics(expected[k], v, "activation" if "grad" in k else "loss")
                              for k, v in tensors.items()}
        result["schedule_max_abs"] = max(abs(a-c) for a,c in zip(prior["schedule"], result["schedule"], strict=True))
        result["optimizer_max_abs"] = [{name: max(float(np.max(np.abs(np.array(a[name]) - c[name])))
                                                    for a, c in zip(left, right, strict=True))
                                        for name in ("parameter", "m", "v")}
                                       for left, right in zip(prior["optimizer"], result["optimizer"], strict=True)]
    else:
        np.savez_compressed(args.output / "loss.npz", **tensors)
    (args.output / "optimizer-loss.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=("paddle", "torch"))
    parser.add_argument("device", choices=("cpu", "gpu"))
    for name in ("archive", "output", "source", "original", "checkpoint"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    run(parser.parse_args())
