"""Run or resume a short audited HierText recognition training smoke test."""

import argparse
import json
import os
from pathlib import Path
import random

from accelerate import Accelerator
from accelerate.utils import GradScalerKwargs
import numpy as np
from safetensors.torch import load_file
import torch

from convert_paddle_checkpoint import digest
from ppocrv6_torch import PPOCRv6Medium
from ppocrv6_torch.data import Vocabulary, preprocess
from ppocrv6_torch.losses import multi_loss
from ppocrv6_torch.training import cosine_schedule, optimizer_for
from verify_dataset_splits import SPLITS, audit


def save_checkpoint(path, model, optimizer, scheduler, accelerator, state):
    payload = {
        **state, "model": accelerator.get_state_dict(model),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": accelerator.scaler.state_dict() if accelerator.scaler else None,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    checksum = digest(temporary)
    os.replace(temporary, path)
    metadata = path.with_suffix(".json")
    temporary = metadata.with_suffix(".tmp")
    temporary.write_text(json.dumps({**state, "sha256": checksum}, indent=2),
                         encoding="utf-8")
    os.replace(temporary, metadata)


def restore_checkpoint(path, model, optimizer, scheduler, accelerator, config):
    metadata = json.loads(path.with_suffix(".json").read_text())
    if digest(path) != metadata["sha256"] or metadata["config"] != config:
        raise ValueError("Checkpoint checksum or training configuration differs")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    accelerator.unwrap_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if accelerator.scaler:
        accelerator.scaler.load_state_dict(checkpoint["scaler"])
    rng = checkpoint["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return checkpoint


def prepared_records(root, split, vocab, cache):
    manifest = root / f"{split}.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        cache_path = cache / f"bgr48x320-v1-{row['image_sha256']}.npy"
        if not cache_path.exists():
            image, _ = preprocess((root / row["image_path"]).read_bytes())
            with cache_path.open("xb") as stream:
                np.save(stream, image, allow_pickle=False)
        image = np.load(cache_path, allow_pickle=False)
        ctc, nrtr, length = vocab.encode(row["transcription"])
        yield image, ctc, nrtr, length, row["transcription"]


def validate(model, records, vocab, device):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.stack([row[0] for row in records])).to(device)
        decoded = vocab.decode(model(x).float().cpu().numpy())
    matches = sum(text == row[-1] for (text, _), row in zip(decoded, records))
    return matches / len(records)


def run(args):
    torch.set_num_threads(2)
    accelerator = Accelerator(mixed_precision=args.precision,
                              kwargs_handlers=[GradScalerKwargs(init_scale=1.0)])
    if accelerator.num_processes != 1:
        raise ValueError("The resume smoke test uses one process")
    if args.precision != "no" and accelerator.device.type != "cuda":
        raise ValueError("Mixed-precision smoke testing requires CUDA")
    audit_now = audit([args.dataset / f"{s}.jsonl" for s in SPLITS], args.dataset)
    prior = json.loads((args.dataset / "audit.json").read_text())
    if not audit_now["passed"] or audit_now["manifest_sha256"] != prior["manifest_sha256"]:
        raise ValueError("The immutable split audit failed")
    hashes = audit_now["manifest_sha256"]
    if set(audit_now["counts"]) != set(SPLITS):
        raise ValueError("All three audited partitions must be present")
    config = {"precision": args.precision, "epochs": 2, "steps_per_epoch": 2, "lr": 0.00005,
              "warmup_epochs": 0, "seed": 2026, "manifest_sha256": hashes,
              "dictionary_sha256": digest(args.dictionary), "initial_sha256": digest(args.initial)}
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.dataset.parent / "preprocessed-cache"
    cache.mkdir(exist_ok=True)
    vocab = Vocabulary(args.dictionary)
    train = list(prepared_records(args.dataset, "train", vocab, cache))
    validation = list(prepared_records(args.dataset, "validation", vocab, cache))
    batch = [torch.from_numpy(np.stack(values)).to(accelerator.device)
             for values in zip(*(row[:4] for row in train))]
    model = PPOCRv6Medium()
    model.load_state_dict(load_file(args.initial))
    optimizer = optimizer_for(model, config["lr"])
    scheduler = cosine_schedule(optimizer, 2, epochs=2, warmup_epochs=0)
    model, optimizer = accelerator.prepare(model, optimizer)
    state = {"config": config, "epoch": 0, "global_step": 0, "best_validation": -1.0,
             "history": [], "resumed": False}
    latest = args.output / "latest.pt"
    if args.resume:
        saved = restore_checkpoint(latest, model, optimizer, scheduler, accelerator, config)
        state = {key: saved[key] for key in state}
        if state["global_step"] != state["epoch"] * 2 or scheduler.last_epoch != state["global_step"]:
            raise ValueError("Restored epoch, scheduler, and global step disagree")
        steps = [int(item["step"]) for item in optimizer.state.values()]
        if not steps or min(steps) != max(steps) or steps[0] != state["global_step"]:
            raise ValueError("Restored Adam step does not match the checkpoint")
        state["resumed"] = True
        state["resumed_from_step"] = state["global_step"]
    device = accelerator.device
    target_names = ("backbone.conv1.stem1.conv.weight", "head.ctc_head.fc.weight",
                    "head.gtc_head.tgt_word_prj.weight")
    for epoch in range(state["epoch"], 1 if args.stop_after_first_epoch else config["epochs"]):
        model.train()
        updates = 0
        attempts = 0
        while updates < 2:
            attempts += 1
            if attempts > 10:
                raise RuntimeError("Could not complete two finite updates in ten attempts")
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch[0], (*batch[1:], None))
            losses = multi_loss(outputs, *batch[1:])
            loss = losses["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            accelerator.backward(loss)
            accelerator.unscale_gradients(optimizer)
            gradients = {}
            for name, parameter in accelerator.unwrap_model(model).named_parameters():
                if name in target_names:
                    value = float(parameter.grad.float().norm().item())
                    gradients[name] = value
            optimizer.step()
            if accelerator.optimizer_step_was_skipped:
                print(f"FP16 overflow, retrying at scale {accelerator.scaler.get_scale()}", flush=True)
                continue
            if any(not np.isfinite(value) or value == 0 for value in gradients.values()):
                raise RuntimeError(f"Invalid gradient norms: {gradients}")
            updates += 1
            scheduler.step()
            state["global_step"] += 1
            state["history"].append({"step": state["global_step"], "loss": float(loss.detach()),
                                     "lr": scheduler.get_last_lr()[0], "gradient_norms": gradients})
        state["epoch"] = epoch + 1
        metric = validate(model, validation, vocab, device)
        state["validation_exact_match"] = metric
        if metric > state["best_validation"]:
            state["best_validation"] = metric
            save_checkpoint(args.output / "best.pt", model, optimizer, scheduler, accelerator, state)
        save_checkpoint(latest, model, optimizer, scheduler, accelerator, state)
    if not args.stop_after_first_epoch and state["history"][-1]["loss"] >= state["history"][0]["loss"]:
        raise RuntimeError("Training loss did not decrease across the smoke run")
    (args.output / "smoke-results.json").write_text(json.dumps({
        **state, "device": str(device), "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda" else None, "torch": torch.__version__, "cuda": torch.version.cuda,
        "optimizer_state_entries": len(optimizer.state), "scheduler_epoch": scheduler.last_epoch,
        "best_sha256": digest(args.output / "best.pt"), "latest_sha256": digest(latest),
    }, indent=2), encoding="utf-8")
    print(f"Smoke completed epoch={state['epoch']} step={state['global_step']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "dictionary", "initial", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--stop-after-first-epoch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
