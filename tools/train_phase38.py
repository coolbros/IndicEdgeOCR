"""Validate 100 supervised updates from frozen PP-OCRv6 using Accelerate."""

import argparse
import json
from pathlib import Path
import random
from time import perf_counter

from accelerate import Accelerator
from accelerate.utils import GradScalerKwargs
import numpy as np
from safetensors.torch import load_file
import torch

from convert_paddle_checkpoint import digest
from ppocrv6_torch import PPOCRv6Medium
from ppocrv6_torch.data import Vocabulary, decode_image, preprocess, resize_image
from ppocrv6_torch.losses import multi_loss
from ppocrv6_torch.train_data import augment, concatenate, multiscale_batches
from ppocrv6_torch.training import cosine_schedule, optimizer_for
from trace_parity import BASELINE
from train_smoke import restore_checkpoint, save_checkpoint
from verify_dataset_splits import audit


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def batch_for(dataset, records, width, height, indices, vocab, device):
    batch = []
    for index in indices:
        image, label = dataset(index)
        ext = [dataset(random.randrange(len(records))) for _ in range(2)]
        # The upstream dual encoder rejects the 24- and 25-token concat outputs.
        image, label = concatenate(image, label, ext, max_length=23)
        x, ratio = resize_image(augment(image), width=width, height=height)
        ctc, nrtr, length = vocab.encode(label)
        batch.append((x, ctc, nrtr, np.int64(length), np.float32(ratio)))
    return [torch.from_numpy(np.stack(values)).to(device) for values in zip(*batch)]


def validate(model, records, root, vocab, device):
    model.eval()
    correct, count = 0, 0
    with torch.no_grad():
        for start in range(0, len(records), 16):
            part = records[start:start + 16]
            x = np.stack([preprocess((root / row["image_path"]).read_bytes())[0] for row in part])
            predictions = vocab.decode(model(torch.from_numpy(x).to(device)).float().cpu().numpy())
            correct += sum(text == row["transcription"] for (text, _), row in zip(predictions, part, strict=True))
            count += len(part)
    return {"exact": correct, "samples": count, "accuracy": correct / count}


def run(args):
    initial_hash = digest(args.initial)
    if initial_hash != BASELINE["checkpoint"]:
        raise ValueError("Training must initialize from the frozen checkpoint")
    paths = [args.dataset / f"{split}.jsonl" for split in ("train", "validation")]
    splits = audit(paths, args.dataset)
    expected = json.loads((args.dataset / "audit.json").read_text(encoding="utf8"))
    if not splits["passed"] or splits["manifest_sha256"] != expected["manifest_sha256"]:
        raise ValueError("Training split audit failed")
    train, validation = map(rows, paths)
    vocab = Vocabulary(args.dictionary)
    torch.set_num_threads(2)
    accelerator = Accelerator(mixed_precision=args.precision, kwargs_handlers=[
        GradScalerKwargs(init_scale=1.0, growth_interval=1000),
    ])
    device = accelerator.device
    if accelerator.num_processes != 1 or device.type != "cuda":
        raise RuntimeError("This run requires one CUDA process")
    config = {"epochs": 2, "steps_per_epoch": 51, "warmup_epochs": 0.1, "first_bs": 8,
              "lr": 5e-5, "seed": 3801, "precision": args.precision, "initial_sha256": initial_hash,
              "manifest_sha256": splits["manifest_sha256"], "dictionary_sha256": digest(args.dictionary)}
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    cache = {}
    def sample(index):
        row = train[index]
        key = row["image_sha256"]
        if key not in cache:
            cache[key] = decode_image((args.dataset / row["image_path"]).read_bytes())
        return cache[key].copy(), row["transcription"]

    original = load_file(args.initial)
    model = PPOCRv6Medium()
    model.load_state_dict(original, strict=True)
    optimizer = optimizer_for(model, config["lr"])
    scheduler = cosine_schedule(optimizer, config["steps_per_epoch"], config["epochs"], config["warmup_epochs"])
    model, optimizer = accelerator.prepare(model, optimizer)
    raw = accelerator.unwrap_model(model)
    targets = ("backbone.conv1.stem1.conv.weight", "head.ctc_head.fc.weight",
               "head.gtc_head.tgt_word_prj.weight")
    latest = args.output / "latest.pt"
    if args.resume:
        restored = restore_checkpoint(latest, model, optimizer, scheduler, accelerator, config)
        state = {k: restored[k] for k in ("config", "epoch", "global_step", "history", "validation", "best")}
        if state["global_step"] != state["epoch"] * config["steps_per_epoch"] or scheduler.last_epoch != state["global_step"]:
            raise RuntimeError("Resume cursor and scheduler disagree")
        if {int(v["step"]) for v in optimizer.state.values()} != {state["global_step"]}:
            raise RuntimeError("Resume Adam state did not restore all update counts")
        state["resumed_from"] = state["global_step"]
        state["restored_sha256"] = digest(latest)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        state = dict(config=config, epoch=0, global_step=0, history=[], best=-1.,
                     validation=[validate(model, validation, args.dataset, vocab, device)])
    started = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(state["epoch"], 1 if args.stop_at_boundary else config["epochs"]):
        model.train()
        batches = multiscale_batches(len(train), config["first_bs"], config["seed"], epoch)
        for index, (width, height, ids) in enumerate(batches[:config["steps_per_epoch"]]):
            batch = batch_for(sample, train, width, height, ids, vocab, device)
            for attempt in range(8):
                optimizer.zero_grad(set_to_none=True)
                lr = optimizer.param_groups[0]["lr"]
                x, ctc, nrtr, lengths, ratio = batch
                losses = multi_loss(model(x, (ctc, nrtr, lengths, ratio)), ctc, nrtr, lengths)
                values = {k: float(v.detach()) for k, v in losses.items()}
                if not all(np.isfinite(v) for v in values.values()):
                    raise RuntimeError(f"Nonfinite losses at update {state['global_step']}: {values}")
                accelerator.backward(losses["loss"])
                accelerator.unscale_gradients(optimizer)
                norms = {k: float(raw.get_parameter(k).grad.float().norm()) for k in targets}
                optimizer.step()
                if not accelerator.optimizer_step_was_skipped:
                    break
            else:
                raise RuntimeError("Repeated AMP overflow")
            if any(not np.isfinite(v) or v <= 0 for v in norms.values()):
                raise RuntimeError(f"Invalid tracked gradient norm: {norms}")
            scheduler.step()
            state["global_step"] += 1
            state["history"].append(dict(step=state["global_step"], lr=lr, losses=values,
                                          norms=norms, overflow_retries=attempt, height=height, batch=len(ids)))
            if (index + 1) % 10 == 0:
                print(f"step {state['global_step']}: {values}", flush=True)
        state["epoch"] = epoch + 1
        validation_metrics = validate(model, validation, args.dataset, vocab, device)
        state["validation"].append(validation_metrics)
        if validation_metrics["accuracy"] > state["best"]:
            state["best"] = validation_metrics["accuracy"]
            save_checkpoint(args.output / "best.pt", model, optimizer, scheduler, accelerator, state)
        save_checkpoint(latest, model, optimizer, scheduler, accelerator, state)
    changes = {k: float((raw.get_parameter(k).detach().cpu() - original[k]).norm()) for k in targets}
    if not all(np.isfinite(v) and v > 0 for v in changes.values()):
        raise RuntimeError("Expected nonzero finite parameter updates")
    if digest(args.initial) != initial_hash:
        raise RuntimeError("Initial checkpoint changed")
    nonzero_lr_steps = sum(row["lr"] > 0 for row in state["history"])
    if args.resume and nonzero_lr_steps < 100:
        raise RuntimeError("Fewer than 100 parameter-update steps with nonzero learning rate")
    record = dict(**state, parameter_l2_changes=changes, seconds=perf_counter() - started,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(), device=torch.cuda.get_device_name(),
                  torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  latest_sha256=digest(latest), best_sha256=digest(args.output / "best.pt"),
                  optimizer_state_entries=len(optimizer.state), nonzero_lr_steps=nonzero_lr_steps)
    name = "resume-results.json" if args.resume else "first-results.json"
    with (args.output / name).open("x", encoding="utf8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "dictionary", "initial", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--precision", choices=("no", "fp16"), default="fp16")
    parser.add_argument("--stop-at-boundary", action="store_true")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
