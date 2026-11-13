"""Prepare, audit and evaluate a larger official HierText validation subset."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
from time import perf_counter
import unicodedata

import cv2
import numpy as np

from convert_paddle_checkpoint import digest
from phase36_experiment import Backend
from ppocrv6_torch.data import Vocabulary, preprocess
from prepare_hiertext_subset import RELEASE, crop_word
from trace_parity import BASELINE
from verify_dataset_splits import audit


SOURCE = "https://open-images-dataset.s3.amazonaws.com/ocr/validation.tgz"
VALIDATION_HASH = "ce7085d8bf24c2f13df852715c34494e6748cafcabb4bfc7a6e2bb320f58d23a"
VALIDATION_PREFIX_HASH = "ef800c539c5735d48a6bc809607fcd50b880bda096d572d902ef5f58acbb9dd1"
TEST_MANIFEST_HASH = "19cff4ad576d543d862bf49fa1138999ecb7a6023135dfe92a8abf710365dfb7"


def write_json(path, value):
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def prepare(args):
    if digest(args.annotations) != VALIDATION_HASH:
        raise ValueError("Unexpected official validation annotation checksum")
    if digest(args.prefix) != VALIDATION_PREFIX_HASH:
        raise ValueError("Unexpected official validation archive prefix checksum")
    prior = args.dataset / "train.jsonl"
    if digest(prior) != BASELINE["train_manifest"]:
        raise ValueError("Unexpected training split")
    if digest(args.dataset / "test.jsonl") != TEST_MANIFEST_HASH:
        raise ValueError("Unexpected test split")
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    (root / "images").mkdir()
    for split in ("train", "test"):
        manifest = args.dataset / f"{split}.jsonl"
        shutil.copyfile(manifest, root / manifest.name)
        for line in manifest.read_text(encoding="utf8").splitlines():
            relative = json.loads(line)["image_path"]
            shutil.copyfile(args.dataset / relative, root / relative)
    with gzip.open(args.annotations, "rt", encoding="utf8") as stream:
        annotations = {row["image_id"]: row for row in json.load(stream)["annotations"]}
    selected, groups, exclusions = [], set(), Counter()
    with tarfile.open(args.prefix, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or Path(member.name).suffix != ".jpg":
                continue
            source = Path(member.name).stem
            row = annotations[source]
            source_bytes = archive.extractfile(member).read()
            image = cv2.imdecode(np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None or list(image.shape[:2]) != [row["image_height"], row["image_width"]]:
                raise ValueError(f"Image dimensions differ from annotation for {source}")
            if source in groups:
                raise ValueError(f"Repeated image in validation archive: {source}")
            groups.add(source)
            source_sha = hashlib.sha256(source_bytes).hexdigest()
            count = 0
            for i, paragraph in enumerate(row["paragraphs"]):
                for j, line in enumerate(paragraph["lines"]):
                    for k, word in enumerate(line["words"]):
                        if count >= 12 or len(selected) >= args.limit:
                            break
                        text = word["text"]
                        if not word["legible"] or word["vertical"] or not text.strip():
                            exclusions["illegible_vertical_or_empty"] += 1
                            continue
                        crop = crop_word(image, word["vertices"])
                        if min(crop.shape[:2]) < 10:
                            exclusions["small_crop"] += 1
                            continue
                        ok, png = cv2.imencode(".png", crop)
                        if not ok:
                            raise ValueError(f"Could not encode crop from {source}")
                        sample = f"{source}_{i}_{j}_{k}"
                        image_path = f"images/{sample}.png"
                        content = png.tobytes()
                        (root / image_path).write_bytes(content)
                        count += 1
                        selected.append({
                            "sample_id": "hiertext:" + sample, "image_path": image_path,
                            "transcription": text, "dataset_source": "HierText",
                            "source_document_id": source, "group_namespace": "Open Images",
                            "split": "validation", "original_split": "validation",
                            "license": "CC-BY-SA-4.0", "source_url": SOURCE,
                            "annotation_commit": RELEASE, "retrieved_on": args.retrieved_on,
                            "image_sha256": hashlib.sha256(content).hexdigest(),
                            "source_image_sha256": source_sha, "handwritten": word["handwritten"],
                            "annotation_index": [i, j, k],
                        })
            if len(selected) >= args.limit:
                break
    if len(selected) != args.limit:
        raise ValueError(f"Archive prefix supplied only {len(selected)} eligible validation crops")
    candidate = root / "validation-candidates.jsonl"
    manifest = root / "validation.jsonl"

    def save(path, rows):
        with path.open("x", encoding="utf8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    save(candidate, selected)
    # Apply only manifest-integrity evidence before running either recognizer.
    initial = audit([root / "train.jsonl", root / "test.jsonl", candidate], root)
    excluded = {pair["samples"][pair["splits"].index("validation")]
                for pair in initial["near_duplicates"] if "validation" in pair["splits"]}
    selected = [row for row in selected if row["sample_id"] not in excluded]
    save(manifest, selected)
    final = audit([root / "train.jsonl", root / "test.jsonl", manifest], root)
    write_json(root / "audit.json", {
        "initial": initial, "final": final, "annotation_sha256": VALIDATION_HASH,
        "archive_prefix_sha256": digest(args.prefix), "excluded_candidate_ids": sorted(excluded),
        "selection_exclusions": dict(exclusions),
    })
    if not final["passed"] or len(selected) < 500:
        raise ValueError("Validation set did not pass audit or has fewer than 500 crops")
    print(f"Audited {len(selected)} validation crops", flush=True)


def evaluate(args):
    root = args.dataset
    if digest(args.original) != BASELINE["original"] or digest(args.checkpoint) != BASELINE["checkpoint"]:
        raise ValueError("Checkpoint digest mismatch")
    manifest = root / "validation.jsonl"
    audit_result = json.loads((root / "audit.json").read_text(encoding="utf8"))["final"]
    if not audit_result["passed"] or digest(manifest) != audit_result["manifest_sha256"][manifest.name]:
        raise ValueError("Dataset audit or manifest mismatch")
    vocab = Vocabulary(args.source / "ppocr/utils/dict/ppocrv6_dict.txt")
    backend = Backend(args.framework, args.device, args.original.resolve(),
                      args.checkpoint.resolve(), args.source.resolve())
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf8").splitlines()]
    buckets = defaultdict(list)
    started = perf_counter()
    for row in rows:
        path = root / row["image_path"]
        if row["split"] != "validation" or digest(path) != row["image_sha256"]:
            raise ValueError("Evaluation image or split mismatch")
        array, _ = preprocess(path.read_bytes(), inference=True)
        buckets[array.shape[-1]].append((row, array))
    preprocessing = perf_counter() - started
    batches = [bucket[i:i + 16] for bucket in buckets.values() for i in range(0, len(bucket), 16)]
    durations, predictions, input_digests = [], [], []
    with backend.fw.no_grad():
        warmup = backend.tensor(np.stack([values for _, values in batches[0]]))
        for _ in range(3):
            backend.model(warmup)
        backend.sync()
        for index, group in enumerate(batches, 1):
            array = np.stack([values for _, values in group])
            input_digests.append(hashlib.sha256(array.tobytes()).hexdigest())
            x = backend.tensor(array)
            backend.sync()
            begin = perf_counter()
            value = backend.model(x)
            backend.sync()
            durations.append(perf_counter() - begin)
            backend.assert_device(value)
            for (row, _), (text, confidence) in zip(group, vocab.decode(backend.array(value))):
                predictions.append({"id": row["sample_id"], "source": row["source_document_id"],
                                    "reference": row["transcription"], "text": text,
                                    "confidence": confidence, "handwritten": row["handwritten"]})
            if index % 20 == 0 or index == len(batches):
                print(f"{args.framework}: {len(predictions)}/{len(rows)} crops", flush=True)
    write_json(args.output, {"runtime": backend.versions(), "manifest_sha256": digest(manifest),
                             "preprocess_ms": preprocessing * 1000,
                             "batch_latency_ms": [x * 1000 for x in durations],
                             "throughput_per_second": len(rows) / sum(durations),
                             "input_batch_sha256": input_digests, "predictions": predictions})


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def compare(args):
    left, right = [json.loads(path.read_text(encoding="utf8")) for path in (args.paddle, args.torch)]
    if (left["manifest_sha256"] != right["manifest_sha256"]
            or left["input_batch_sha256"] != right["input_batch_sha256"]):
        raise ValueError("Evaluations did not consume the same records and preprocessed bytes")
    samples, breakdown = [], defaultdict(Counter)
    for p, t in zip(left["predictions"], right["predictions"], strict=True):
        if (p["id"], p["reference"]) != (t["id"], t["reference"]):
            raise ValueError("Mismatched evaluation example")
        ref = p["reference"]
        normalized = unicodedata.normalize("NFC", ref)
        row = {"id": p["id"], "source": p["source"], "label": ref, "paddle": p["text"],
               "torch": t["text"], "agree": p["text"] == t["text"],
               "confidence_delta": abs(p["confidence"] - t["confidence"]),
               "characters": len(ref), "nfc_characters": len(normalized), "words": len(ref.split())}
        for key, text in (("paddle", p["text"]), ("torch", t["text"])):
            row[key + "_char_edits"] = edit_distance(ref, text)
            row[key + "_nfc_edits"] = edit_distance(normalized, unicodedata.normalize("NFC", text))
            row[key + "_word_edits"] = edit_distance(ref.split(), text.split())
        samples.append(row)
        for group in ("all", "handwritten" if p["handwritten"] else "printed",
                      "ASCII" if ref.isascii() else "non-ASCII",
                      "length_1_to_5" if len(ref) <= 5 else "length_6_plus"):
            breakdown[group].update({key: value for key, value in row.items()
                                     if isinstance(value, (int, float))})
            breakdown[group]["samples"] += 1
    summary = {}
    for group, stats in breakdown.items():
        summary[group] = {"samples": stats["samples"],
                          "exact_agreement": stats["agree"] / stats["samples"]}
        for key in ("paddle", "torch"):
            summary[group][key] = {name: stats[f"{key}_{edits}_edits"] / stats[denom]
                                    for name, edits, denom in (("cer", "char", "characters"),
                                        ("nfc_cer", "nfc", "nfc_characters"), ("wer", "word", "words"))}
    errors = np.array([row["confidence_delta"] for row in samples])
    write_json(args.output, {"summary": summary, "manifest_sha256": left["manifest_sha256"],
                            "max_confidence_difference": float(errors.max()),
                            "mean_confidence_difference": float(errors.mean()),
                            "runtime": {key: {k: v for k, v in value.items()
                                              if k in ("runtime", "throughput_per_second", "batch_latency_ms",
                                                       "preprocess_ms")}
                                        for key, value in (("paddle", left), ("torch", right))},
                            "samples": samples})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function, paths in (
        ("prepare", prepare, ("annotations", "prefix", "dataset", "output")),
        ("evaluate", evaluate, ("dataset", "original", "checkpoint", "source", "output")),
        ("compare", compare, ("paddle", "torch", "output")),
    ):
        command = commands.add_parser(name)
        command.set_defaults(function=function)
        for path in paths:
            command.add_argument("--" + path, type=Path, required=True)
        if name == "prepare":
            command.add_argument("--limit", type=int, default=1080)
            command.add_argument("--retrieved-on", required=True)
        if name == "evaluate":
            command.add_argument("--framework", choices=("paddle", "torch"), required=True)
            command.add_argument("--device", choices=("cpu", "gpu"), required=True)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
