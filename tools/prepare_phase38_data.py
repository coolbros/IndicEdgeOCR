"""Prepare a source-isolated HierText word-crop subset for bounded training."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np

from convert_paddle_checkpoint import digest
from phase36_evaluate import VALIDATION_HASH, write_json
from ppocrv6_torch.data import Vocabulary
from prepare_hiertext_subset import RELEASE, crop_word
from verify_dataset_splits import audit

TRAIN_HASH = "bf7a271c7a016e66ad87556078f0371f8fb12beaa2e2e60fc33458ad57c3e4fd"
BASE = "https://open-images-dataset.s3.amazonaws.com/ocr"


def extract(root, raw, split, limit, vocab, retrieved):
    annotation = raw / f"{split}.jsonl.gz"
    prefix = raw / f"{split}.prefix"
    if digest(annotation) != {"train": TRAIN_HASH, "validation": VALIDATION_HASH}[split]:
        raise ValueError(f"Unexpected {split} annotation digest")
    with gzip.open(annotation, "rt", encoding="utf8") as stream:
        annotations = {item["image_id"]: item for item in json.load(stream)["annotations"]}
    rows, exclusions = [], Counter()
    with tarfile.open(prefix, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or Path(member.name).suffix != ".jpg":
                continue
            source = Path(member.name).stem
            data = annotations[source]
            raw_image = archive.extractfile(member).read()
            source_hash = hashlib.sha256(raw_image).hexdigest()
            image = cv2.imdecode(np.frombuffer(raw_image, np.uint8), cv2.IMREAD_COLOR)
            if image is None or list(image.shape[:2]) != [data["image_height"], data["image_width"]]:
                raise ValueError(f"Unreadable source or wrong size: {source}")
            for i, paragraph in enumerate(data["paragraphs"]):
                for j, line in enumerate(paragraph["lines"]):
                    for k, word in enumerate(line["words"]):
                        label = word["text"]
                        if not word["legible"] or word["vertical"] or not label.strip():
                            exclusions["orientation_or_legibility"] += 1
                            continue
                        if len(label) > 23 or any(c not in vocab.indices for c in label):
                            exclusions["label_not_encodable_in_full"] += 1
                            continue
                        crop = crop_word(image, word["vertices"])
                        if min(crop.shape[:2]) < 10:
                            exclusions["small_crop"] += 1
                            continue
                        ok, encoded = cv2.imencode(".png", crop)
                        if not ok:
                            raise ValueError(f"Unable to encode crop from {source}")
                        sample = f"{source}_{i}_{j}_{k}"
                        path = f"images/{sample}.png"
                        content = encoded.tobytes()
                        (root / path).write_bytes(content)
                        rows.append(dict(sample_id=f"hiertext:{sample}", image_path=path,
                                         transcription=label, source_document_id=source,
                                         source_image_sha256=source_hash, image_sha256=hashlib.sha256(content).hexdigest(),
                                         split=split, original_split=split, group_namespace="Open Images",
                                         dataset_source="HierText", source_url=f"{BASE}/{split}.tgz",
                                         retrieved_on=retrieved, annotation_commit=RELEASE, license="CC-BY-SA-4.0"))
                        if len(rows) == limit:
                            return rows, dict(exclusions)
    raise ValueError(f"Found only {len(rows)} eligible crops in {split}")


def save(path, rows):
    with path.open("x", encoding="utf8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare(args):
    root = args.output
    (root / "images").mkdir(parents=True)
    vocab = Vocabulary(args.dictionary)
    rows = {}
    counts = {}
    for split, target in (("train", 5200), ("validation", 256)):
        rows[split], counts[split] = extract(root, args.raw, split, target, vocab, args.retrieved_on)
        save(root / f"{split}-candidates.jsonl", rows[split])
    before = audit([root / f"{split}-candidates.jsonl" for split in rows], root)
    # Keep the validation partition fixed and exclude flagged training examples.
    exclusions = {pair["samples"][pair["splits"].index("train")] for pair in before["near_duplicates"]}
    for error in before["errors"]:
        if "cross-split exact image" not in error:
            raise ValueError(error)
    val_hashes = {row["image_sha256"] for row in rows["validation"]}
    rows["train"] = [row for row in rows["train"] if row["sample_id"] not in exclusions
                     and row["image_sha256"] not in val_hashes]
    for split, items in rows.items():
        save(root / f"{split}.jsonl", items)
    result = audit([root / f"{split}.jsonl" for split in rows], root)
    if not result["passed"] or len(rows["train"]) < 5000:
        raise ValueError("Final HierText audit or 5,000-crop minimum failed")
    result.update(candidate_counts=before["counts"], candidate_exact_findings=before["errors"],
                  excluded_near_ids=sorted(exclusions), rejected=counts,
                  archive_prefix_sha256={s: digest(args.raw / f"{s}.prefix") for s in rows},
                  statistics={s: {"source_images": len({r["source_document_id"] for r in v}),
                                  "characters": dict(Counter(c for r in v for c in r["transcription"])),
                                  "word_lengths": dict(Counter(len(r["transcription"]) for r in v))}
                              for s, v in rows.items()})
    write_json(root / "audit.json", result)
    print(result["counts"], result["manifest_sha256"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "dictionary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--retrieved-on", required=True)
    prepare(parser.parse_args())
