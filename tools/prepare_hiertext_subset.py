"""Create a small recognition fixture from official HierText split archives."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import tarfile

import cv2
import numpy as np

from verify_dataset_splits import SPLITS, audit
from ppocrv6_torch.data import Vocabulary


RELEASE = "70b6620b2b112597d8219e11eee9773a1403827c"


def crop_word(image, vertices):
    x, y = np.asarray(vertices).T
    left, right = max(0, math.floor(x.min())), min(image.shape[1], math.ceil(x.max()) + 1)
    top, bottom = max(0, math.floor(y.min())), min(image.shape[0], math.ceil(y.max()) + 1)
    return image[top:bottom, left:right]


def prepare(raw, output, vocab, retrieved_on, limit):
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    groups = {}
    exclusions = Counter()
    records = {}
    annotations = {}
    for split in SPLITS:
        annotation_path = raw / f"{split}.jsonl.gz"
        with gzip.open(annotation_path, "rt", encoding="utf-8") as stream:
            rows = json.load(stream)["annotations"]
        annotations[split] = {"images": len(rows), "sha256": hashlib.sha256(
            annotation_path.read_bytes()).hexdigest()}
        records[split] = {}
        for row in rows:
            source = row["image_id"]
            if source in groups:
                raise ValueError(f"Repeated source in official metadata: {source}")
            groups[source] = split
            records[split][source] = row

    manifests = []
    for split in SPLITS:
        source_url = f"https://open-images-dataset.s3.amazonaws.com/ocr/{split}.tgz"
        prefix = raw / f"{split}.tgz.prefix"
        selected = []
        with tarfile.open(prefix, "r|gz") as archive:
            for member in archive:
                if not member.isfile() or Path(member.name).suffix != ".jpg":
                    continue
                source = Path(member.name).stem
                row = records[split][source]
                content = archive.extractfile(member).read()
                img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                if img is None or list(img.shape[:2]) != [row["image_height"], row["image_width"]]:
                    raise ValueError(f"Source dimensions disagree: {source}")
                candidate = None
                for i, paragraph in enumerate(row["paragraphs"]):
                    for j, line in enumerate(paragraph["lines"]):
                        for k, word in enumerate(line["words"]):
                            text = word["text"]
                            if (not word["legible"] or word["vertical"] or
                                    not 3 <= len(text) <= 18 or any(c not in vocab.indices for c in text)):
                                exclusions[f"{split}: unsuitable label or orientation"] += 1
                                continue
                            crop = crop_word(img, word["vertices"])
                            if min(crop.shape[:2]) < 14 or crop.shape[1] < crop.shape[0]:
                                exclusions[f"{split}: unsuitable crop"] += 1
                                continue
                            candidate = (i, j, k, text, crop)
                            break
                        if candidate:
                            break
                    if candidate:
                        break
                if candidate is None:
                    exclusions[f"{split}: source has no eligible word"] += 1
                    continue
                i, j, k, text, crop = candidate
                image_path = f"images/{source}_{i}_{j}_{k}.png"
                ok, png = cv2.imencode(".png", crop)
                if not ok:
                    raise ValueError("Could not encode HierText crop")
                (output / image_path).write_bytes(png.tobytes())
                selected.append({
                    "sample_id": f"hiertext:{source}:{i}:{j}:{k}", "image_path": image_path,
                    "transcription": text, "dataset_source": "HierText",
                    "source_document_id": source, "group_namespace": "Open Images",
                    "split": split, "original_split": split, "license": "CC-BY-SA-4.0",
                    "image_sha256": hashlib.sha256(png.tobytes()).hexdigest(),
                    "source_image_sha256": hashlib.sha256(content).hexdigest(),
                    "source_url": source_url, "retrieved_on": retrieved_on,
                    "annotation_commit": RELEASE, "annotation_index": [i, j, k],
                })
                if len(selected) == limit:
                    break
        if len(selected) != limit:
            raise ValueError(f"Expected {limit} eligible {split} images in archive prefix")
        manifest = output / f"{split}.jsonl"
        with manifest.open("x", encoding="utf-8", newline="\n") as stream:
            for row in selected:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        manifests.append(manifest)
    result = audit(manifests, output)
    result.update(official_metadata=annotations, prefix_selection_exclusions=dict(exclusions))
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise ValueError("Dataset split audit failed. See audit.json.")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--retrieved-on", required=True)
    parser.add_argument("--per-split", type=int, default=4)
    args = parser.parse_args()
    prepare(args.raw, args.output, Vocabulary(args.dictionary), args.retrieved_on, args.per_split)
