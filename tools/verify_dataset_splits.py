"""Audit JSONL OCR manifests against real images and cross-split duplication."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


SPLITS = ("train", "validation", "test")
REQUIRED = ("sample_id", "image_path", "transcription", "dataset_source",
            "split", "license", "image_sha256", "retrieved_on", "source_url")


def dhash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return int.from_bytes(np.packbits(small[:, :-1] > small[:, 1:]).tobytes(), "big")


def audit(manifests, root, threshold=4):
    if not 0 <= threshold <= 7:
        raise ValueError("Near-duplicate threshold must be in [0, 7]")
    root = root.resolve()
    seen_ids, seen_hashes, groups = {}, {}, {}
    hashes, buckets = {}, defaultdict(list)
    errors, near = [], []
    counts, missing_groups = Counter(), Counter()
    files = {}
    for manifest in manifests:
        raw = manifest.read_bytes()
        files[manifest.name] = hashlib.sha256(raw).hexdigest()
        for line_number, line in enumerate(raw.splitlines(), 1):
            loc = f"{manifest.name}:{line_number}"
            try:
                row = json.loads(line)
                if any(not isinstance(row.get(k), str) or not row[k].strip() for k in REQUIRED):
                    raise ValueError("missing required data")
                split, sample_id = row["split"], row["sample_id"]
                if split not in SPLITS or row.get("original_split") != split:
                    raise ValueError("unknown split or changed official split")
                if manifest.stem in SPLITS and manifest.stem != split:
                    raise ValueError("row belongs to a different split manifest")
                counts[split] += 1
                path = (root / row["image_path"]).resolve()
                if Path(row["image_path"]).is_absolute() or not path.is_relative_to(root):
                    raise ValueError("image path escapes dataset root")
                content = path.read_bytes()
                image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("unreadable image")
                sha = hashlib.sha256(content).hexdigest()
                if sha != row["image_sha256"]:
                    raise ValueError("incorrect image SHA-256")
                if sample_id in seen_ids:
                    errors.append(f"{loc}: duplicate sample ID {sample_id} ({seen_ids[sample_id]})")
                seen_ids[sample_id] = split
                if sha in seen_hashes and seen_hashes[sha][0] != split:
                    errors.append(f"{loc}: cross-split exact image {sample_id} / {seen_hashes[sha][1]}")
                seen_hashes[sha] = (split, sample_id)
                document = row.get("source_document_id")
                if document is not None:
                    if not isinstance(document, str) or not document.strip():
                        raise ValueError("invalid source-document ID")
                    key = (row.get("group_namespace", row["dataset_source"]), document)
                    if key in groups and groups[key] != split:
                        errors.append(f"{loc}: cross-split source document {document}")
                    groups[key] = split
                else:
                    missing_groups[split] += 1
                fingerprint = dhash(image)
                # A distance of at most four implies a shared byte in this index.
                bands = [(i, (fingerprint >> (8 * i)) & 255) for i in range(8)]
                candidates = {j for band in bands for j in buckets[band]}
                for other in sorted(candidates):
                    other_split, other_id, other_hash = hashes[other]
                    distance = (fingerprint ^ other_hash).bit_count()
                    if other_split != split and distance <= threshold:
                        near.append({"samples": [other_id, sample_id],
                                     "splits": [other_split, split], "distance": distance})
                index = len(hashes)
                hashes[index] = (split, sample_id, fingerprint)
                for band in bands:
                    buckets[band].append(index)
            except (OSError, ValueError, TypeError, AttributeError, cv2.error) as exc:
                errors.append(f"{loc}: {exc}")
    return {"counts": dict(counts), "missing_source_groups": dict(missing_groups),
            "manifest_sha256": files, "near_duplicate_threshold": threshold,
            "near_duplicates": near, "errors": errors, "passed": not errors and not near}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("manifests", type=Path, nargs="+")
    args = parser.parse_args()
    result = audit(args.manifests, args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "near_duplicates"}))
    raise SystemExit(0 if result["passed"] else 1)
