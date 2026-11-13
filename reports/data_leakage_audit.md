# Data leakage audit

The official HierText metadata had **11,639 unique source image IDs** across train, validation and test. Cropping preserved the official partitions. The smoke manifests each contain four source images and one recognition crop per source. The audit found zero repeated sample IDs, zero exact cross-split crop duplicates, zero source-group collisions, zero missing groups, zero invalid records, and zero cross-split near-duplicate pairs at 64-bit dHash Hamming distance ≤4.

| Split | Crops | Manifest SHA-256 |
|---|---:|---|
| Train | 4 | `4fc83fe5c90366051e75d2f10ef61624a68dbec3785e263a1bcd14042cdfed73` |
| Validation | 4 | `4abe8e6074bee711b3a5b1dc4c259fdaaba02efc29707b7ffd5b96e0d7cf6759` |
| Test | 4 | `19cff4ad576d543d862bf49fa1138999ecb7a6023135dfe92a8abf710365dfb7` |

Among visited candidates, the preparation excluded 30 train and 36 validation words for labels or orientation, and 24 train and 50 validation words for crop dimensions. One visited validation source had no eligible word. All other dataset images outside the selected sources were outside this smoke subset. The manifests use exclusive creation, are outside Git, and are checked against the audit before training. Test data are scanned for integrity and duplication. Checkpoint selection uses only validation accuracy.

```bash
python tools/verify_dataset_splits.py --root artifacts/hiertext/subset \
  --output artifacts/hiertext/audit-repeat.json \
  artifacts/hiertext/subset/train.jsonl artifacts/hiertext/subset/validation.jsonl \
  artifacts/hiertext/subset/test.jsonl
```

The visual duplicate audit covers the selected crops. dHash can miss significant transformations. Open Images source IDs do not identify separate photographs of the same physical document, and the original PP-OCRv6 pretraining membership is unknown. Automated tests cover cross-split ID, exact-byte, group and near-duplicate failures, and accept ordinary identical transcriptions.
