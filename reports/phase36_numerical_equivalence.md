# Phase 3.6 numerical equivalence

## Fixed inputs and acceptance criteria

This phase started on clean `main` at `677f174`, after the Phase 3.5 paired replay. The pinned PaddleOCR checkout is `dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf`. All 441 converted tensors compare exactly after the explicit layout mapping. No checkpoint, architecture, dictionary, fixture, or Phase 3.5 tolerance changed.

| Immutable input | SHA-256 |
|---|---|
| Official Paddle checkpoint | `c4a110423c23739f157ff870e27f41649d8c89e4c8df5230c2cd2e9fd90a7e92` |
| Converted checkpoint | `0afc351e31cea4f6d670e68652bd22a11ab44273584e156159dbdb83fafb02ef` |
| Phase 3.5 v4 Paddle fixture | `e861cbf75fe498d05e4edc0fa568a09cabe88f6224b2e38427ee557df65af8fc` |
| Phase 3.5 reproduced baseline JSON | `f635a681b652209c854b22b23a8e00d246cb31a511d176c331d70b562d28f28b` |

Tensor rows require the expected shape and dtype and every element within `atol + rtol * abs(reference)`. The original activation, logits and loss thresholds are `atol=1e-5, rtol=1e-4`. Probabilities and standalone log-softmax use `atol=1e-6, rtol=1e-4`. The experiment records maximum and mean absolute errors, error percentiles, relative L2, cosine similarity and element counts. See the [Phase 3.5 trace](parity_trace.md) for the original replay.

## Held-out validation

The evaluation uses 1,080 legible, horizontal word crops from 93 source images in the official HierText validation partition. Selection follows tar order, takes up to 12 eligible crops from each source image, and requires both crop dimensions to be at least 10 pixels. The selection excluded 359 empty, vertical or illegible annotations and 526 small crops before reaching the target count. It uses the pretrained dictionary unchanged and does not filter by dictionary membership. The official annotations are CC BY-SA 4.0. See [HierText release and partitions](https://github.com/google-research-datasets/hiertext) and [project dataset provenance](dataset_provenance.md).

| Input or manifest | SHA-256 |
|---|---|
| Official validation annotations | `ce7085d8bf24c2f13df852715c34494e6748cafcabb4bfc7a6e2bb320f58d23a` |
| First 64 MiB of validation archive | `ef800c539c5735d48a6bc809607fcd50b880bda096d572d902ef5f58acbb9dd1` |
| Validation manifest | `a6c78abb1c0e6e5f2e3c3c814fd9daf1bc582d35bc3819c091a91e61a7df5ed7` |
| Prior train manifest | `4fc83fe5c90366051e75d2f10ef61624a68dbec3785e263a1bcd14042cdfed73` |
| Prior test manifest | `19cff4ad576d543d862bf49fa1138999ecb7a6023135dfe92a8abf710365dfb7` |

The audit found zero cross-split sample IDs, source-image IDs, exact crop hashes or perceptual dHash matches at Hamming distance ≤4 across the 1,080 validation, four prior training and four prior test records. The four test records are used only for the split audit. Source-image metadata cannot establish higher-level document identities or which images the official checkpoint saw during pretraining. Selection favors early archive entries and horizontal text, and the evaluation characterizes this subset.

## Reproduction

From the repository root, set `PYTHONPATH=src` and use the prepared Phase 2/3 HierText audit. The two Kaggle jobs write private versioned outputs. Pull them into fresh directories under `artifacts/phase36/`.

```bash
python tools/prepare_kaggle_phase36.py --owner agastyavar \
  --directory artifacts/phase36/staged-both --audit artifacts/hiertext/subset/audit.json \
  --reference agastyavar/indic-edge-ocr-phase23/4 --device both
kaggle kernels push -p artifacts/phase36/staged-both --accelerator NvidiaTeslaT4 --timeout 7200

python tools/prepare_kaggle_phase36.py --owner agastyavar \
  --directory artifacts/phase36/staged-eval --audit artifacts/hiertext/subset/audit.json \
  --reference agastyavar/indic-edge-ocr-phase23/4 --evaluation
kaggle kernels push -p artifacts/phase36/staged-eval --timeout 5400

kaggle kernels output agastyavar/indic-edge-ocr-phase36-both -p artifacts/phase36/download-both
kaggle kernels output agastyavar/indic-edge-ocr-phase36-eval -p artifacts/phase36/download-eval
python -m unittest discover -s tests
```

For local runs, invoke `tools/phase36_experiment.py paddle cpu` followed by `tools/phase36_experiment.py torch cpu` with the same `--archive`, `--original`, `--checkpoint`, `--dataset` and `--source`. Pass the first invocation's `reference.npz` as `--reference` and give each run its own `--output` directory. Use `gpu` for both processes with their CUDA wheels and the same visible GPU. The runner requires a clean pinned PaddleOCR checkout and verifies the paired capture digest before Torch comparison.
