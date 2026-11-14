# Phase 3.8 bounded training validation

The private Kaggle job uses an official HierText train/validation partition subset for engineering validation. The source is Open Images via the HierText publisher. Annotations are CC BY-SA 4.0 and underlying photographs retain source attribution and applicable terms. HierText is independent of the PP-OCRv6 pretraining recipe. No test examples enter training or validation. [HierText release](https://github.com/google-research-datasets/hiertext).

Preparation reads the first 640 MiB of the official training archive and the first 64 MiB of validation. It retains crops with legible horizontal labels of at most 23 code points that the frozen dictionary encodes in full and at least 10 pixels on each side. The prepared audit retained 5,198 training and 256 validation crops. Source-image IDs, source and image SHA-256, and dHash checks use the existing auditor. The validation list is fixed first and the flagged training crops are excluded. Audit metrics include counts by source, character and length.

| Manifest | SHA-256 |
|---|---|
| Train | `edf26d9969719c2dd4026cd1245002c59699d56e16031363aef830e3b335c5d0` |
| Validation | `bd15ac59ec8d982cde7184e36d90f7f8e227ceec6f760c52e12fdb5be3d5e4c7` |

## Commands

```bash
python tools/prepare_kaggle_phase38.py --owner agastyavar \
  --directory artifacts/phase38/training-job --audit artifacts/hiertext/subset/audit.json
kaggle kernels push -p artifacts/phase38/training-job --accelerator NvidiaTeslaT4 --timeout 7200
kaggle kernels output agastyavar/indic-edge-ocr-phase38-training/2 -p artifacts/phase38/training-v2 \
  --file-pattern '(audit|results|latest|best)[.]json$|[.]jsonl$' --page-size 50
python -m unittest discover -s tests
```

Stage the independent frozen inference rerun with `--inference`, the primitive replays with `--backward cpu` or `--backward gpu`, and the optimizer and loss reference with `--reference cpu` or `--reference gpu`. Use `--accelerator NvidiaTeslaT4` for GPU jobs. The runner uses just one GPU of the allocated T4 pair.

The run starts from the frozen converted checkpoint and uses Accelerate in FP16 while the loss computations upcast logits to FP32. The run uses 51 optimizer steps per epoch for two epochs, with five warmup steps and an engineering LR of 5e-5. Excluding the initial zero-LR warmup step leaves 101 parameter-update opportunities. The runner saves validation-selected best and latest, exits at the first boundary, then resumes in a fresh process and verifies the global cursor and Adam state. Per-update metrics record both recognition losses, LR, gradient norms, AMP overflow retries and multi-scale canvas.
