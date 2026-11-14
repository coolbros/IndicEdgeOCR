# Phase 3.7 backward attribution

Baseline `main` was clean at `b850ee7da73d180cd4296abade221d3a373cfda9`. The PaddleOCR checkout is pinned to `dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf`. All 16 available Phase 3.6 files in its checksum index matched. The earlier capture NPZ hashes are recorded in their corresponding JSON files. The Phase 3.6 tensor bundles live in the original Kaggle outputs.

| Verified immutable input | SHA-256 |
|---|---|
| Official Paddle checkpoint | `c4a110423c23739f157ff870e27f41649d8c89e4c8df5230c2cd2e9fd90a7e92` |
| Converted checkpoint | `0afc351e31cea4f6d670e68652bd22a11ab44273584e156159dbdb83fafb02ef` |
| Phase 3.5 v4 fixture | `e861cbf75fe498d05e4edc0fa568a09cabe88f6224b2e38427ee557df65af8fc` |
| Training fixture manifest | `4fc83fe5c90366051e75d2f10ef61624a68dbec3785e263a1bcd14042cdfed73` |
| Phase 3.5 comparison JSON | `f635a681b652209c854b22b23a8e00d246cb31a511d176c331d70b562d28f28b` |
| Phase 3.5 exact replay JSON | `3477b2b74bbaa1a0b66b6141eDE8AEA7A77116FE2B7583777E3A594038479104` |

The experiments use an unchanged recognizer and loss with no optimizer steps. Both engines use training mode with dropout disabled, and start from the pinned checkpoint and BN buffers. Reported BN state comes from the first native forward. The native loss comparison uses each engine's own loss derivatives. The matched comparison injects the Paddle logit gradients into both graphs. Exact local replays independently supply Paddle argument bytes and the recorded upstream coefficient. The attention primitive trace also uses a deterministic diagnostic seed (NumPy seed 3701), propagated through its scoring primitives. Capture JSON lists every named gradient requested and retains shape, dtype, layout and digest for saved tensors. Full-network calls request all connected parameters, including both LayerNorm affine gradients. Paddle Linear weight gradients are transposed to Torch layout before comparison.

## Reproduction

Set `PYTHONPATH=src` for direct local calls. Kaggle stages an isolated Python 3.12.10 environment, validates the archive and regenerated official checkpoint conversion, and writes artifacts to a new notebook version. The Phase 3.6 reproduction flag reruns the existing comparator in a new output directory.

```bash
python tools/prepare_kaggle_phase37.py --owner agastyavar \
  --directory artifacts/phase37/job-cpu --audit artifacts/hiertext/subset/audit.json \
  --reference agastyavar/indic-edge-ocr-phase23/4 --device cpu --reproduce-phase36
kaggle kernels push -p artifacts/phase37/job-cpu --timeout 7200
python tools/prepare_kaggle_phase37.py --owner agastyavar \
  --directory artifacts/phase37/job-gpu --audit artifacts/hiertext/subset/audit.json \
  --reference agastyavar/indic-edge-ocr-phase23/4 --device gpu --reproduce-phase36
kaggle kernels push -p artifacts/phase37/job-gpu --accelerator NvidiaTeslaT4 --timeout 7200
kaggle kernels output agastyavar/indic-edge-ocr-phase37-cpu -p artifacts/phase37/cpu \
  --file-pattern '(capture|results)[.]json$' --page-size 50
kaggle kernels output agastyavar/indic-edge-ocr-phase37-gpu -p artifacts/phase37/gpu \
  --file-pattern '(capture|results)[.]json$' --page-size 50
python -m unittest discover -s tests
```

For a local replay, pass identical inputs to `python tools/trace_backward.py paddle cpu` and `python tools/trace_backward.py torch cpu`. The shared arguments are `--source references/PaddleOCR --archive artifacts/phase23-kaggle-v4/paddle-real-cpu/real_parity.npz --original weights/original/PP-OCRv6_medium_rec_pretrained.pdparams --checkpoint weights/converted/PP-OCRv6_medium_rec.safetensors --dataset artifacts/hiertext/subset`. Give each invocation a new `--output` directory. The Torch invocation requires `--reference` pointing to the Paddle directory with `capture.json` and `tensors.npz`. For GPU runs use the isolated CUDA wheels and expose a single physical device to both processes.

## Recognition scope

The prior frozen recognizers agreed on all 1,080 T4 HierText validation crops. The subset includes 997 printed and 83 handwritten crops, 1,044 ASCII and 36 non-ASCII labels, from 93 source images. It favors horizontal text and early archive entries. The forward path and dictionary are unchanged, so this phase does not repeat the benchmark. See the separate [Indic evaluation corpus plan](indic_evaluation_corpus_plan.md).
