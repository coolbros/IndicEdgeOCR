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

## T4 FP32 results

The [paired T4 run](https://www.kaggle.com/code/agastyavar/indic-edge-ocr-phase37-gpu/versions/3) captured 3,787 records and 270 identical named gradient request sets in each framework. It used one visible Tesla T4, driver 580.178.04, compute capability 7.5, CUDA 12.6, Paddle 3.2.0 with cuDNN 9.5.1 and Torch 2.9.1+cu126 with cuDNN 9.10.2. The harness checked FP32 CUDA tensors and disabled TF32. Re-running the Phase 3.6 comparator gave image-gradient maximum error 0.0396423 and relative L2 3.07837e-4, the original GPU values.

| T4 replay with fixed inputs and upstream coefficients | Gradient rows passing | Maximum error in input gradient |
|---|---:|---:|
| CTC projection | 3 / 3 | 0 |
| NRTR pooled-map projection | 2 / 2 | 0 |
| NRTR output projection | 2 / 2 | 0 |
| CTC encoder | 36 / 36 | 1.788e-7 |
| Whole NRTR decoder | 81 / 83 | 3.681e-6 |
| Individually replayed NRTR decoder operators and blocks | 368 / 368 | See tensor metrics |
| Backbone with CTC gradient injected at pooled feature | 185 / 199 | 7.915e-5 |
| Backbone with NRTR gradient injected at pooled feature | 152 / 199 | 4.742e-4 |
| Backbone with summed gradient injected at pooled feature | 154 / 199 | 4.644e-4 |

On the T4 natural CTC path, a fixed logit gradient makes the projection's feature gradient byte-identical even with differing natural features. The first observed strict **stage-input** gradient failure is at the input to stage 2, after its only block in reverse order. Earlier in backward order, stage 5 already has local parameter gradient misses. In the CTC branch these include a BN scale in block 5.0. All selected block 5.2 replays pass. The measurements localize strict stage-input misses to composed block 2.0, while earlier parameter misses concern different outputs of the backward pass.

On the T4 natural NRTR path, the projected-memory gradient is already outside tolerance (max 2.760e-5, 492 outliers), although the head receives the same logit derivative in both frameworks. Fixing projected memory reduces its maximum error to 3.681e-6 with zero outliers. The two remaining whole-decoder gradient misses each have one element, in the embedding weight and block 2 cross-attention KV weight. Isolated replay of the corresponding decoder KV projection passes. The full matched-memory run still has 24 forward-logit outliers. A local operator probe may pass while small differences propagate across operators. The current trace does not isolate the embedding's gather/scatter backward.

Direct pooled-gradient injection cuts the T4 natural NRTR image-gradient maximum from 4.244e-2 to 4.742e-4, with relative L2 from 4.531e-4 to 4.155e-6. It leaves 151 image elements outside tolerance. The CTC backbone retains errors after the equivalent injection. This localizes most of the NRTR discrepancy upstream of the pooled backbone gradient, with a residual attributable to the backbone replay.

| T4 capture | SHA-256 |
|---|---|
| Paddle tensor NPZ | `157fd928cc98b4b1efd9eda67f020793df62dc57225396df082978341e1624c5` |
| Torch tensor NPZ | `84c96b741ebe5aaff16bbd44a6fc19ebaa9f52519207991ff43b4fabe557a32d` |
| Paddle capture JSON | `4a8bef0ccab2ce5bb55ead2e6706d11834d6a88462d0935fd2c9c0db41675943` |
| Torch comparison JSON | `268dc7b3f536445feeaa4302f994184af633195f37f167dbab9b7eea9528ce38` |
