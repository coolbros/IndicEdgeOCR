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

## CPU and CUDA comparison

Paddle 3.2.0 and PyTorch 2.9.1 ran on the same Kaggle worker. The GPU run used one visible Tesla T4 (compute capability 7.5), driver 580.178.04 and CUDA 12.6, with cuDNN 9.5.1 for Paddle and 9.10.2 for PyTorch. The runner disables TF32, requests the highest Torch FP32 matmul precision, disables dropout, and checks that measured activations, model weights and gradients actually reside on the requested device in FP32. Both runtimes used Python 3.12.10 and NumPy 1.26.4. [Paired CPU and T4 run](https://www.kaggle.com/code/agastyavar/indic-edge-ocr-phase36-both/versions/1).

| Paired comparison | CPU | T4 FP32 |
|---|---:|---:|
| Natural inference boundary rows passing | 872 / 1,280 | 1,259 / 1,280 |
| Natural training forward and BN rows passing | 255 / 536 | 472 / 536 |
| Running BN state rows passing | 122 / 122 | 122 / 122 |
| Pooled training backbone max error (outliers) | 5.703e-5 (147) | 1.264e-5 (0) |
| NRTR memory projection max error (outliers) | 1.841e-4 (128) | 2.861e-5 (1) |
| CTC training logits max error (outliers) | 3.877e-4 (16,122) | 7.629e-5 (1,094) |
| NRTR training logits max error (outliers) | 3.999e-3 (46,039) | 2.863e-4 (2,006) |
| Final CTC probabilities passing | 4 / 4 | 4 / 4 |
| CTC argmax sequences agreeing | 4 / 4 | 4 / 4 |

These hooked natural-boundary counts use a different denominator from the Phase 3.5 exact-input replay. The largest probability error over the four images is 2.384e-7 on each device. The earliest CPU training tolerance miss is the second stem convolution's BN, as in Phase 3.5. On the T4, the first natural backbone training miss is the final RepDWConv BN in `blocks4.3`. In eval on the first crop, the earliest T4 miss is `blocks5.2`. The training logits still fail strict parity on either backend.

| Paired-run artifact | SHA-256 |
|---|---|
| CPU Paddle capture NPZ | `a837267ecca47f6c13126798dcc24b81c5dea93117b9d0b744ce93049156cf22` |
| T4 Paddle capture NPZ | `ee7b2cf333bb118fb5105553770afe771111cb8fd7140cf228b2b68d445ac4b5` |
| CPU Torch metrics JSON | `7179940354b5958e23dfdaf46e50ecf5ecd95c8f1bbef71f1248125ff2fa2034` |
| T4 Torch metrics JSON | `13c8d750fbf1aff230d8614090807b97073248c8414bd0f5eb3706dc87005fe2` |

## Backbone intervention

The companion CPU job replays composed modules from their exact Paddle inputs. Each diagnostic resets checkpoint parameters and running state before evaluation. FP64 uses the same module in double and rounds the output to FP32 for comparison. No-MKLDNN changes only the Torch CPU backend. Numbers are maximum absolute errors followed by element outliers in parentheses. The single block in stage 2 aliases the stage result, and the whole backbone contains the preceding stage and stem misses. [CPU replay and interventions](https://www.kaggle.com/code/agastyavar/indic-edge-ocr-phase36-cpu/versions/1).

| Exact-input boundary | Phase 3.5 | CPU FP32 | CPU FP64 | CPU FP32 without MKLDNN |
|---|---:|---:|---:|---:|
| Eval block 2.0 and stage 2 | 2.134e-5 (2) | 2.235e-5 (2) | 2.104e-5 (2) | 2.235e-5 (2) |
| Eval entire backbone | 2.587e-5 (1) | 4.172e-5 (11) | 2.849e-5 (2) | 3.052e-5 (1) |
| Train stem2b | 1.907e-5 (5) | 1.907e-5 (1) | 1.907e-5 (6) | 1.907e-5 (0) |
| Train composed stem | 2.474e-5 (4) | 2.642e-5 (3) | 2.187e-5 (3) | 2.605e-5 (5) |
| Train entire backbone | 5.703e-5 (147) | 7.322e-5 (209) | 5.996e-5 (135) | 6.640e-5 (184) |

The component-level trace passes exact-input stem2b convolution, BN and ReLU. The residual addition in eval block 2.0 is exact and the isolated token and channel mixers pass. The composed eval block still fails after FP64, consistent with a sensitivity to the arithmetic of the Paddle reference and its intermediate rounding. In training, turning off MKLDNN closes the local stem2b miss on this CPU worker, but does not close the stem or whole graph. No backend switch was made in the production model.

Injecting the exact Paddle output into the Torch training forward localizes propagation. Differences are nonmonotonic across interventions. Injecting after stage 3 reduces NRTR max error sharply, while the stage 4 intervention is larger. This demonstrates that scalar error summaries do not predict how a particular perturbation passes through the decoder.

| Paddle output inserted after | Pooled backbone max | NRTR memory max | CTC logits max | NRTR logits max |
|---|---:|---:|---:|---:|
| stem2a | 5.281e-5 | 1.915e-4 | 3.090e-4 | 3.109e-3 |
| stem2b | 7.035e-5 | 2.232e-4 | 3.433e-4 | 2.579e-3 |
| Complete stem | 4.458e-5 | 1.813e-4 | 2.108e-4 | 1.571e-3 |
| Stage 2 | 1.913e-5 | 4.911e-5 | 1.431e-4 | 1.862e-3 |
| Stage 3 | 1.931e-5 | 6.199e-5 | 6.676e-5 | 3.672e-5 |
| Stage 4 | 6.318e-6 | 2.670e-5 | 5.722e-5 | 6.962e-4 |
| Stage 5 or pooled backbone | 0 | 0 | 1.717e-5 | 1.645e-5 |

At the stage 5 and pooled-backbone interventions, both heads' final logits pass the original thresholds. This removes all 2.9.1 CPU logit outliers for this diagnostic with identical late backbone features. The natural graph still misses. The CPU intervention metrics JSON SHA-256 is `99ed728ed46f716e86d23f9654ca5a6d5e4a5b1217ab94699d9b362c35b99b05` and its Paddle capture SHA-256 is `c91840be8e18e046d17b91497caeb594119490cf2e81ded75ba20825be69d34c`.

To reproduce the separate CPU intervention or decoder-only probe, stage with `--device cpu` or `--device cpu --decoder-only` and submit without an accelerator argument. To reproduce the T4 validation, stage with `--evaluation --device gpu` and submit with `--accelerator NvidiaTeslaT4`. Small downloadable outputs can be selected with `--file-pattern 'results[.]json$'` for tensor metrics and `--file-pattern '(recognition|audit|predictions-paddle|predictions-torch)[.]json$|validation[.]jsonl$' --page-size 200` for validation. Large captures, annotations and crop data stay in private Kaggle outputs and the ignored local artifact directory.

## Recognition equivalence on HierText validation

The same 1,080 crops passed through both frozen CTC recognizers on one T4. The evaluator verified all 100 preprocessed batch hashes and sample IDs before comparing outputs. There were **zero decoded-sequence disagreements**. The mean absolute confidence difference was 2.705e-7 and the maximum was 1.788e-5. CER is micro-averaged code-point edit distance and WER is micro-averaged whitespace-token edit distance. Raw Unicode and NFC CER agree on this subset. [T4 validation run](https://www.kaggle.com/code/agastyavar/indic-edge-ocr-phase36-eval-gpu/versions/1).

| Group | Crops | Decoded agreement | CER, both | WER, both |
|---|---:|---:|---:|---:|
| All | 1,080 | 100% | 12.27% | 21.39% |
| Printed | 997 | 100% | 12.13% | 19.66% |
| Handwritten | 83 | 100% | 14.62% | 42.17% |
| ASCII labels | 1,044 | 100% | 12.08% | 20.02% |
| Non-ASCII labels | 36 | 100% | 18.93% | 61.11% |
| 1–5 code points | 626 | 100% | 10.34% | 19.65% |
| 6+ code points | 454 | 100% | 13.29% | 23.79% |

| T4 model timing | Paddle | PyTorch |
|---|---:|---:|
| Crops per second | 225.95 | 221.36 |
| Median batch latency | 58.16 ms | 60.69 ms |
| Offline preprocessing for all 1,080 crops | 494 ms | 479 ms |

Model timings cover synchronized forward execution after warmup, with up to 16 crops per width-bucketed batch. They exclude decode, file I/O, preprocessing and host-to-device transfer. This one-worker run gives observed latency, without establishing a precise performance ratio. CER/WER describe the pretrained recognizer on the selected validation crops. Exact agreement concerns implementation equivalence on those crops.

| Validation artifact | SHA-256 |
|---|---|
| Per-sample metrics and aggregate report JSON | `f07a1f4d2ce231e61700cc998d9a0f860809463fa29cbbbb9367ba31ae4b80b1` |
| Paddle predictions JSON | `6759a16ab0bf545bc2f9ce43de2fbdb76589b8f58bd63fcda5c4ee815e38affa` |
| PyTorch predictions JSON | `230b6428d8ae80883edc842d2567c32d14a0220fbf74d92c3a18dca0e85c8f3d` |
| Validation leakage audit JSON | `fc3d64372a67bb18e4ba03ad98f6fd7dbfa5e7a281bcfce35d6cbf9f7e5a7ab8` |
