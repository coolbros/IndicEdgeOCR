# Native PyTorch checkpoint and numerical parity

The port instantiates 55,527,798 trainable scalars, with 2,622,784 non-trainable scalars in 123 buffers. Conversion maps 441 tensors bijectively, including 39 linear transposes, 122 BatchNorm running-state renames and a direct copy of the sinusoidal table. Weight and buffer dtypes are FP32. The Phase 1 comparison checks all converted tensors for bit-exact value agreement after layout conversion. Local and Kaggle conversion produced the same safetensors SHA-256, `0afc351e31cea4f6d670e68652bd22a11ab44273584e156159dbdb83fafb02ef`. The input checkpoint SHA-256 is `c4a110423c23739f157ff870e27f41649d8c89e4c8df5230c2cd2e9fd90a7e92`.

## Immutable Phase 1 inference

PyTorch 2.9.1 CPU FP32 compared with the existing Paddle 3.2.0 T4 archive `d07c2d3eb5f31656a462e150bbc47e802dcd718bf3fc835d0b8464fdb0dc5bb5`. The archive remains intact. Shapes and dtypes agreed for 220 tensors. With `rtol=1e-4, atol=1e-5` for activations and logits, and `atol=1e-6` for probabilities, 206 of 223 comparisons passed, including all three exact decodes. Seventeen layers had outliers beyond tolerance. The largest confidence difference was 4.29e-6. Official sample preprocessing was bit-exact.

| Output | Maximum absolute error | Outliers |
|---|---:|---:|
| Official sample final backbone | 1.76e-5 | 1 |
| Official sample CTC logits | 3.34e-5 | 17 |
| Synthetic 0 CTC logits | 2.91e-5 | 87 |
| Synthetic 1 CTC logits | 2.49e-5 | 17 |
| Synthetic 1 neck output | 3.15e-5 | 6 |

The original training archive used stochastic dropout. The new controlled capture uses real training words with dropout disabled and BatchNorm in training mode. It captures both RepDWConv branches, the pre-BN sum, CTC and NRTR intermediates and losses, and BatchNorm running state. The machine-readable report contains per-layer shape, dtype, maximum and mean error, relative L2, cosine similarity and element pass counts.

## Real-data CPU FP32

On Kaggle Linux, Paddle 3.2.0 and PyTorch 2.9.1+cpu used identical input and label arrays from four official HierText training images. Independent preprocessing and label encoding matched bit for bit. All 122 BN running buffers and all three loss comparisons passed. The four inference strings agreed exactly (maximum confidence difference 5.96e-8). Under the activation and logit thresholds above, 555 of 980 total comparisons passed in the initial controlled run. [Kaggle version 2](https://www.kaggle.com/code/agastyavar/indic-edge-ocr-phase23/versions/2).

| CPU FP32 output | Maximum absolute error | Mean absolute error | Relative L2 | Passed elements |
|---|---:|---:|---:|---:|
| Train NRTR memory projection | 1.841e-4 | 2.292e-6 | 5.929e-6 | 81,792 / 81,920 |
| Train CTC logits | 3.877e-4 | 1.363e-5 | 8.277e-6 | 2,977,478 / 2,993,600 |
| Train NRTR logits | 3.999e-3 | 1.986e-5 | 1.615e-4 | 477,953 / 523,992 |
| CTC loss | 4.926e-4 | 4.926e-4 | 8.823e-5 | 1 / 1 |
| NRTR loss | 5.054e-5 | 5.054e-5 | 5.507e-6 | 1 / 1 |
| Total loss | 4.425e-4 | 4.425e-4 | 2.998e-5 | 1 / 1 |

The embedding and sinusoidal outputs are exact. The first decoder block's self-attention and first normalization match, while the memory projection already has outliers. Error grows across cross-attention and subsequent normalization. The first training stem also has small floating-point differences. These results establish checkpoint identity and parity for the listed inputs, losses and decodes. Strict activation and logit parity remains open.

The v2 real-data archive SHA-256 is `22455e61a913c09baf50465303b8c60f6f916d68d5d13704fc72af314a729285`. The original Phase 1 fixture uses a different output directory and checksum.

The follow-up replayed ten operators with exact Paddle inputs and the same declared tolerances. All ten passed and the end-to-end comparison had 565 passes out of 990 checks. The isolated NRTR LayerNorm probes passed with maximum differences ≤1.91e-6. A full graph discrepancy remains even when individual probes pass, so the next investigation should trace cumulative stem/backbone error into decoder cross-attention with paired intermediate replays. The v4 archive SHA-256 is `e861cbf75fe498d05e4edc0fa568a09cabe88f6224b2e38427ee557df65af8fc`. [Kaggle version 4](https://www.kaggle.com/code/agastyavar/indic-edge-ocr-phase23/versions/4).

| Identical-input CPU probe | Max absolute error | Result |
|---|---:|---|
| Stem 3×3 convolution | 1.58e-6 | Pass |
| Even-kernel SAME convolution | 9.54e-6 | Pass |
| SAME max pooling | 0 | Pass |
| Unfused RepDWConv | 5.36e-6 | Pass |
| Exact GELU | 9.37e-7 | Pass |
| LightSVTR local path | 4.77e-7 | Pass |
| LightSVTR global mixer | 8.94e-8 | Pass |
| NRTR block 0 cross-attention LayerNorm | 4.77e-7 | Pass |
| NRTR block 0 final LayerNorm | 7.15e-7 | Pass |
| NRTR block 3 final LayerNorm | 1.91e-6 | Pass |

## Training persistence and resume

Accelerate 1.15.0 ran FP16 on one Tesla T4 with PyTorch 2.9.1+cu126 and CUDA 12.6. The original scaler setting of 128 caused an overflow. The retry started at scale 1 with dynamic backoff. The successful run used four distinct, audited HierText training crops and completed four optimizer updates across two epochs, resuming in a new process at epoch 1 and global step 2. The final optimizer had 318 state entries and the scheduler reached step 4. All updates reported finite nonzero gradient norms for the stem, CTC projection and NRTR projection. The first and fourth training losses were 8.821903 and 1.590969.

The final GPU `latest.pt` SHA-256 is `814061c32ab1609bb301fad37a7633c501ab998e5a216f6abbe5bc6904438ac4`. The validation-selected `best.pt` SHA-256 is `eaba545022d12a66efeb10e3f339ce08d32a50dce7027674f1071fd9a95ed6e7`. The smoke val set has four samples. The separate held-out evaluator was exercised on the frozen CPU smoke checkpoint with four test samples. Those results assess the pipeline wiring only. [Private Kaggle outputs](https://www.kaggle.com/code/agastyavar/indic-edge-ocr-phase23/versions/4).

## Operator semantics

- The native inference path keeps the checkpoint's unfused RepDWConv. Explicit reparameterization folds both BN layers, the identity, and the 1×1 branch. SAME padding adds one bottom and right cell for the even stem operators, with zero for convolution and negative infinity for max pooling.
- Paddle CPU's standard kernel updates running variance with population variance. Paddle dispatches 4D CUDA BatchNorm to cuDNN, whose running-variance update uses the sample statistic. Both use 0.9 on the old running estimate, biased variance for the training normalization, and epsilon 1e-5. The port uses explicit CPU running-stat updates and PyTorch functional CUDA BatchNorm.
- LightSVTR normalizes before its residual sublayers and multiplies Q by the scale before QKᵀ. NRTR multiplies after QKᵀ, applies its upper-triangular additive causal mask, and normalizes after residuals. Paddle embedding zeros padded outputs even if the loaded padding row has nonzero values.
- Paddle Warp-CTC takes raw logits and applies normalization inside the kernel. PyTorch uses explicit FP32 log-softmax and an arithmetic mean over unreduced sequence losses. NRTR smoothing applies 0.1 only to the other classes and ignores padding in the token mean. Exact GELU, HardSigmoid and Swish retain the upstream definitions.
- The optimizer groups use coupled Adam L2 with factors 0 for stem BN affine parameters, 1e-5 for the CTC projection and 3e-5 elsewhere. Paddle's and PyTorch's Adam epsilon placement can differ in bias-corrected updates. This phase does not claim optimizer-step equivalence. The smoke test intentionally uses a short schedule and a lower learning rate than the upstream 100-epoch recipe.

## Reproduction

```bash
export PYTHONPATH=src
python tools/convert_paddle_checkpoint.py \
  weights/original/PP-OCRv6_medium_rec_pretrained.pdparams \
  weights/converted/PP-OCRv6_medium_rec.safetensors
python tools/compare_torch_reference.py \
  --original weights/original/PP-OCRv6_medium_rec_pretrained.pdparams \
  --checkpoint weights/converted/PP-OCRv6_medium_rec.safetensors \
  --output artifacts/torch-phase1-cpu.json
python -m unittest discover -s tests
```

Prepare the dataset with the commands in [dataset provenance](dataset_provenance.md), then run [the Kaggle comparison](../README.md#phase-23).
