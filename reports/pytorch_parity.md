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
