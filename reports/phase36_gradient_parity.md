# Phase 3.6 gradient parity

Each framework starts with the verified original state and consumes the immutable Phase 3.5 training batch. The live forward uses training BatchNorm and eval-mode dropout. Each framework loads the initial parameters and BN buffers again before further independent forward experiments. Training capture records updated running means and variances. CPU and CUDA use their corresponding running-variance behavior from the existing port.

The CTC and NRTR loss experiments differentiate each loss separately from the shared training forward, recording head inputs, logits and selected weights. The full experiment differentiates the sum into the image and selected stem, backbone, projection and decoder parameters. A separate fixed-logit experiment feeds actual Paddle logits to both loss operators so inherited backbone error cannot affect that comparison. Converted Linear weight gradients use Torch layout in the metrics. No optimizer update is part of these runs.

The baseline uses the existing native PyTorch loss. Ablations change CTC alone, NRTR alone, then NRTR's reduction order. The original activation thresholds apply to gradient rows. Reports include the element count and tolerance pass count, maximum and mean absolute error, relative L2, cosine and shape. Reproduction uses the paired commands in [Numerical equivalence](phase36_numerical_equivalence.md).

## Paired measurements

“Fixed logits” uses the same archived Paddle logits in both frameworks. Numeric rows below show maximum absolute error. Relative L2, mean absolute error, cosine similarity, shape and per-element counts are in the result JSON.

| Experiment | CPU | T4 FP32 |
|---|---:|---:|
| Fixed-logit CTC loss | 6.404e-4, fail | 3.338e-6, pass |
| Fixed-logit CTC logit gradient | 8.713e-5, fail | 6.812e-7, pass |
| Fixed-logit smoothed NRTR loss | 6.676e-5, pass | 9.537e-7, pass |
| Fixed-logit NRTR logit gradient | 1.900e-5, fail | 8.568e-8, pass |
| Full training-forward total loss | 4.425e-4, pass | 8.583e-6, pass |
| Natural CTC head gradient rows | 0 / 4 | 3 / 4 |
| Natural NRTR head gradient rows | 0 / 6 | 2 / 6 |
| Full-graph gradient rows | 0 / 10 | 2 / 10 |

The CPU CTC loss and logit gradient on fixed inputs miss independently of the backbone. The CUDA fixed-logit losses and gradients all pass. On CPU, changing CTC to the decomposed sum increases its fixed-logit loss error to 6.781e-4 and the gradient error to 8.883e-5. The decomposed NRTR loss moves slightly closer (6.580e-5), while its logit gradient worsens to 2.025e-5. Reductions have negligible effect on the severe CPU decoder and full-graph gradient discrepancies.

| Natural gradient | CPU max / relative L2 / cosine | T4 max / relative L2 / cosine |
|---|---:|---:|
| CTC feature sequence | 4.094e-5 / 4.077e-4 / near 1 | 1.565e-6 / 1.386e-5 / near 1 |
| NRTR projected memory | 5.343e3 / 0.999989 / 0.82677 | 2.586e-5 / 3.505e-4 / >0.99999998 |
| NRTR first decoder cross-attention KV weight | 1.420e5 / 0.999989 / see JSON | 6.257e-4 / 4.228e-4 / see JSON |
| Full-graph image input | 8.246e6 / 0.999985 / 0.94882 | 3.964e-2 / 3.078e-4 / >0.99999999 |

Strict full-graph gradient parity remains open. CPU image gradients have 184,320 outliers, with mean absolute error 6.793e4. CUDA image gradients have 158,644 outliers, with mean 3.269e-4. CPU NRTR logit gradients are much closer than the projected memory gradient, and the isolated loss experiments rule out the log-softmax ablation as a remedy for most of this divergence. CUDA shows high gradient cosine similarity on the sampled full graph, while the strict elementwise criterion still fails.

## CPU LayerNorm backward divergence

The first measured decoder divergence in reverse order is the input of the last LayerNorm. Its outgoing gradient is identical. With identical memory, tokens and upstream gradient, the adjacent MLP output gradient differs by 8.483e-2. The mismatch grows through earlier decoder blocks and reaches 135.83 at projected memory. This rules out inherited backbone differences as an explanation for this large CPU result.

The pinned Paddle CPU LayerNorm backward applies the learned scale to the input-gradient calculation only when it also produces the scale gradient. A standalone probe with the real decoder's final LayerNorm verifies the effect directly. Asking just for the input gradient gives a maximum difference of 6.520 and relative L2 of 1.157. Asking for the input and scale gradients reduces this to 1.907e-6 and relative L2 of 9.170e-8, within the original threshold. [Paddle CPU LayerNorm gradient source](https://github.com/PaddlePaddle/Paddle/blob/v3.2.0/paddle/phi/kernels/cpu/layer_norm_grad_kernel.cc).

| Matched-decoder backward | Original thresholds | Projected memory max error |
|---|---:|---:|
| Request activations only | 1 / 25 | 135.83 |
| Also request LayerNorm parameter gradients | 25 / 25 | 2.532e-8 |

The severe CPU results above are valid observations of the selected-gradient API, which leaves LayerNorm scales outside the requested gradients. The result cannot be attributed to ordinary training backward. The revised harness requests all relevant LayerNorm scale gradients as well. The dedicated decoder probe can be reproduced with `--device cpu --decoder-only` on the staging command. [Matched-memory decoder run](https://www.kaggle.com/code/agastyavar/indic-edge-ocr-phase36-cpu-decoder/versions/3).

The matched-decoder metrics JSON SHA-256 is `1640d61e3091a8c23fab544b6745e63d504dfd1d6878fafa5acd517366714140`. The corresponding Paddle capture SHA-256 is `3f040fe3a77c6c455619cdc7bea67ca04ef176e98b02dd099408d59140bff5e3`.
