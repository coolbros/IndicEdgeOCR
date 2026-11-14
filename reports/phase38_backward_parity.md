# Phase 3.8 backward parity

The targeted replays build on [Phase 3.7](phase37_backward_attribution.md) and consume a fresh private attachment of the immutable Paddle tensor capture for each device. Default architecture and backward kernels remain the same. The new harness checks a 3×3 DW/BN path, 1×1 DW, identity, their sum and post-sum BN, SE arithmetic, residual addition and channel-mixer convolutions and BNs. The real upstream implementation has no separate BN inside the 1×1 DW branch or on the identity. Both feed the outer BN after summation. The SE primitive tests use identical generated cotangents while the RepDWConv and selected mixer probes use the captured branch coefficients.

The NRTR replay compares the block 2 KV projection and a repeated-token embedding example with and without reversed token order. The repeated-token example is a diagnostic fixture. The prior whole-graph embedding outlier requires separate attribution from the synthetic scatter probe. LayerNorm is unchanged, and the full gradient request regression is retained. A new regression exercises the real CTC and NRTR consumers at a shared pooled feature with reduced vocabulary, comparing joint backward, explicit sum and reversed branch order.

## CPU matched primitive results

The [CPU replay](https://www.kaggle.com/code/agastyavar/indic-edge-ocr-phase38-backward-cpu/versions/2) compared 223 tensor rows, including 167 gradients. 138 gradient rows meet the original thresholds. The sums with all three released branch values and identical cotangents have exact input gradients. The SE primitives and standalone SE pass all 23 comparisons per branch with the diagnostic seed. Composed token-mixer parameter misses remain with propagated coefficients. The CPU repeated-index embedding and its order-control outputs are exact across frameworks. The isolated decoder KV projection passes all four rows with maximum 3.815e-6.

| CPU matched replay | Branch seed | Maximum absolute error | Relative L2 | Outliers |
|---|---|---:|---:|---:|
| RepDWConv combined input | CTC | 2.930e-3 | 1.525e-7 | 0 |
| RepDWConv combined input | NRTR | 1.563e-2 | 1.735e-7 | 2 |
| Token-mixer input | CTC | 2.148e-2 | 6.649e-7 | 3 |
| Token-mixer input | NRTR | 3.125e-2 | 4.863e-7 | 2 |
| Block 5.0 isolated compress BN bias | CTC | 3.815e-5 | 6.469e-6 | 1 |
| Block 5.0 isolated compress conv weight | NRTR | 2.480e-5 | 3.744e-6 | 1 |

Small isolated convolution weight and BN bias reduction differences occur even with matched inputs. Inner BN bias relative error can be large because its reference norm is near zero. The input gradient of the composed operations also reflects rounding in internal branch calculations. This probe establishes exact addition backward for separately supplied operands and sparse local reduction errors. It does not prove a unique primitive cause for every composed outlier. No semantic defect requiring a production change is evidenced by the replays. The Phase 3.7 exact-input reductions and loss-isolation evidence remains applicable.

CPU metrics SHA-256 `892877a05bc6b1b0246ef0b1dfd15dfe4a7350bae9354f3dd005e1158cdc0ad6` includes tensor shapes, input and parameter rows, error percentiles and original thresholds.
