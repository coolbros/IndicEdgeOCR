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

## T4 matched primitive results

The [T4 replay](https://www.kaggle.com/code/agastyavar/indic-edge-ocr-phase38-backward-gpu/versions/1) compared the same 223 tensor rows, with 163/167 gradient rows within the original thresholds. The worker ran Paddle 3.2.0 and Torch 2.9.1+cu126, CUDA 12.6, cuDNN 9.5.1 and 9.10.2 respectively, and driver 580.178.04. FP32 computations used one visible Tesla T4 with TF32 disabled. The exact RepDWConv addition and identity gradients agree. The combined RepDWConv input gradients pass with maxima 1.953e-3 (CTC) and 3.906e-3 (NRTR) and relative L2 below 7.1e-8. Isolated SE input-gradient max is 4.191e-9. The NRTR channel-mixer expand weight has three outliers both in the mixer and in the isolated convolution. Max errors are 1.018e-4 and 8.740e-5, with relative L2 near 2.2e-6. Both late compress BN/conv probes pass on T4. That backend dependence is consistent with differences in reduction and accumulation.

The decoder KV input and weight gradients are exact in the matched test. Its bias differs by 4.098e-8 and passes. The previously observed whole-decoder KV weight miss therefore depends on the composed graph for this fixture.

The repeated-token CUDA probe reveals a distinct Paddle 3.2 behavior. Forward embeddings are exact. All 512 strict gradient outliers are on padding row 0, with max 115.238. Checking the captured Paddle tensor against the ordered FP32 scatter of the supplied cotangents, excluding padding, gives a nonpadding max of 7.629e-6 and zero nonpadding outliers. Reversal changes accumulation by at most 1.526e-5 within each runtime. The pinned Paddle CUDA scatter receives a padding index in its functor, but the dense kernel scatters every index. The configured PyTorch embedding excludes PAD from its gradient and masks its forward output. That behavior is kept because the padding row contributes zero to the forward function. This CUDA padding difference does not explain the prior whole-decoder nonpadding outlier at row 80. The controlled repeated-index probe provides no evidence of a material semantic mismatch for nonpadding rows. [Paddle CUDA implementation](https://github.com/PaddlePaddle/Paddle/blob/v3.2.0/paddle/phi/kernels/gpu/embedding_grad_kernel.cu).

T4 comparison JSON SHA-256 is `3e069f0167145ab173944dae2372ef782c76fa6800f919bb8e4191f8978b0db0`. The Paddle capture used for the padding-row check has SHA-256 `4ca4ffa049d0431b9b35689f6633b965095d98305a1739d41d9aed6d55c05547`. Commands are in [training validation](phase38_training_validation.md).
