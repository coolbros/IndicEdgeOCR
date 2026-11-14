# Phase 3.7 LayerNorm backward

Paddle 3.2 CPU `LayerNormGradKernel` conditions part of the input-gradient calculation on whether the scale gradient is requested. With that output active, it forms `dy * gamma` before subtracting rowwise terms. Without that output it uses unscaled `dy`. The Phase 3.6 selected-gradient result therefore depends on the request set. The complete diagnostic requests input, scale and bias on identical tensors and upstream coefficients. Every actual decoder LayerNorm is replayed with its affine parameters, normalization axis and epsilon. The partial request is repeated on the final decoder LayerNorm as an explicit control.

The Torch diagnostics compare native normalization with double precision, a transposed-contiguous-transposed input, and a centered-variance decomposition. These are experimental probes. The production model keeps native LayerNorm.

[Source for the pinned Paddle CPU kernel](https://github.com/PaddlePaddle/Paddle/blob/v3.2.0/paddle/phi/kernels/cpu/layer_norm_grad_kernel.cc). [Exact replay commands and artifact format](phase37_backward_attribution.md).

## T4 replay

All 12 decoder LayerNorms pass for forward, input, scale and bias (48 rows). Maximum native errors are 3.815e-6 in forward, 4.768e-7 for input and scale, and 9.537e-7 for bias. The 12 sets of gradients also pass with FP64, strided FP32 and the centered-variance decomposition. Some bias maximums improve in FP64, with no change to strict pass counts. The actual final-layer partial-request diagnostic also passes on T4. The standalone input-gradient max is 7.451e-9. Neither a custom backward nor a framework change is indicated for these matched T4 probes.
