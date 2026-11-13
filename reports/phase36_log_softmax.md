# Phase 3.6 Paddle log-softmax

## Implementation

PaddlePaddle 3.2.0's dense CPU FP32 `LogSoftmaxKernel` uses `LogSoftmaxFunctor` with row-major Eigen tensor expressions. It flattens dimensions preceding the canonical axis and preserves trailing dimensions. It subtracts the reduction-axis maximum and clamps each shifted value below −64 before computing the exponential. It then sums the exponentials on the same axis, takes their log and subtracts it from each clamped value. Scalars return zero and empty inputs bypass the reduction. The CPU experiment disables oneDNN through the inherited runner flag. [Paddle CPU forward source](https://github.com/PaddlePaddle/Paddle/blob/v3.2.0/paddle/phi/kernels/cpu/log_softmax_kernel.cc).

The Paddle build pins Eigen at `f612df273689a19d25b45ca4f8269463207c4fee`. For a contiguous innermost floating-point sum, Eigen can use a recursive split at 1,024 scalar coefficients or 1,024 SIMD packets. Packet leaves can accumulate in two separate accumulators before combining them with the scalar tail. Outer-axis reductions follow a different path. The wheel's effective compilation flags and packet width were not established. Eigen exp/log approximations and reassociation can both contribute to error. [Paddle Eigen build configuration](https://github.com/PaddlePaddle/Paddle/blob/v3.2.0/cmake/external/eigen.cmake) · [Pinned Eigen reduction source](https://gitlab.com/libeigen/eigen/-/blob/f612df273689a19d25b45ca4f8269463207c4fee/unsupported/Eigen/CXX11/src/Tensor/TensorReduction.h).

The opt-in `paddle_compatible_log_softmax` applies the same finite-input shifted clamp and normalizes with a PyTorch sum, a pairwise tree, or a sequential diagnostic. Its native autograd differentiates the actual expression, including its clamp. Paddle registers a backward computing `dy - exp(y) * sum(dy)` directly. Those derivatives agree mathematically in the unclipped region and can differ substantially on clipped inputs. The implementation uses ordinary differentiable primitives throughout. [Paddle CPU backward source](https://github.com/PaddlePaddle/Paddle/blob/v3.2.0/paddle/phi/kernels/cpu/log_softmax_grad_kernel.cc).

The real CTC and NRTR archived logits have minimum max-shifted values of about −46.37 and −18.62, respectively, so their differences cannot be attributed to clipping. The probes also exercise leading and middle axes, a strided Torch input, ties, nonfinite inputs and large values near the threshold. Finite-difference unit tests cover both sides of the clamp and avoid its nondifferentiable point. Timings and temporary tensor sizes are recorded separately from correctness. Working tensor byte estimates exclude allocator overhead and saved autograd state.

The existing loss implementation remains the default. `multi_loss` accepts independent `ctc_log_softmax` and `nrtr_log_softmax` keyword arguments. Paddle's Warp-CTC normalizes internally. The actual CTC and smoothed NRTR loss and backward comparisons are in [Gradient parity](phase36_gradient_parity.md). Reproduce the operator probes through the CPU command in [Numerical equivalence](phase36_numerical_equivalence.md).

## Real-logit results

| Captured input | Reduction | Forward max error | Forward outliers | Gradient max error | Gradient outliers |
|---|---|---:|---:|---:|---:|
| CTC | Native Torch | 3.405e-4 | 54 | 6.067e-4 | 8 |
| CTC | Decomposed Torch sum | 3.633e-4 | 53 | 6.496e-4 | 8 |
| CTC | Pairwise | 3.643e-4 | 53 | 6.496e-4 | 8 |
| CTC | Sequential | 9.313e-10 | 0 | 6.586e-4 | 11 |
| NRTR | Native Torch | 1.469e-4 | 21 | 3.488e-4 | 5 |
| NRTR | Decomposed Torch sum | 1.550e-4 | 21 | 3.686e-4 | 6 |
| NRTR | Pairwise | 1.564e-4 | 21 | 3.684e-4 | 6 |
| NRTR | Sequential | 0 | 0 | 3.872e-4 | 8 |

The sequential sum closes both original standalone **forward** misses. This strongly implicates accumulation order on the sampled wheel. It does not establish the wheel's exact SIMD build configuration. The same reduction worsens both real-logit gradient comparisons. On the threshold and tie probe the decomposed implementations match the Paddle forward exactly, while their finite-difference-validated gradient differs by up to 0.024. Native Torch returns negative infinity for a finite-plus-negative-infinity row, whereas Paddle and the clamped operation make that entry finite. All three showed matching NaN placement on the tested all-negative-infinity, positive-infinity and NaN rows.

The sequential path runs thousands of concatenation-free additions in Python and retains an autograd graph through them. A native extension could reduce its latency, but it would still need to differentiate the actual clamped expression. This phase did not add a compiled dependency because reproducing the sequential forward alone provides no demonstrated loss or gradient benefit. Native Torch remains the production choice.

On the paired worker, the cold forward took 163 ms versus 5.7 ms for native Torch on the CTC tensor (4×40×18,710), and 148 ms versus 14.8 ms on the NRTR tensor (4×7×18,714). These single-call times include cold-start variability. The decomposition creates at least 22.84 MiB of full-size shifted and exponential temporary tensors on CTC and 4.00 MiB on NRTR, excluding the output and autograd storage. Random FP32 and strided middle-axis comparisons pass forward and backward for every choice. On the extreme leading-axis probe the clamped implementations match Paddle's forward exactly and have 32 gradient mismatches.

The repeated CPU job measured the following warmed no-grad forward medians over five calls. [CPU timing metrics](https://www.kaggle.com/code/agastyavar/indic-edge-ocr-phase36-cpu/versions/1).

| Input | Native Torch | Torch sum | Pairwise sum | Sequential |
|---|---:|---:|---:|---:|
| CTC | 3.29 ms | 5.43 ms | 9.17 ms | 151.21 ms |
| NRTR | 0.47 ms | 0.68 ms | 1.65 ms | 139.49 ms |
