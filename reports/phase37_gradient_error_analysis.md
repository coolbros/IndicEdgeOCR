# Phase 3.7 gradient error analysis

A gradient row passes when every element meets the original activation rule, `abs(error) <= 1e-5 + 1e-4 * abs(Paddle)`, with matching shape and dtype. The report also records absolute-error percentiles, relative L2, cosine similarity, fraction and count of failed elements, and magnitude and axis breakdowns where the layout can be assigned. Relative L2 uses a full tensor norm and is undefined when the reference has zero norm. A discrepancy in the natural graph can reflect upstream errors. The local replays compare operating points with both input and gradient fixed.

The accumulation diagnosis compares actual joint VJPs with a sum of individually propagated branch gradients, and repeats joint VJPs with the output request order reversed. At the shared pooled feature these comparisons measure the consumer accumulation directly. At earlier stages they additionally include rounding from executing operators once on jointly accumulated coefficients or separately for the two branches.

[Reproduction and input hashes](phase37_backward_attribution.md).

## T4 accumulation and attribution

At the shared pooled tensor, the maximum error between the explicit two-branch sum and the joint gradient is zero in Paddle and 2.980e-8 in Torch, passing every element. Reversing the joint output order gives zero change at both the shared tensor and image input in both frameworks. Summing the separately propagated *image* gradients differs from executing joint backward once by up to 1.650e-4 in Paddle and 1.373e-4 in Torch. Thus the distinction chiefly arises downstream of accumulation, where kernels round different intermediate quantities.

The native T4 full-loss image gradient has relative L2 3.083e-4. Substituting identical Paddle logit gradients gives 3.283e-4. The output-gradient differences alone therefore do not explain the remaining full-graph error on this fixture. See the [paired attribution report](phase37_backward_attribution.md) for input injections and exact replay checks.

The T4 block 2.0 input outliers are concentrated above reference magnitude 1 (6 of 9 for CTC, 15 of 18 for NRTR). The maximum is in channel 20 in both probes. The CTC 99th-percentile residual is 4.768e-7, compared with a 2.194e-2 maximum. The NRTR 99th-percentile residual is 9.537e-7, compared with a 3.516e-2 maximum. The whole decoder's embedding row has one outlier at weight index [80, 216]. Its gradient is sparse, and the maximum is more informative here than the all-element percentiles.

At the first failing **stage-level** T4 parameter replay, stage 5, the branch-specific mismatch includes the channel-mixer compress convolution weight in block 5.0 for both branches, and block 5.1 for NRTR. The stage result alone cannot attribute those particular weight outliers to the convolution's backward kernel because the weight receives a recomputed coefficient inside the stage. Exact block-level replays localize different strict misses in the corresponding compress BN scales. A remaining operator-level gap is the late-stage compress convolution and BN on independently fixed internal tensors for every block.

## CPU accumulation and backend sensitivity

The explicit branch sum at the pooled tensor differs from joint backward by zero in Paddle and 2.980e-8 in Torch. Both pass. Reversing the two joint outputs causes zero change at the pooled tensor and image in both frameworks. At image input, summing separately propagated branch gradients differs from joint backward by up to 1.144e-4 (Paddle) and 6.866e-5 (Torch). With the CTC and NRTR gradients fixed at pooled input, the remaining stage-2 differences appear even in isolated single-branch tests.

| Torch CPU setting | CTC image max / outliers, injected | NRTR image max / outliers, injected | Natural full-loss image relative L2 |
|---|---:|---:|---:|
| Two threads, MKLDNN on | 5.379e-4 / 1,163 | 1.472e-3 / 2,643 | 8.467e-3 |
| One thread, MKLDNN on | 4.692e-4 / 1,129 | 1.740e-3 / 2,972 | 1.260e-2 |
| Two threads, MKLDNN off | 3.929e-4 / 950 | 1.805e-3 / 2,968 | 1.164e-2 |

Switching backend or thread count changes the errors and does not remove the first strict stage-input boundary. In the full-loss CPU image gradient, absolute error has median 3.225e-3, 99th percentile 8.546e-2 and maximum 1.095. With the logit coefficients matched, the maximum is 1.119 and the relative L2 is 8.820e-3. Removing this source of coefficient mismatch leaves nearly all full-graph error. There is no measured basis for revisiting log-softmax as the dominant cause.
