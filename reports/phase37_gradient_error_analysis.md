# Phase 3.7 gradient error analysis

A gradient row passes when every element meets the original activation rule, `abs(error) <= 1e-5 + 1e-4 * abs(Paddle)`, with matching shape and dtype. The report also records absolute-error percentiles, relative L2, cosine similarity, fraction and count of failed elements, and magnitude and axis breakdowns where the layout can be assigned. Relative L2 uses a full tensor norm and is undefined when the reference has zero norm. A discrepancy in the natural graph can reflect upstream errors. The local replays compare operating points with both input and gradient fixed.

The accumulation diagnosis compares actual joint VJPs with a sum of individually propagated branch gradients, and repeats joint VJPs with the output request order reversed. At the shared pooled feature these comparisons measure the consumer accumulation directly. At earlier stages they additionally include rounding from executing operators once on jointly accumulated coefficients or separately for the two branches.

[Reproduction and input hashes](phase37_backward_attribution.md).
