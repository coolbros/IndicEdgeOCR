"""Paddle 3.2 CPU log-softmax formula and diagnostic FP32 reduction orders."""

import torch


def paddle_compatible_log_softmax(x, dim=-1, implementation="torch"):
    """Apply the CPU shifted-logit clamp before normalization.

    Autograd differentiates the decomposed forward, including the clamp.
    Paddle's registered backward uses the usual unclipped log-softmax gradient.
    """
    if implementation not in ("torch", "pairwise", "sequential"):
        raise ValueError(f"Unknown log-softmax implementation: {implementation}")
    if x.ndim == 0:
        return x.log_softmax(dim)
    values = x.movedim(dim, -1)
    if values.shape[-1] == 0:
        return x.clone()
    shifted = (values - values.amax(-1, keepdim=True)).clamp_min(-64.)
    terms = shifted.exp()
    if implementation == "torch":
        total = terms.sum(-1, keepdim=True)
    elif implementation == "pairwise":
        while terms.shape[-1] > 1:
            pairs = terms.shape[-1] // 2
            reduced = terms[..., :2 * pairs:2] + terms[..., 1:2 * pairs:2]
            terms = torch.cat((reduced, terms[..., 2 * pairs:]), dim=-1)
        total = terms
    else:
        total = terms[..., :1]
        for index in range(1, terms.shape[-1]):
            total = total + terms[..., index:index + 1]
    return (shifted - total.log()).movedim(-1, dim)
