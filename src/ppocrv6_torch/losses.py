"""PaddleOCR CTC and smoothed NRTR loss reductions."""

import torch
from torch.nn import functional as F
from .ops.paddle_log_softmax import paddle_compatible_log_softmax


def _log_softmax(x, dim, implementation):
    if implementation == "native":
        return x.log_softmax(dim)
    return paddle_compatible_log_softmax(x, dim, implementation)


def multi_loss(outputs, ctc, nrtr, lengths, *, ctc_log_softmax="native",
               nrtr_log_softmax="native", blank=0, pad=0):
    ctc_logits = outputs["ctc"].float()
    input_lengths = torch.full((ctc.shape[0],), ctc_logits.shape[1], dtype=torch.long)
    ctc_loss = F.ctc_loss(_log_softmax(ctc_logits, 2, ctc_log_softmax).transpose(0, 1), ctc,
                          input_lengths, lengths.cpu(), blank=blank, reduction="none").mean()
    logits = outputs["gtc"].float()
    tokens = nrtr[:, 1:1 + logits.shape[1]]
    logp = _log_softmax(logits, -1, nrtr_log_softmax)
    target = logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    smoothed = -0.9 * target - (0.1 / (logits.shape[-1] - 1)) * (logp.sum(-1) - target)
    nrtr_loss = smoothed[tokens != pad].mean()
    return {"CTCLoss": ctc_loss, "NRTRLoss": nrtr_loss, "loss": ctc_loss + nrtr_loss}
