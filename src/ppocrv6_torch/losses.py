"""PaddleOCR CTC and smoothed NRTR loss reductions."""

import torch
from torch.nn import functional as F


def multi_loss(outputs, ctc, nrtr, lengths):
    ctc_logits = outputs["ctc"].float()
    input_lengths = torch.full((ctc.shape[0],), ctc_logits.shape[1], dtype=torch.long)
    ctc_loss = F.ctc_loss(ctc_logits.log_softmax(2).transpose(0, 1), ctc,
                          input_lengths, lengths.cpu(), blank=0, reduction="none").mean()
    logits = outputs["gtc"].float()
    tokens = nrtr[:, 1:1 + logits.shape[1]]
    logp = logits.log_softmax(-1)
    target = logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    smoothed = -0.9 * target - (0.1 / (logits.shape[-1] - 1)) * (logp.sum(-1) - target)
    nrtr_loss = smoothed[tokens != 0].mean()
    return {"CTCLoss": ctc_loss, "NRTRLoss": nrtr_loss, "loss": ctc_loss + nrtr_loss}
