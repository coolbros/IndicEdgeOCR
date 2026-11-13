"""Parameter groups for the PaddleOCR Adam/L2 configuration."""

import math

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR


def optimizer_for(model, lr=0.0005):
    groups = {0.0: [], 1e-5: [], 3e-5: []}
    for name, parameter in model.named_parameters():
        decay = 3e-5
        if name.startswith("backbone.conv1.") and ".bn." in name:
            decay = 0.0
        elif name.startswith("head.ctc_head.fc."):
            decay = 1e-5
        groups[decay].append(parameter)
    return Adam([{"params": params, "weight_decay": decay} for decay, params in groups.items()],
                lr=lr, betas=(0.9, 0.999), eps=1e-8, decoupled_weight_decay=False)


def cosine_schedule(optimizer, steps_per_epoch, epochs=100, warmup_epochs=5):
    warmup = round(steps_per_epoch * warmup_epochs)
    total = steps_per_epoch * epochs

    def factor(step):
        if step < warmup:
            return step / warmup
        return (1 + math.cos(math.pi * (step - warmup) / total)) / 2

    return LambdaLR(optimizer, factor)
