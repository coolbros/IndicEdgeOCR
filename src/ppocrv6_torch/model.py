# Architecture adapted from PaddleOCR (Copyright PaddlePaddle Authors).
# Licensed under the Apache License, Version 2.0.
"""Native PyTorch PaddleOCR medium recognizer with the auxiliary NRTR head."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import BatchNorm2d, PPLCNetV4, conv2d
from .vocabulary import RecognitionTokens


class Mlp(nn.Module):
    def __init__(self, dim, activation):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = activation()
        self.fc2 = nn.Linear(4 * dim, dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def projections(x, count, heads):
    batch, tokens, _ = x.shape
    return x.reshape(batch, tokens, count, heads, -1).permute(2, 0, 3, 1, 4).unbind()


def attend(scores, value, dropout):
    context = dropout(scores.softmax(dim=-1)) @ value
    return context.transpose(1, 2).flatten(2)


class SVTRAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(192, 576)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(192, 192)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        q, k, v = projections(self.qkv(x), 3, 8)
        scores = (q * 24**-0.5) @ k.transpose(-2, -1)
        return self.proj_drop(self.proj(attend(scores, v, self.attn_drop)))


class SVTRBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(192, eps=1e-5)
        self.mixer = SVTRAttention()
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(192, eps=1e-5)
        self.mlp = Mlp(192, nn.SiLU)

    def forward(self, x):
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class ConvBNLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = conv2d(768, 192, 1, uniform=True)
        self.norm = BatchNorm2d(192)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class EncoderWithLightSVTR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_reduce = ConvBNLayer()
        local = nn.Conv2d(192, 192, (1, 7), padding=(0, 3), groups=192, bias=False)
        nn.init.kaiming_normal_(local.weight)
        self.local_conv = nn.Sequential(local, BatchNorm2d(192), nn.SiLU())
        self.svtr_block = nn.ModuleList([SVTRBlock(), SVTRBlock()])
        self.norm = nn.LayerNorm(192, eps=1e-6)
        self.skip_conv = ConvBNLayer()
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02, a=-0.04, b=0.04)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        skip = self.skip_conv(x)
        z = self.conv_reduce(x)
        z = z + self.local_conv(z)
        height, width = z.shape[2:]
        z = z.flatten(2).transpose(1, 2)
        for block in self.svtr_block:
            z = block(z)
        z = self.norm(z).transpose(1, 2).reshape(-1, 192, height, width)
        return z + skip


class Im2Seq(nn.Module):
    def forward(self, x):
        if x.shape[2] != 1:
            raise ValueError("Im2Seq requires a singleton height")
        return x.squeeze(2).transpose(1, 2)


class SequenceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_reshape = Im2Seq()
        self.encoder = EncoderWithLightSVTR()

    def forward(self, x):
        return self.encoder_reshape(self.encoder(x))


class CTCHead(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.fc = nn.Linear(192, classes)

    def forward(self, x):
        logits = self.fc(x)
        return logits if self.training else logits.softmax(dim=2)


class FCTranspose(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 512, bias=False)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        return self.fc(x.transpose(1, 2))


class Embeddings(nn.Module):
    def __init__(self, classes, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(classes, 512, padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, std=512**-0.5)

    def forward(self, tokens):
        # Paddle zeroes the output at padding positions even with a nonzero row.
        x = self.embedding(tokens).masked_fill(tokens.unsqueeze(-1) == self.embedding.padding_idx, 0.0)
        return x * math.sqrt(512)


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        position = torch.arange(5000, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, 512, 2) * (-math.log(10000.0) / 512))
        pe = torch.stack(((position * scale).sin(), (position * scale).cos()), dim=-1)
        self.register_buffer("pe", pe.flatten(1).unsqueeze(1))

    def forward(self, x):
        return self.dropout(x.transpose(0, 1) + self.pe[:x.shape[1]]).transpose(0, 1)


class NRTRAttention(nn.Module):
    def __init__(self, self_attn):
        super().__init__()
        self.self_attn = self_attn
        if self_attn:
            self.qkv = nn.Linear(512, 1536)
        else:
            self.q = nn.Linear(512, 512)
            self.kv = nn.Linear(512, 1024)
        self.attn_drop = nn.Dropout(0.0)
        self.out_proj = nn.Linear(512, 512)

    def forward(self, query, memory=None, mask=None):
        if self.self_attn:
            q, k, v = projections(self.qkv(query), 3, 16)
        else:
            q, = projections(self.q(query), 1, 16)
            k, v = projections(self.kv(memory), 2, 16)
        scores = (q @ k.transpose(-2, -1)) * 32**-0.5
        if mask is not None:
            scores = scores + mask
        return self.out_proj(attend(scores, v, self.attn_drop))


class NRTRBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = NRTRAttention(True)
        self.cross_attn = NRTRAttention(False)
        self.norm1 = nn.LayerNorm(512, eps=1e-5)
        self.norm2 = nn.LayerNorm(512, eps=1e-5)
        self.norm3 = nn.LayerNorm(512, eps=1e-5)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.mlp = Mlp(512, nn.ReLU)

    def forward(self, x, memory, mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, mask=mask)))
        x = self.norm2(x + self.dropout2(self.cross_attn(x, memory)))
        return self.norm3(x + self.dropout3(self.mlp(x)))


class NRTRHead(nn.Module):
    def __init__(self, classes, padding_idx=0):
        super().__init__()
        self.embedding = Embeddings(classes, padding_idx)
        self.positional_encoding = PositionalEncoding()
        self.decoder = nn.ModuleList(NRTRBlock() for _ in range(4))
        self.tgt_word_prj = nn.Linear(512, classes, bias=False)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, memory, targets, lengths):
        tokens = targets[:, :int(lengths.max()) + 1]
        x = self.positional_encoding(self.embedding(tokens))
        mask = torch.full((tokens.shape[1],) * 2, -torch.inf, device=x.device,
                          dtype=x.dtype).triu(1)[None, None]
        for layer in self.decoder:
            x = layer(x, memory, mask)
        return self.tgt_word_prj(x)


class MultiHead(nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.encoder_reshape = Im2Seq()
        self.ctc_encoder = SequenceEncoder()
        self.ctc_head = CTCHead(tokens.ctc_classes)
        self.before_gtc = nn.Sequential(nn.Flatten(2), FCTranspose())
        self.gtc_head = NRTRHead(tokens.nrtr_classes, tokens.pad)

    def forward(self, x, targets=None):
        z = self.ctc_encoder(x)
        ctc = self.ctc_head(z)
        if not self.training:
            return ctc
        if targets is None:
            raise ValueError("Training forward requires CTC and NRTR labels and lengths")
        _, nrtr, lengths, *_ = targets
        return {"ctc": ctc, "ctc_neck": z,
                "gtc": self.gtc_head(self.before_gtc(x), nrtr, lengths)}


class PPOCRv6Medium(nn.Module):
    """Input is normalized NCHW. Training targets follow Paddle's KeepKeys order."""

    def __init__(self, ctc_classes=18710, nrtr_classes=18714, *, blank=0, pad=0, bos=2, eos=3):
        super().__init__()
        self.tokens = RecognitionTokens(ctc_classes, nrtr_classes, blank, pad, bos, eos)
        self.backbone = PPLCNetV4()
        self.head = MultiHead(self.tokens)

    def forward(self, x, targets=None):
        return self.head(self.backbone(x), targets)

    def rep(self):
        self.backbone.rep()
        return self
