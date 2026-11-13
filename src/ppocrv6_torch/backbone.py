# Architecture adapted from PaddleOCR (Copyright PaddlePaddle Authors).
# Licensed under the Apache License, Version 2.0.
"""Recognition-only LCNetV4 medium backbone."""

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


class BatchNorm2d(nn.Module):
    """Paddle FP32 CPU and four-dimensional CUDA/cuDNN running statistics."""

    def __init__(self, channels: int, gamma: float = 1.0):
        super().__init__()
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.full((channels,), gamma))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, x):
        if not self.training:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight, self.bias, eps=self.eps)
        count = x.numel() // x.shape[1]
        if count == 1:
            return x
        # CUDA's 4D path uses cuDNN's sample variance for the running estimate.
        if x.is_cuda:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight, self.bias, True, 0.1, self.eps)
        variance, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0)
        with torch.no_grad():
            self.running_mean.lerp_(mean, 0.1)
            self.running_var.lerp_(variance, 0.1)
        return ((x - mean[None, :, None, None])
                * torch.rsqrt(variance[None, :, None, None] + self.eps)
                * self.weight[None, :, None, None] + self.bias[None, :, None, None])


class EvenSameConv(nn.Conv2d):
    def forward(self, x):
        return super().forward(F.pad(x, (0, 1, 0, 1)))


class EvenSameMaxPool(nn.Module):
    def forward(self, x):
        return F.max_pool2d(F.pad(x, (0, 1, 0, 1), value=-torch.inf), 2, 1)


def conv2d(in_channels, out_channels, kernel, stride=1, padding=0, groups=1,
           *, bias=False, uniform=False):
    conv = (EvenSameConv if padding == "SAME" else nn.Conv2d)(
        in_channels, out_channels, kernel, stride,
        0 if padding == "SAME" else padding, groups=groups, bias=bias,
    )
    # Paddle default Conv2D fan-in uses the full input channel count.
    fan_in = in_channels * conv.kernel_size[0] * conv.kernel_size[1]
    with torch.no_grad():
        if uniform:
            nn.init.kaiming_uniform_(conv.weight)
        else:
            conv.weight.normal_(0.0, (2.0 / fan_in) ** 0.5)
        if bias:
            conv.bias.zero_()
    return conv


@torch.no_grad()
def fuse_conv_bn(conv, bn):
    fused = type(conv)(conv.in_channels, conv.out_channels, conv.kernel_size,
                       conv.stride, conv.padding, conv.dilation, conv.groups,
                       bias=True, device=conv.weight.device, dtype=conv.weight.dtype)
    scale = bn.weight * torch.rsqrt(bn.running_var + bn.eps)
    fused.weight.copy_(conv.weight * scale[:, None, None, None])
    bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.bias)
    fused.bias.copy_(bn.bias + (bias - bn.running_mean) * scale)
    return fused.eval()


class Conv2D_BN(nn.Sequential):
    def __init__(self, cin, cout, kernel=1, stride=1, padding=0, groups=1, gamma=1.0):
        super().__init__(OrderedDict(
            conv=conv2d(cin, cout, kernel, stride, padding, groups),
            bn=BatchNorm2d(cout, gamma),
        ))

    def fuse(self):
        return fuse_conv_bn(self.conv, self.bn)


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, kernel, stride=1, padding=None):
        super().__init__()
        padding = (kernel - 1) // 2 if padding is None else padding
        self.conv = conv2d(cin, cout, kernel, stride, padding)
        self.bn = BatchNorm2d(cout)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def rep(self):
        if isinstance(self.bn, BatchNorm2d):
            self.conv = fuse_conv_bn(self.conv, self.bn)
            self.bn = nn.Identity()


class StemBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem1 = ConvBNAct(3, 64, 3, 2)
        self.stem2a = ConvBNAct(64, 32, 2, padding="SAME")
        self.stem2b = ConvBNAct(32, 64, 2, padding="SAME")
        self.stem3 = ConvBNAct(128, 64, 3, 2)
        self.stem4 = ConvBNAct(64, 128, 1)
        self.pool = EvenSameMaxPool()

    def forward(self, x):
        x = self.stem1(x)
        branch = self.stem2b(self.stem2a(x))
        return self.stem4(self.stem3(torch.cat((self.pool(x), branch), dim=1)))

    def rep(self):
        for layer in (self.stem1, self.stem2a, self.stem2b, self.stem3, self.stem4):
            layer.rep()


class SELayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = conv2d(channels, channels // 4, 1, bias=True)
        self.relu = nn.ReLU()
        self.conv2 = conv2d(channels // 4, channels, 1, bias=True)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        z = x.mean((2, 3), keepdim=True)
        return x * self.hardsigmoid(self.conv2(self.relu(self.conv1(z))))


class RepDWConv(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.padding = kernel_size // 2
        self.conv = Conv2D_BN(channels, channels, kernel_size, padding=self.padding,
                              groups=channels)
        self.conv1 = conv2d(channels, channels, 1, groups=channels)
        self.bn = BatchNorm2d(channels)
        self.reparam_conv = None

    def forward(self, x):
        if self.reparam_conv is not None:
            return self.reparam_conv(x)
        return self.bn(self.conv(x) + self.conv1(x) + x)

    @torch.no_grad()
    def fuse(self):
        if self.reparam_conv is not None:
            return self.reparam_conv
        inner = self.conv.fuse()
        inner.weight.add_(F.pad(self.conv1.weight + 1, (self.padding,) * 4))
        return fuse_conv_bn(inner, self.bn)

    def rep(self):
        if self.reparam_conv is None:
            self.reparam_conv = self.fuse()
            del self.conv, self.conv1, self.bn


class LCNetV4Block(nn.Module):
    def __init__(self, cin, cout, stride, se):
        super().__init__()
        self.has_residual = cin == cout and stride == 1
        mixer = OrderedDict()
        if self.has_residual:
            mixer["rep_dw"] = RepDWConv(cin)
        else:
            mixer["dw_conv"] = Conv2D_BN(cin, cin, 3, stride, 1, cin)
        if se:
            mixer["se"] = SELayer(cin)
        self.token_mixer = nn.Sequential(mixer)
        self.channel_mixer = nn.Sequential(OrderedDict(
            expand=Conv2D_BN(cin, 2 * cin),
            act=nn.GELU(approximate="none"),
            compress=Conv2D_BN(2 * cin, cout, gamma=0.0 if self.has_residual else 1.0),
        ))

    def forward(self, x):
        x = self.token_mixer(x)
        y = self.channel_mixer(x)
        return x + y if self.has_residual else y

    def rep(self):
        if self.has_residual:
            self.token_mixer.rep_dw.rep()
        elif isinstance(self.token_mixer.dw_conv, Conv2D_BN):
            self.token_mixer.dw_conv = self.token_mixer.dw_conv.fuse()
        for name in ("expand", "compress"):
            layer = getattr(self.channel_mixer, name)
            if isinstance(layer, Conv2D_BN):
                setattr(self.channel_mixer, name, layer.fuse())


NET_CONFIG_REC = (
    ((128, 128, 1, True),),
    ((128, 256, 1, False), (256, 256, 1, False), (256, 256, 1, True)),
    ((256, 512, (2, 1), False),) + ((512, 512, 1, True), (512, 512, 1, False)) * 3,
    ((512, 768, (2, 1), False), (768, 768, 1, True), (768, 768, 1, False)),
    (),
)


class PPLCNetV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = StemBlock()
        for index, config in enumerate(NET_CONFIG_REC, 2):
            self.add_module(f"blocks{index}", nn.Sequential(*(LCNetV4Block(*x) for x in config)))

    def forward(self, x):
        x = self.conv1(x)
        for index in range(2, 7):
            x = getattr(self, f"blocks{index}")(x)
        if self.training:
            return F.adaptive_avg_pool2d(x, (1, 40))
        if x.shape[2] < 3:
            raise ValueError("Evaluation pooling requires feature height >= 3")
        return F.avg_pool2d(x, (3, 2))

    def rep(self):
        if self.training:
            raise ValueError("Backbone reparameterization requires eval mode")
        self.conv1.rep()
        for index in range(2, 7):
            for block in getattr(self, f"blocks{index}"):
                block.rep()
