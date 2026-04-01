"""
CNN architectures for liquid glass effect prediction.

All models: input [B, 3, H, W] → output [B, 3, H, W]
Fully convolutional with same-padding, so output resolution matches input.
"""

import torch
import torch.nn as nn


def _conv_block(in_ch, out_ch, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
        nn.ReLU(inplace=True),
    )


class MicroNet(nn.Module):
    """3 layers, 8 channels. ~600 params."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            _conv_block(3, 8),
            _conv_block(8, 8),
            nn.Conv2d(8, 3, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)  # Residual: predict the delta


class TinyNet(nn.Module):
    """5 layers, 16 channels. ~5K params."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            _conv_block(3, 16),
            _conv_block(16, 16),
            _conv_block(16, 16),
            _conv_block(16, 16),
            nn.Conv2d(16, 3, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class SmallNet(nn.Module):
    """7 layers, 32 channels. ~30K params."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 32),
            _conv_block(32, 32),
            _conv_block(32, 32),
            _conv_block(32, 32),
            _conv_block(32, 32),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class WideShallowNet(nn.Module):
    """2 layers, 64 channels. ~12K params. Prioritizes width over depth."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            _conv_block(3, 64),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class BottleneckNet(nn.Module):
    """Encoder-decoder with bottleneck. 5 layers, variable channels. ~15K params."""

    def __init__(self):
        super().__init__()
        self.enc1 = _conv_block(3, 16)
        self.enc2 = _conv_block(16, 32)
        self.bottleneck = _conv_block(32, 32)
        self.dec1 = _conv_block(32, 16)
        self.dec2 = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d1 = self.dec1(b + e2)  # Skip connection
        d2 = self.dec2(d1 + e1)  # Skip connection
        return x + d2


class DeepNarrowNet(nn.Module):
    """9 layers, 8 channels. ~5K params. Tests depth vs width tradeoff."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            _conv_block(3, 8),
            _conv_block(8, 8),
            _conv_block(8, 8),
            _conv_block(8, 8),
            _conv_block(8, 8),
            _conv_block(8, 8),
            _conv_block(8, 8),
            _conv_block(8, 8),
            nn.Conv2d(8, 3, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class UNet(nn.Module):
    """U-Net encoder-decoder with skip connections. ~8M params.

    Full receptive field via downsampling — can capture long-range
    displacement effects. Not for mobile, but establishes the quality ceiling.
    """

    def __init__(self, base_ch=64):
        super().__init__()
        # Encoder
        self.enc1 = self._double_conv(3, base_ch)
        self.enc2 = self._double_conv(base_ch, base_ch * 2)
        self.enc3 = self._double_conv(base_ch * 2, base_ch * 4)
        self.enc4 = self._double_conv(base_ch * 4, base_ch * 8)

        # Bottleneck
        self.bottleneck = self._double_conv(base_ch * 8, base_ch * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = self._double_conv(base_ch * 16, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = self._double_conv(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = self._double_conv(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = self._double_conv(base_ch * 2, base_ch)

        self.final = nn.Conv2d(base_ch, 3, 1)
        self.pool = nn.MaxPool2d(2)

    @staticmethod
    def _double_conv(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return x + self.final(d1)


# Registry for easy access
MODELS = {
    "micro": MicroNet,
    "tiny": TinyNet,
    "small": SmallNet,
    "wide_shallow": WideShallowNet,
    "bottleneck": BottleneckNet,
    "deep_narrow": DeepNarrowNet,
    "unet": UNet,
}


def get_model(name: str) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
    return MODELS[name]()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: nn.Module, size: int = 256) -> int:
    """Rough FLOPs estimate for a single forward pass."""
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # FLOPs = 2 * K*K * C_in * C_out * H * W
            k = m.kernel_size[0] * m.kernel_size[1]
            total += 2 * k * m.in_channels * m.out_channels * size * size
    return total
