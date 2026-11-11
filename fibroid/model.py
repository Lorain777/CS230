"""2D U-Net and the DropBlock / random-skip-hardship variants.

Variants
--------
standard       Vanilla U-Net.
dropblock      DropBlock after every convolution (Ghiasi et al., NeurIPS 2018).
skip_hardship  DropBlock plus, during training only, randomly zeroing whole
               skip connections. The name is not a published method: it means
               making the decoder work without some of its encoder shortcuts.
"""

import torch
import torch.nn.functional as F
from torch import nn

VARIANTS = ("standard", "dropblock", "skip_hardship")


class DropBlock2d(nn.Module):
    """Contiguous spatial dropout for convolutional feature maps.

    During training a keep_prob schedule typically ramps from 1.0 down to the
    final value; call `set_keep_prob` each epoch. At inference the layer is a
    no-op. Masks are drawn independently per channel, as recommended in the
    DropBlock paper.
    """

    def __init__(self, block_size=7, keep_prob=0.9):
        super().__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob

    def set_keep_prob(self, keep_prob):
        self.keep_prob = float(keep_prob)

    def forward(self, x):
        if not self.training or self.keep_prob >= 1.0:
            return x
        _, _, height, width = x.shape
        block = min(self.block_size, height, width)
        if block < 1:
            return x

        # gamma is the seed-drop probability that yields keep_prob after the
        # seeds expand into block x block squares that stay inside the map.
        gamma = (
            (1.0 - self.keep_prob)
            * height
            * width
            / (block * block * max(height - block + 1, 1) * max(width - block + 1, 1))
        )
        # Sample only in the valid centre so every expanded block fits.
        padding = (block - 1) // 2
        seeds = torch.zeros_like(x)
        centre = torch.rand(x.shape[0], x.shape[1], height - block + 1, width - block + 1, device=x.device)
        seeds[:, :, padding : padding + centre.shape[2], padding : padding + centre.shape[3]] = (
            centre < gamma
        ).to(x.dtype)
        mask = 1.0 - F.max_pool2d(seeds, kernel_size=block, stride=1, padding=padding)
        # Odd block sizes keep the pooled shape equal to the input; clamp just in case.
        if mask.shape[-2:] != x.shape[-2:]:
            mask = mask[..., :height, :width]
        return x * mask * (mask.numel() / (mask.sum() + 1e-6))


def conv_block(in_channels, out_channels, dropblock=False, block_size=7, keep_prob=0.9):
    """Two 3x3 convolutions. With DropBlock the order matches SD/SA-UNet:
    Conv → DropBlock → BatchNorm → ReLU."""
    layers = []
    for cin, cout in ((in_channels, out_channels), (out_channels, out_channels)):
        layers.append(nn.Conv2d(cin, cout, 3, padding=1, bias=False))
        if dropblock:
            layers.append(DropBlock2d(block_size, keep_prob))
        layers.append(nn.BatchNorm2d(cout))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    """Encoder stages of `channels` widths joined by 2x max-pooling; the last
    stage is the bottleneck. The decoder mirrors the encoder with skip
    connections and ends in a 1x1 convolution that outputs logits."""

    def __init__(
        self,
        in_channels,
        out_channels=1,
        channels=(32, 64, 128, 256),
        variant="standard",
        dropblock_keep_prob=0.9,
        dropblock_size=7,
        skip_drop_prob=0.25,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        self.variant = variant
        self.skip_drop_prob = float(skip_drop_prob)
        use_drop = variant in ("dropblock", "skip_hardship")

        widths = [in_channels, *channels]
        self.encoders = nn.ModuleList(
            conv_block(a, b, use_drop, dropblock_size, dropblock_keep_prob)
            for a, b in zip(widths, widths[1:])
        )
        self.pool = nn.MaxPool2d(2)
        pairs = list(zip(channels[:0:-1], channels[-2::-1]))  # (256, 128), (128, 64), (64, 32)
        self.upsamplers = nn.ModuleList(nn.ConvTranspose2d(a, b, 2, stride=2) for a, b in pairs)
        self.decoders = nn.ModuleList(
            conv_block(2 * b, b, use_drop, dropblock_size, dropblock_keep_prob) for _, b in pairs
        )
        self.head = nn.Conv2d(channels[0], out_channels, 1)

    def set_keep_prob(self, keep_prob):
        for module in self.modules():
            if isinstance(module, DropBlock2d):
                module.set_keep_prob(keep_prob)

    def _drop_skip(self, skip):
        """Zero whole skip maps during training for the skip_hardship variant.

        A per-sample Bernoulli gate is used so some images in a batch keep a
        given skip while others lose it. At inference every skip is kept.
        """
        if self.variant != "skip_hardship" or not self.training or self.skip_drop_prob <= 0:
            return skip
        keep = (torch.rand(skip.shape[0], 1, 1, 1, device=skip.device) >= self.skip_drop_prob).to(
            skip.dtype
        )
        return skip * keep

    def forward(self, x):
        skips = []
        for encoder in self.encoders[:-1]:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.encoders[-1](x)
        for upsample, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skips)):
            x = decoder(torch.cat([upsample(x), self._drop_skip(skip)], dim=1))
        return self.head(x)
