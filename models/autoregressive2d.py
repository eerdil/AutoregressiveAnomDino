import torch.nn as nn
import torch

class MaskedConv2d(nn.Conv2d):
    """
    Causal 2D conv à la PixelCNN.
    mask_type:
      'A' = cannot see current pixel (strict autoregressive for first layer)
      'B' = can see current pixel but not future ones (for later layers)
    """
    def __init__(self, in_channels, out_channels, kernel_size, mask_type='A', **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        assert mask_type in ['A', 'B']
        self.mask_type = mask_type

        self.register_buffer("mask", torch.ones_like(self.weight))
        self._build_mask()

    def _build_mask(self):
        kH, kW = self.kernel_size
        yc, xc = kH // 2, kW // 2

        mask = torch.ones_like(self.weight)

        # Zero out "future" positions
        mask[:, :, yc+1:, :] = 0
        mask[:, :, yc, xc+1:] = 0

        if self.mask_type == 'A':
            # Also block the current position (yc, xc)
            mask[:, :, yc, xc] = 0

        self.mask = mask

    def forward(self, x):
        # apply mask to weights
        self.weight.data *= self.mask
        return super().forward(x)

class CenterMaskedConv2d(nn.Conv2d):
    """
    Center-masked conv: sees all neighbors except the center pixel.
    Used for the first layer in non-causal 'context' mode for anomaly detection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        self.register_buffer("mask", torch.ones_like(self.weight))
        self._build_mask()

    def _build_mask(self):
        kH, kW = self.kernel_size
        yc, xc = kH // 2, kW // 2

        mask = torch.ones_like(self.weight)
        # Zero out only the center location
        mask[:, :, yc, xc] = 0
        self.mask = mask

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
    
class AR2DModel(nn.Module):
    """
    2D autoregressive / reconstruction model over DINO feature maps.

    If causal=True:
        uses MaskedConv2d (AR / PixelCNN-style)
    If causal=False:
        uses standard Conv2d (sees both past and future pixels)

    Input : [B, C, H, W]
    Output: [B, C, H, W]  (prediction of features at each location)
    """
    def __init__(
        self,
        in_channels,
        hidden_channels=256,
        n_layers=5,
        kernel_size=3,
        causal=True,
        center_masked_first=False,
    ):
        super().__init__()
        self.causal = causal
        self.center_masked_first = center_masked_first

        if self.causal and self.center_masked_first:
            raise ValueError("center_masked_first cannot be True when causal=True.")

        padding = kernel_size // 2

        layers = []

        # First layer
        if causal:
            layers.append(
                MaskedConv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    mask_type='A',
                )
            )
        elif self.center_masked_first:
            layers.append(
                CenterMaskedConv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
        layers.append(nn.ReLU(inplace=True))

        # Middle layers
        for _ in range(n_layers - 2):
            if causal:
                layers.append(
                    MaskedConv2d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        mask_type='B',
                    )
                )
            else:
                layers.append(
                    nn.Conv2d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    )
                )
            layers.append(nn.ReLU(inplace=True))

        # Last layer
        if causal:
            layers.append(
                MaskedConv2d(
                    hidden_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    mask_type='B',
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    hidden_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)