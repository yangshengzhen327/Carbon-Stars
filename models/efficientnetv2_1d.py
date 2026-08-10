"""
EfficientNetV2-style 1D model for spectral classification.

Interface:
    EfficientNetV2_1D(in_channel, out_channel, spectrum_size)
"""

import torch
import torch.nn as nn


class ConvBNAct1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, groups: int = 1, act: bool = True):
        super().__init__()
        padding = kernel // 2
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm1d(out_ch),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(8, int(channels * se_ratio))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.gate(w)
        return x * w


class FusedMBConv1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden = int(in_ch * expand_ratio)
        self.use_residual = stride == 1 and in_ch == out_ch

        if expand_ratio == 1:
            self.block = ConvBNAct1D(in_ch, out_ch, kernel=3, stride=stride, act=True)
        else:
            self.block = nn.Sequential(
                ConvBNAct1D(in_ch, hidden, kernel=3, stride=stride, act=True),
                ConvBNAct1D(hidden, out_ch, kernel=1, stride=1, act=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class MBConv1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = int(in_ch * expand_ratio)
        self.use_residual = stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct1D(in_ch, hidden, kernel=1, stride=1, act=True))
        else:
            hidden = in_ch
        layers.extend(
            [
                ConvBNAct1D(hidden, hidden, kernel=3, stride=stride, groups=hidden, act=True),
                SEBlock1D(hidden, se_ratio=se_ratio),
                ConvBNAct1D(hidden, out_ch, kernel=1, stride=1, act=False),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class EfficientNetV2_1D(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, spectrum_size: int):
        super().__init__()
        _ = spectrum_size

        self.stem = ConvBNAct1D(in_channel, 24, kernel=3, stride=2, act=True)

        stage_cfg = [
            # repeats, out_ch, stride, expand_ratio, use_fused
            (2, 24, 1, 1, True),
            (3, 48, 2, 4, True),
            (4, 64, 2, 4, True),
            (4, 128, 2, 4, False),
            (6, 160, 1, 6, False),
        ]

        layers = []
        in_ch = 24
        for repeats, out_ch, stride, expand_ratio, use_fused in stage_cfg:
            for i in range(repeats):
                s = stride if i == 0 else 1
                if use_fused:
                    layers.append(FusedMBConv1D(in_ch, out_ch, stride=s, expand_ratio=expand_ratio))
                else:
                    layers.append(MBConv1D(in_ch, out_ch, stride=s, expand_ratio=expand_ratio))
                in_ch = out_ch
        self.features = nn.Sequential(*layers)

        self.head = nn.Sequential(
            ConvBNAct1D(in_ch, 256, kernel=1, stride=1, act=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(256, out_channel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        return x
