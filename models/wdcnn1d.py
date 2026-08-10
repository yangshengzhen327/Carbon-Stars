"""
WDCNN1D for 1D spectral binary classification.

The constructor keeps the same interface used by the training script:
    WDCNN1D(in_channel, out_channel, spectrum_size)
"""

import torch.nn as nn


class WDCNN1D(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, spectrum_size: int):
        super().__init__()
        _ = spectrum_size

        self.features = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64, stride=16, padding=24, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, out_channel),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
