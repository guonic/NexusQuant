# -*- coding: utf-8 -*-
"""
Multi-scale CNN for K-line pattern recognition.

Three branches (kernel 3, 7, 11) capture short (3-day), medium (15-day), and long (60-day) structure.
Input: (batch, 5, 60) — channels [open, high, low, close, volume], 60 time steps.
Output: (batch,) — probability in [0, 1].
"""

import torch
import torch.nn as nn


class MultiScaleCNN(nn.Module):
    """
    Multi-scale 1D CNN for K-line pattern (e.g. step-down + V-reversal).

    Branches:
    - kernel 3: short-term (e.g. V-reversal tip).
    - kernel 7: medium-term (e.g. 2-week platform).
    - kernel 11: long-term (overall trend).
    """

    def __init__(self, in_channels: int = 5, seq_len: int = 60):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        out_per_branch = 16
        pool_size = 2
        after_pool = seq_len // pool_size  # 30

        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
        )

        fc_in = out_per_branch * 3 * after_pool
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_combined = torch.cat((x1, x2, x3), dim=1)
        x_flat = x_combined.view(x_combined.size(0), -1)
        return self.fc(x_flat).squeeze(-1)
