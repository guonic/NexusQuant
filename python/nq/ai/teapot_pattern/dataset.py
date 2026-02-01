# -*- coding: utf-8 -*-
"""
Dataset and normalization for Teapot K-line pattern (5, 60) tensors.
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_slice_for_model(data_slice: np.ndarray) -> np.ndarray:
    """
    Normalize 60 x 5 slice [open, high, low, close, volume] for model input.
    Same logic as seed dataset builder: price min-max, volume min-max; output (5, 60).
    """
    data_slice = np.asarray(data_slice, dtype=np.float64)
    if data_slice.size == 0:
        raise ValueError("Empty slice")
    prices = data_slice[:, :4]
    p_min, p_max = np.min(prices), np.max(prices)
    norm_prices = (prices - p_min) / (p_max - p_min + 1e-9)
    v = data_slice[:, 4:5]
    v_min, v_max = np.min(v), np.max(v)
    norm_v = (v - v_min) / (v_max - v_min + 1e-9)
    combined = np.hstack((norm_prices, norm_v))
    return combined.T.astype(np.float32)


class KLineDataset(Dataset):
    """
    Binary dataset: positive .pt files (label=1) + negative .pt files (label=0).
    Optional augmentation: Gaussian noise on price, or time shift.
    """

    def __init__(
        self,
        pos_dir: str,
        neg_dir: str,
        noise_std: float = 0.0,
        shift_max: int = 0,
    ):
        """
        Args:
            pos_dir: Directory of positive .pt files (5, 60).
            neg_dir: Directory of negative .pt files (5, 60).
            noise_std: If > 0, add N(0, noise_std) to input (training augmentation).
            shift_max: If > 0, random circular shift in [-shift_max, shift_max] (training augmentation).
        """
        pos_dir = Path(pos_dir)
        neg_dir = Path(neg_dir)
        self.pos_files = sorted(pos_dir.glob("*.pt")) if pos_dir.exists() else []
        self.neg_files = sorted(neg_dir.glob("*.pt")) if neg_dir.exists() else []
        self.files: List[Path] = list(self.pos_files) + list(self.neg_files)
        self.labels = [1] * len(self.pos_files) + [0] * len(self.neg_files)
        self.noise_std = noise_std
        self.shift_max = shift_max

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = torch.load(self.files[idx], weights_only=True)
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        data = data.float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.noise_std > 0:
            data = data + torch.randn_like(data) * self.noise_std
            data = data.clamp(0.0, 1.0)
        if self.shift_max > 0:
            shift = int(torch.randint(-self.shift_max, self.shift_max + 1, (1,)).item())
            if shift != 0:
                data = torch.roll(data, shift, dims=-1)

        return data, label
