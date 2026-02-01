# -*- coding: utf-8 -*-
"""
DTW-based Golden Pattern Matcher (黄金形态模版库).

Uses Dynamic Time Warping to measure similarity between a target 60-day close series
and a library of golden templates. Invariant to when the "step" occurs (e.g. day 10 vs 12).
"""

import logging
from pathlib import Path
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    _FASTDTW_AVAILABLE = True
except ImportError:
    _FASTDTW_AVAILABLE = False
    fastdtw = None
    euclidean = None


def _load_template(path: Union[str, Path]) -> np.ndarray:
    """Load one template: .pt (5, 60) -> close channel index 3; .npy (60,) or (5, 60) -> close."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".pt":
        try:
            import torch
            data = torch.load(path, map_location="cpu", weights_only=True)
            if hasattr(data, "numpy"):
                data = data.numpy()
            else:
                data = np.asarray(data)
        except Exception:
            data = np.load(path, allow_pickle=True)
        if data.ndim == 2 and data.shape[0] >= 4:
            return np.asarray(data[3, :], dtype=np.float64).flatten()
        if data.ndim == 1:
            return np.asarray(data, dtype=np.float64).flatten()
        raise ValueError(f"Unexpected .pt shape: {data.shape}")
    if suffix == ".npy":
        data = np.load(path, allow_pickle=True)
        if data.ndim == 2 and data.shape[1] >= 4:
            return np.asarray(data[:, 3], dtype=np.float64).flatten()
        if data.ndim == 2 and data.shape[0] >= 4:
            return np.asarray(data[3, :], dtype=np.float64).flatten()
        if data.ndim == 1:
            return np.asarray(data, dtype=np.float64).flatten()
        raise ValueError(f"Unexpected .npy shape: {data.shape}")
    raise ValueError(f"Unsupported template file: {path}")


class PatternMatcher:
    """
    Golden-pattern similarity engine using DTW (fastdtw).

    Lower score = more similar to templates. Returns mean of top-k smallest DTW distances.
    """

    def __init__(
        self,
        golden_filenames: List[Union[str, Path]],
        top_k: int = 3,
    ):
        """
        Args:
            golden_filenames: Paths to .pt or .npy template files (60-day close or (5,60)).
            top_k: Number of nearest templates to average for score (default: 3).
        """
        if not _FASTDTW_AVAILABLE:
            raise ImportError("fastdtw and scipy are required: pip install fastdtw scipy")
        self.templates: List[np.ndarray] = []
        for f in golden_filenames:
            try:
                t = _load_template(f)
                if len(t) != 60:
                    logger.warning("Template %s length %d != 60, skip", f, len(t))
                    continue
                self.templates.append(t.astype(np.float64))
            except Exception as e:
                logger.warning("Failed to load template %s: %s", f, e)
        self.top_k = min(top_k, len(self.templates)) if self.templates else 0
        logger.info("Loaded %d golden templates (top_k=%d).", len(self.templates), self.top_k)

    def get_score(self, target_series: np.ndarray) -> float:
        """
        Compute similarity score = mean of top-k smallest DTW distances to templates.
        Lower score = more similar. Target must be 1D length 60 (normalized close).
        """
        target_series = np.asarray(target_series, dtype=np.float64).flatten()
        if len(target_series) != 60:
            return np.inf
        if not self.templates:
            return np.inf
        distances: List[float] = []
        x = target_series.reshape(-1, 1)
        for temp in self.templates:
            y = temp.reshape(-1, 1)
            distance, _ = fastdtw(x, y, dist=euclidean)
            distances.append(float(distance))
        k = self.top_k
        return float(np.mean(sorted(distances)[:k]))
