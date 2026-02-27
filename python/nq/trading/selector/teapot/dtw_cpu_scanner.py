"""
CPU-side 5-channel DTW brute-force scanner.

Uses numpy/pandas only for matching logic. Local (rolling-window) normalization
so each window's 1.0 axis is the consolidation ceiling of that window.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default channel weights (O, H, L, C, V) for weighted Euclidean distance
DEFAULT_WEIGHTS = np.array([0.4, 0.4, 0.4, 1.0, 0.6], dtype=np.float64)


def load_template(template_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load gold template from .pt file; return numpy array and meta.

    Args:
        template_path: Path to .pt file (torch.save format).

    Returns:
        (template_np, meta): template_np shape (5, T), meta dict.
    """
    import torch

    try:
        pack = torch.load(template_path, map_location="cpu", weights_only=False)
    except TypeError:
        pack = torch.load(template_path, map_location="cpu")
    tensor = pack["tensor"]
    template_np = tensor.numpy().astype(np.float64)
    meta = pack.get("meta", {})
    if template_np.shape[0] != 5:
        raise ValueError(f"Expected 5 channels, got {template_np.shape[0]}")
    return template_np, meta


def normalize_window(
    window: pd.DataFrame,
    conso_len: int,
    vol_ma20: np.ndarray,
) -> np.ndarray:
    """
    Normalize one window with local anchor (first conso_len days = consolidation).

    Price anchor = max(close, high) over window[:conso_len].
    Volume = volume / vol_ma20 (same length as window).

    Args:
        window: DataFrame with columns open, high, low, close, volume; length T.
        conso_len: Number of days at start of window used as consolidation (anchor).
        vol_ma20: Array of length T (vol_ma20 for the window rows).

    Returns:
        Array shape (5, T): O, H, L, C, V normalized.
    """
    T = len(window)
    if conso_len <= 0 or conso_len > T:
        conso_len = min(20, T)

    conso = window.iloc[:conso_len]
    close_max = conso["close"].max()
    high_max = conso["high"].max()
    price_anchor = max(close_max, high_max)
    if price_anchor <= 0:
        price_anchor = 1.0

    o_norm = window["open"].values.astype(np.float64) / price_anchor
    h_norm = window["high"].values.astype(np.float64) / price_anchor
    l_norm = window["low"].values.astype(np.float64) / price_anchor
    c_norm = window["close"].values.astype(np.float64) / price_anchor

    vol_ma = np.asarray(vol_ma20, dtype=np.float64)
    vol_ma[vol_ma <= 0] = np.nan
    v_mean = np.nanmean(vol_ma) if np.any(np.isfinite(vol_ma)) else window["volume"].mean()
    v_norm = window["volume"].values.astype(np.float64) / np.where(
        np.isnan(vol_ma), v_mean, vol_ma
    )
    v_norm = np.nan_to_num(v_norm, nan=1.0, posinf=1.0, neginf=0.0)

    return np.stack([o_norm, h_norm, l_norm, c_norm, v_norm])


def has_trap_structure(
    sample_slice: np.ndarray,
    trap_depth_ratio: float = 0.97,
) -> bool:
    """
    Require normalized close to have a dip in the middle (conso → trap → breakout).

    Checks: min(close) in middle segment <= trap_depth_ratio * mean(close) in first and last segments.
    """
    close_norm = sample_slice[3, :]
    T = close_norm.size
    if T < 9:
        return True
    q = T // 4
    first_mean = float(np.mean(close_norm[:q]))
    last_mean = float(np.mean(close_norm[-q:]))
    mid_min = float(np.min(close_norm[q : T - q]))
    if first_mean <= 0 or last_mean <= 0:
        return False
    return (
        mid_min <= first_mean * trap_depth_ratio
        and mid_min <= last_mean * trap_depth_ratio
    )


def _is_top_pattern(
    sample_slice: np.ndarray,
    conso_len: int,
    start_flat_max_rise: float = 0.05,
    last_rising_min_ratio: float = 1.0,
) -> bool:
    """
    Return True if this window looks like a TOP (rise then fall) and should be rejected.

    We want bottom pattern: flat start → dip → breakout. So we reject when:
    1) First segment is strongly rising (not flat conso).
    2) Last segment is declining (peak already passed).
    """
    close_norm = sample_slice[3, :]
    T = close_norm.size
    if T < conso_len + 6:
        return False

    # 1) First conso_len days should not ramp up too much (flat conso, not "rise to peak")
    start_rise = float(close_norm[conso_len - 1] - close_norm[0])
    if start_rise > start_flat_max_rise:
        return True  # reject: strong rise at start = top pattern

    # 2) Last 3 days should not be lower than previous 3 (recovery, not "past peak falling")
    if T >= 6:
        last3 = float(np.mean(close_norm[-3:]))
        prev3 = float(np.mean(close_norm[-6:-3]))
        if prev3 > 1e-9 and last3 / prev3 < last_rising_min_ratio:
            return True  # reject: last part declining
    return False


def get_similarity_score(
    sample_slice: np.ndarray,
    template_tensor: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Weighted Euclidean distance (RMSE of weighted diff).

    Dist = sqrt( mean( (weights * (sample - template))^2 ) ).

    Args:
        sample_slice: (5, T).
        template_tensor: (5, T).
        weights: (5,) or (5,1).

    Returns:
        Non-negative distance; lower = more similar.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = w[:, np.newaxis]
    diff = (sample_slice - template_tensor) * w
    return float(np.sqrt(np.mean(diff ** 2)))


def brute_force_cpu_scanner(
    stock_df: pd.DataFrame,
    template_path: str,
    conso_len: int = 20,
    threshold: float = 0.18,
    breakout_min: float = 1.0,
    require_trap_structure: bool = True,
    trap_depth_ratio: float = 0.97,
    reject_top_pattern: bool = True,
    start_flat_max_rise: float = 0.05,
    last_rising_min_ratio: float = 1.0,
    weights: Optional[np.ndarray] = None,
    date_col: str = "date",
    ts_code_col: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Scan one stock's full series with sliding window; return hits below threshold.

    Each window is normalized locally (anchor = max(close, high) of first conso_len
    days). Template is loaded once from .pt (5 channels, length T).

    Args:
        stock_df: Must have date_col, open, high, low, close, volume, vol_ma20.
        template_path: Path to gold template .pt.
        conso_len: Days at start of each window used as consolidation for anchor.
        threshold: Max distance to count as hit (e.g. 0.15--0.18).
        breakout_min: Only accept window if last-day close (normalized) >= this (default 1.0).
        require_trap_structure: If True, require a dip in the middle (conso→trap→breakout).
        trap_depth_ratio: Middle min(close) must be <= this ratio of first/last segment mean (default 0.97).
        reject_top_pattern: If True, reject "rise then fall" (top) windows: flat start + last part rising.
        start_flat_max_rise: Max rise in first conso segment (default 0.05); reject if start ramps up more.
        last_rising_min_ratio: Require mean(last 3) / mean(prev 3) >= this (default 1.0); reject if last part declining.
        weights: (5,) channel weights; default from DEFAULT_WEIGHTS.
        date_col: Date column name.
        ts_code_col: Optional symbol column (e.g. ts_code) for result metadata.

    Returns:
        List of dicts: start_index, end_index, start_date, end_date, score,
        [ts_code], optional data_slice (5, T) if you need to save training samples.
    """
    template_np, meta = load_template(template_path)
    T = template_np.shape[1]
    weights = weights if weights is not None else DEFAULT_WEIGHTS
    weights = np.asarray(weights, dtype=np.float64)

    df = stock_df.sort_values(date_col).reset_index(drop=True)
    if "vol_ma20" not in df.columns:
        df["vol_ma20"] = df["volume"].rolling(window=20).mean()

    n = len(df)
    if n < T:
        logger.warning("Stock series length %d < template length %d", n, T)
        return []

    ticker = None
    if ts_code_col and ts_code_col in df.columns:
        ticker = str(df[ts_code_col].iloc[0])

    dates = pd.to_datetime(df[date_col])
    vol_ma20 = df["vol_ma20"].values

    found: List[Dict[str, Any]] = []
    for t in range(n - T + 1):
        window = df.iloc[t : t + T]
        vol_slice = vol_ma20[t : t + T]
        sample_slice = normalize_window(window, conso_len, vol_slice)

        score = get_similarity_score(sample_slice, template_np, weights)

        # Only accept if window ends with breakout (close >= conso ceiling), not drop
        last_close_norm = float(sample_slice[3, -1])
        if last_close_norm < breakout_min:
            continue

        # Require conso → trap (dip) → breakout structure, not "flat then direct up"
        if require_trap_structure and not has_trap_structure(sample_slice, trap_depth_ratio):
            continue

        # Reject top pattern: strong rise at start or last part declining (peak already passed)
        if reject_top_pattern and _is_top_pattern(
            sample_slice, conso_len, start_flat_max_rise, last_rising_min_ratio
        ):
            continue

        if score < threshold:
            start_date = dates.iloc[t]
            end_date = dates.iloc[t + T - 1]
            hit = {
                "start_index": int(t),
                "end_index": int(t + T - 1),
                "start_date": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
                "end_date": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
                "score": float(score),
            }
            if ticker is not None:
                hit["ts_code"] = ticker
            hit["data_slice"] = sample_slice.astype(np.float32)
            found.append(hit)

    found.sort(key=lambda h: h["score"])
    return found
