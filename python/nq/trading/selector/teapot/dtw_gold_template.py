"""
DTW gold template extraction with 5-channel spatial-temporal normalization.

Five channels: Open, High, Low, Close, Volume.
Price channels anchored by consolidation ceiling (1.0); volume by vol_ma20 (activity).
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Suggested channel weights for DTW Euclidean distance on 4090:
# Dist = sqrt( w_O*dO^2 + w_H*dH^2 + w_L*dL^2 + w_C*dC^2 + w_V*dV^2 )
# Close most important; O/H/L for shape; Volume for energy.
DTW_CHANNEL_WEIGHTS: List[float] = [0.4, 0.4, 0.4, 1.0, 0.6]  # O, H, L, C, V
CHANNEL_NAMES: List[str] = ["Open", "High", "Low", "Close", "Volume"]


def generate_pytorch_template(
    df: pd.DataFrame,
    total_range: Tuple[str, str],
    consolidation_range: Tuple[str, str],
    save_path: str,
    date_col: str = "date",
    use_float16: bool = False,
) -> Dict[str, Any]:
    """
    Generate 5-channel gold template (Open, High, Low, Close, Volume).

    Spatial-temporal separation:
    - Price channels (O,H,L,C): normalized by consolidation ceiling max(close, high) = 1.0.
    - Volume channel: normalized by 20-day average volume (relative activity).

    Args:
        df: DataFrame with [date, open, high, low, close, volume]. Optional vol_ma20;
            if missing, computed as volume.rolling(20).mean(). Caller must load at least 20
            days before total_range[0] so vol_ma20 is valid in the template window.
        total_range: (start, end) YYYY-MM-DD for full template (conso + trap + reversal).
        consolidation_range: (start, end) YYYY-MM-DD for consolidation zone (1.0 anchor).
        save_path: Path to save .pt file.
        date_col: Name of date column (default 'date').
        use_float16: If True, save tensor as float16 to save memory for 4090 batch DTW.

    Returns:
        Dict with 'tensor' [5, T], 'meta' (price_anchor, channels, final_c_norm, max_h_norm, avg_v_norm, channel_weights).

    Raises:
        ValueError: If date ranges match no data.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    if "vol_ma20" not in df.columns:
        df["vol_ma20"] = df["volume"].rolling(window=20).mean()

    full_mask = (df[date_col] >= total_range[0]) & (df[date_col] <= total_range[1])
    conso_mask = (
        (df[date_col] >= consolidation_range[0])
        & (df[date_col] <= consolidation_range[1])
    )

    full_seq = df.loc[full_mask].copy()
    conso_seq = df.loc[conso_mask].copy()

    if len(full_seq) == 0 or len(conso_seq) == 0:
        raise ValueError(
            f"Date range matched no data: total_range={total_range}, consolidation_range={consolidation_range}"
        )

    # Price anchor: consolidation zone ceiling = max(close, high) (1.0 reference)
    close_max = float(conso_seq["close"].max())
    high_max = float(conso_seq["high"].max())
    price_anchor = max(close_max, high_max)

    # 5-channel normalization
    o_norm = full_seq["open"].values.astype(np.float64) / price_anchor
    h_norm = full_seq["high"].values.astype(np.float64) / price_anchor
    l_norm = full_seq["low"].values.astype(np.float64) / price_anchor
    c_norm = full_seq["close"].values.astype(np.float64) / price_anchor

    vol_ma = full_seq["vol_ma20"].fillna(full_seq["volume"].mean()).values.astype(
        np.float64
    )
    vol_ma[vol_ma <= 0] = np.nan
    v_norm = full_seq["volume"].values.astype(np.float64) / np.where(
        np.isnan(vol_ma), full_seq["volume"].mean(), vol_ma
    )
    v_norm = np.nan_to_num(v_norm, nan=1.0, posinf=1.0, neginf=0.0)

    template_tensor = torch.tensor(
        np.stack([o_norm, h_norm, l_norm, c_norm, v_norm]),
        dtype=torch.float16 if use_float16 else torch.float32,
    )

    final_c_norm = float(c_norm[-1])
    max_h_norm = float(np.max(h_norm))
    avg_v_norm = float(np.mean(v_norm))

    save_data = {
        "tensor": template_tensor,
        "meta": {
            "price_anchor": price_anchor,
            "channels": CHANNEL_NAMES,
            "channel_weights": DTW_CHANNEL_WEIGHTS,
            "final_c_norm": final_c_norm,
            "max_h_norm": max_h_norm,
            "avg_v_norm": avg_v_norm,
            "start_date": total_range[0],
            "end_date": total_range[1],
            "conso_start": consolidation_range[0],
            "conso_end": consolidation_range[1],
        },
    }

    torch.save(save_data, save_path)
    logger.info(
        "5ch template saved: anchor=%.4f, final_c_norm=%.4f, max_h_norm=%.4f, avg_v_norm=%.2f, shape=%s, path=%s",
        price_anchor,
        final_c_norm,
        max_h_norm,
        avg_v_norm,
        tuple(template_tensor.shape),
        save_path,
    )

    return save_data
