# -*- coding: utf-8 -*-
"""
Vertical Piercing Scanner (多周期阻力簇垂直贯穿).

Captures the pattern where price breaks through multiple moving average resistance layers
in a single candle after being suppressed below all MAs (deep pit state).

Logic:
1. Build multi-period MA layers (MA5, MA10, MA20, MA30, MA60) as "resistance cloud"
2. Detect "deep pit": price below all MAs for N days (suppressed state)
3. Signal: A yang line that opens below cloud_top but closes above cloud_top,
   indicating a vertical piercing through all resistance layers.

This captures "动能反转" (momentum reversal) rather than "形态收复" (pattern recovery).
"""

import logging
from typing import List, Optional

import polars as pl

logger = logging.getLogger(__name__)


def _rolling_any(expr: pl.Expr, window: int, over: list) -> pl.Expr:
    """True if any value in the rolling window is truthy (Polars has no rolling_any)."""
    e = expr.cast(pl.Int32).rolling_sum(window)
    if over:
        e = e.over(over)
    return e >= 1


def capture_vertical_piercing(
    df: pl.DataFrame,
    ts_code_col: str = "ts_code",
    ma_periods: Optional[List[int]] = None,
    deep_pit_days: int = 5,
    require_yang_line: bool = True,
    require_volume_expansion: bool = False,
    volume_ratio_threshold: float = 1.5,
) -> pl.DataFrame:
    """
    Compute vertical piercing signal: price breaks through all MA resistance layers.

    Logic summary:
    1. Build multi-period MA layers (resistance cloud)
    2. cloud_top = max(MA5, MA10, MA20, MA30, MA60) - top of resistance
    3. cloud_bottom = min(MA5, MA10, MA20, MA30, MA60) - bottom of resistance
    4. is_under_cloud: close < cloud_bottom (deep pit state)
    5. signal_piercing: had is_under_cloud for N days, today close > cloud_top,
       open < cloud_top (breakout today), and optionally yang line + volume expansion.

    Expects columns: trade_date, open, high, low, close [, volume].
    If ts_code_col is present, all rolling/over is per stock.

    Args:
        df: Daily OHLC (and optionally volume) DataFrame.
        ts_code_col: Group column for multi-stock (e.g. "ts_code").
        ma_periods: MA periods list (default: [5, 10, 20, 30, 60]).
        deep_pit_days: Lookback days for "had been under cloud" (default: 5).
        require_yang_line: If True, signal requires close > open (default True).
        require_volume_expansion: If True, require volume > avg_volume * threshold (default False).
        volume_ratio_threshold: Volume expansion threshold (default: 1.5).

    Returns:
        DataFrame with added columns: ma5, ma10, ma20, ma30, ma60, cloud_top, cloud_bottom,
        is_under_cloud, signal_piercing.
    """
    over = [ts_code_col] if ts_code_col in df.columns else []

    def _over(expr: pl.Expr) -> pl.Expr:
        return expr.over(over) if over else expr

    if ma_periods is None:
        ma_periods = [5, 10, 20, 30, 60]

    # --- 1. Build multi-period resistance layers (Layers) ---
    ma_cols = []
    for p in ma_periods:
        col_name = f"ma{p}"
        df = df.with_columns(
            _over(pl.col("close").rolling_mean(window_size=p)).alias(col_name)
        )
        ma_cols.append(col_name)

    # --- 2. Cloud top and bottom: max/min of all MAs ---
    df = df.with_columns([
        pl.max_horizontal([pl.col(col) for col in ma_cols]).alias("cloud_top"),
        pl.min_horizontal([pl.col(col) for col in ma_cols]).alias("cloud_bottom"),
    ])

    # --- 3. Deep pit state: price below cloud_bottom ---
    df = df.with_columns(
        (pl.col("close") < pl.col("cloud_bottom")).alias("is_under_cloud")
    )

    # --- 4. Core signal: Vertical Piercing (The Piercing) ---
    # A. Had been under cloud for N days (deep pit)
    had_under_cloud = _rolling_any(
        _over(pl.col("is_under_cloud").shift(1)),
        deep_pit_days,
        over,
    )

    # B. Today close > cloud_top (pierces through entire cloud)
    above_cloud_top = pl.col("close") > pl.col("cloud_top")

    # C. Open < cloud_top (opens below resistance, breaks out today)
    open_below_cloud_top = pl.col("open") < pl.col("cloud_top")

    # D. Yang line (optional)
    yang = pl.col("close") > pl.col("open") if require_yang_line else pl.lit(True)

    # E. Volume expansion (optional)
    volume_ok = pl.lit(True)
    if require_volume_expansion and "volume" in df.columns:
        avg_volume = _over(pl.col("volume").rolling_mean(window_size=20))
        volume_ok = pl.col("volume") > (avg_volume * volume_ratio_threshold)

    df = df.with_columns(
        (
            had_under_cloud
            & above_cloud_top
            & open_below_cloud_top
            & yang
            & volume_ok
        ).fill_null(False).alias("signal_piercing")
    )

    return df


class VerticalPiercingScanner:
    """
    Scanner that runs capture_vertical_piercing with configurable parameters.

    Use for batch scan: load daily data, call analyze(), then filter signal_piercing.
    """

    def __init__(
        self,
        ma_periods: Optional[List[int]] = None,
        deep_pit_days: int = 5,
        require_yang_line: bool = True,
        require_volume_expansion: bool = False,
        volume_ratio_threshold: float = 1.5,
    ):
        """
        Initialize scanner.

        Args:
            ma_periods: MA periods list (default: [5, 10, 20, 30, 60]).
            deep_pit_days: Lookback days for "had been under cloud".
            require_yang_line: If True, signal requires close > open.
            require_volume_expansion: If True, require volume expansion.
            volume_ratio_threshold: Volume expansion threshold.
        """
        self.ma_periods = ma_periods or [5, 10, 20, 30, 60]
        self.deep_pit_days = deep_pit_days
        self.require_yang_line = require_yang_line
        self.require_volume_expansion = require_volume_expansion
        self.volume_ratio_threshold = volume_ratio_threshold

    def analyze(
        self,
        df: pl.DataFrame,
        ts_code_col: str = "ts_code",
    ) -> pl.DataFrame:
        """
        Run vertical piercing logic and add signal_piercing.

        Args:
            df: Daily OHLC DataFrame (columns: trade_date, open, high, low, close [, volume]).
            ts_code_col: Group column for multi-stock.

        Returns:
            DataFrame with MA, cloud, and signal columns added.
        """
        return capture_vertical_piercing(
            df,
            ts_code_col=ts_code_col,
            ma_periods=self.ma_periods,
            deep_pit_days=self.deep_pit_days,
            require_yang_line=self.require_yang_line,
            require_volume_expansion=self.require_volume_expansion,
            volume_ratio_threshold=self.volume_ratio_threshold,
        )
