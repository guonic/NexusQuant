# -*- coding: utf-8 -*-
"""
Tangle Reversal Scanner (缠绕扭转器 - 缓跌中的揉搓见底).

Captures the pattern where MAs tangle together (convergence) after a slow decline,
then price stabilizes above all MAs with MAs starting to diverge upward.

This is the hardest pattern to capture: "无声的见底" (silent bottom) without violent
breakout candles. The essence is "均线缠绕后的多头扭转" (bullish reversal after MA tangle).

Logic:
1. Compute MA compactness: distance between short/mid-term MAs (MA5, MA10, MA20)
2. Tangle zone: price moves within MAs, MAs converge (is_tangled)
3. Reversal signal: price stabilizes above all MAs for N days, MAs start diverging upward

This captures "能量守恒" (energy conservation) - when costs converge, reversal is near.
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


def capture_tangle_reversal(
    df: pl.DataFrame,
    ts_code_col: str = "ts_code",
    ma_periods: Optional[List[int]] = None,
    compactness_threshold: float = 0.015,
    tangle_window: int = 5,
    consecutive_days: int = 2,
    require_yang_line: bool = True,
    require_ma_ordering: bool = True,
) -> pl.DataFrame:
    """
    Compute tangle reversal signal: MAs converge then price stabilizes above all MAs.

    Logic summary:
    1. Compute short/mid-term MAs (MA5, MA10, MA20)
    2. Compactness: (ma_max - ma_min) / ma_min < threshold (MAs tangle)
    3. had_tangle: had is_tangled in last N days
    4. above_all_ma: close > ma_max (price above all MAs)
    5. ma_ordering: ma5 > ma10 (MAs start diverging upward, optional)
    6. signal_tangle: had_tangle & above_all_ma for consecutive_days & yang line

    Expects columns: trade_date, open, high, low, close.
    If ts_code_col is present, all rolling/over is per stock.

    Args:
        df: Daily OHLC DataFrame.
        ts_code_col: Group column for multi-stock (e.g. "ts_code").
        ma_periods: MA periods list (default: [5, 10, 20]).
        compactness_threshold: Max relative distance for tangle (default: 0.015 = 1.5%).
        tangle_window: Lookback days for "had tangle" (default: 5).
        consecutive_days: Days price must stay above all MAs (default: 2).
        require_yang_line: If True, signal requires close > open (default True).
        require_ma_ordering: If True, require ma5 > ma10 (default True).

    Returns:
        DataFrame with added columns: ma5, ma10, ma20, ma_max, ma_min, is_tangled,
        had_tangle, above_all_ma, ma_ordering, signal_tangle.
    """
    over = [ts_code_col] if ts_code_col in df.columns else []

    def _over(expr: pl.Expr) -> pl.Expr:
        return expr.over(over) if over else expr

    if ma_periods is None:
        ma_periods = [5, 10, 20]

    # --- 1. Compute core MA cluster (focus on short/mid-term) ---
    ma_cols = []
    for p in ma_periods:
        col_name = f"ma{p}"
        df = df.with_columns(
            _over(pl.col("close").rolling_mean(window_size=p)).alias(col_name)
        )
        ma_cols.append(col_name)

    # --- 2. Compute MA cluster compactness (Compactness) ---
    # Calculate std or range of MA5, MA10, MA20. Smaller value = more tangled.
    df = df.with_columns([
        pl.max_horizontal([pl.col(col) for col in ma_cols]).alias("ma_max"),
        pl.min_horizontal([pl.col(col) for col in ma_cols]).alias("ma_min"),
    ])
    df = df.with_columns(
        (
            (pl.col("ma_max") - pl.col("ma_min")) / (pl.col("ma_min") + 1e-10) < compactness_threshold
        ).alias("is_tangled")
    )

    # --- 3. Detect "detachment after tangle" ---
    # We look for: the "tangle" moment (like 18.91), then price "stabilizes" above tangle lines
    df = df.with_columns([
        # Had tangle state in last N days
        _rolling_any(
            pl.col("is_tangled"),
            tangle_window,
            over,
        ).alias("had_tangle"),
        # Current price fully stabilized above MA cluster
        (pl.col("close") > pl.col("ma_max")).alias("above_all_ma"),
    ])

    # MA ordering: ma5 > ma10 (MAs start diverging upward, optional)
    if require_ma_ordering:
        df = df.with_columns(
            (pl.col("ma5") > pl.col("ma10")).alias("ma_ordering")
        )
    else:
        df = df.with_columns(pl.lit(True).alias("ma_ordering"))

    # --- 4. Capture signal ---
    # Condition: had tangle, today yang line, and this is consecutive_days-th day staying above
    # For consecutive_days: price must stay above_all_ma for consecutive N days
    # Signal triggers on the consecutive_days-th day (first time satisfying the condition)
    if consecutive_days == 1:
        # First day: above_all_ma True, previous False
        consecutive_above = (
            pl.col("above_all_ma")
            & (~_over(pl.col("above_all_ma").shift(1)).fill_null(False))
        )
    else:
        # Consecutive N days: above_all_ma for last N days (including today)
        # Check: today above_all_ma AND past (N-1) days all above_all_ma
        # And (N+1) days ago was NOT above (this is the first time satisfying consecutive N days)
        past_days_above = _over(
            pl.col("above_all_ma").shift(1).rolling_sum(consecutive_days - 1)
        ) >= (consecutive_days - 1)
        # Day before the consecutive period was NOT above
        prev_not_above = ~_over(pl.col("above_all_ma").shift(consecutive_days)).fill_null(False)
        consecutive_above = (
            pl.col("above_all_ma")
            & past_days_above
            & prev_not_above
        )

    # Yang line (optional)
    yang = pl.col("close") > pl.col("open") if require_yang_line else pl.lit(True)

    df = df.with_columns(
        (
            pl.col("had_tangle")
            & consecutive_above
            & pl.col("ma_ordering")
            & yang
        ).fill_null(False).alias("signal_tangle")
    )

    return df


class TangleReversalScanner:
    """
    Scanner that runs capture_tangle_reversal with configurable parameters.

    Use for batch scan: load daily data, call analyze(), then filter signal_tangle.
    """

    def __init__(
        self,
        ma_periods: Optional[List[int]] = None,
        compactness_threshold: float = 0.015,
        tangle_window: int = 5,
        consecutive_days: int = 2,
        require_yang_line: bool = True,
        require_ma_ordering: bool = True,
    ):
        """
        Initialize scanner.

        Args:
            ma_periods: MA periods list (default: [5, 10, 20]).
            compactness_threshold: Max relative distance for tangle (default: 0.015).
            tangle_window: Lookback days for "had tangle".
            consecutive_days: Days price must stay above all MAs.
            require_yang_line: If True, signal requires close > open.
            require_ma_ordering: If True, require ma5 > ma10.
        """
        self.ma_periods = ma_periods or [5, 10, 20]
        self.compactness_threshold = compactness_threshold
        self.tangle_window = tangle_window
        self.consecutive_days = consecutive_days
        self.require_yang_line = require_yang_line
        self.require_ma_ordering = require_ma_ordering

    def analyze(
        self,
        df: pl.DataFrame,
        ts_code_col: str = "ts_code",
    ) -> pl.DataFrame:
        """
        Run tangle reversal logic and add signal_tangle.

        Args:
            df: Daily OHLC DataFrame (columns: trade_date, open, high, low, close).
            ts_code_col: Group column for multi-stock.

        Returns:
            DataFrame with MA, tangle, and signal columns added.
        """
        return capture_tangle_reversal(
            df,
            ts_code_col=ts_code_col,
            ma_periods=self.ma_periods,
            compactness_threshold=self.compactness_threshold,
            tangle_window=self.tangle_window,
            consecutive_days=self.consecutive_days,
            require_yang_line=self.require_yang_line,
            require_ma_ordering=self.require_ma_ordering,
        )
