# -*- coding: utf-8 -*-
"""
Fractal Box Scanner (分型转折自适应动态箱体).

Based on "fractal boundary" logic: when trend direction clearly changes, the old
Block is closed. Only "congestion" (overlap/churn) days are merged into a block;
smooth step-down or step-up days force a new block. This avoids merging a whole
downtrend into one giant platform.

Signal: "break out of the hood" (破壳而出) — price had broken below a previous
congestion zone floor, then closes above that zone's ceiling (with optional yang line).

Ref: docs/wip/selector design — 基于分型转折的自适应动态箱体.
"""

import logging
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


def _rolling_any(expr: pl.Expr, window: int, over: list) -> pl.Expr:
    """True if any value in the rolling window is truthy (Polars has no rolling_any)."""
    e = expr.cast(pl.Int32).rolling_sum(window)
    if over:
        e = e.over(over)
    return e >= 1


def capture_topological_final(
    df: pl.DataFrame,
    ts_code_col: str = "ts_code",
    had_below_floor_window: int = 10,
    require_yang_line: bool = True,
) -> pl.DataFrame:
    """
    Compute fractal-boundary blocks and "break out of hood" signal (破壳而出).

    Logic summary:
    1. Directional break: h_step_down/l_step_down (step down), h_step_up/l_step_up (step up).
    2. is_congestion: NOT (smooth step down) AND NOT (smooth step up) → overlap/churn.
    3. block_id: increments on each trending day (is_congestion=False), so blocks are
       runs of congestion separated by trending days.
    4. b_ceiling, b_floor, b_len: per-block high/low/count.
    5. the_hood: when leaving a multi-day congestion block, remember its (ceiling, floor);
       forward_fill so we know "the resistance hood we had drilled below".
    6. signal_pure_topology: had close < the_hood.f in last N days, today close > the_hood.c,
       prev close <= the_hood.c (breakout today), and optionally close > open (yang line).

    Expects columns: trade_date, open, high, low, close, volume (volume optional for signal).
    If ts_code_col is present, all rolling/over is per stock.

    Args:
        df: Daily OHLC (and optionally volume) DataFrame.
        ts_code_col: Group column for multi-stock (e.g. "ts_code").
        had_below_floor_window: Lookback days for "had broken below hood floor" (default 10).
        require_yang_line: If True, signal requires close > open (default True).

    Returns:
        DataFrame with added columns: h_step_down, l_step_down, h_step_up, l_step_up,
        is_congestion, block_id, b_ceiling, b_floor, b_len, the_hood (struct c,f),
        signal_pure_topology.
    """
    over = [ts_code_col] if ts_code_col in df.columns else []

    def _over(expr: pl.Expr) -> pl.Expr:
        return expr.over(over) if over else expr

    # --- 1. Directional break ---
    df = df.with_columns([
        _over(pl.col("high") < pl.col("high").shift(1)).alias("h_step_down"),
        _over(pl.col("low") < pl.col("low").shift(1)).alias("l_step_down"),
        _over(pl.col("high") > pl.col("high").shift(1)).alias("h_step_up"),
        _over(pl.col("low") > pl.col("low").shift(1)).alias("l_step_up"),
    ])

    # Congestion = neither smooth step-down nor smooth step-up
    df = df.with_columns(
        (
            ((pl.col("h_step_down") & pl.col("l_step_down")) == False)
            & ((pl.col("h_step_up") & pl.col("l_step_up")) == False)
        ).alias("is_congestion")
    )

    # --- 2. Block ID: increment on each trending day ---
    df = df.with_columns(
        _over((pl.col("is_congestion") == False).cast(pl.Int32).cum_sum()).alias("block_id")
    )

    # --- 3. Block ceiling, floor, length ---
    block_over = [*over, "block_id"]
    df = df.with_columns([
        pl.col("high").max().over(block_over).alias("b_ceiling"),
        pl.col("low").min().over(block_over).alias("b_floor"),
        pl.col("block_id").count().over(block_over).alias("b_len"),
    ])

    # --- 4. the_hood: previous multi-day congestion zone when we "drill down" ---
    df = df.with_columns(
        _over(
            pl.when(
                (_over(pl.col("block_id").diff(1)) != 0) & (_over(pl.col("b_len").shift(1)) >= 2)
            )
            .then(
                pl.struct([
                    _over(pl.col("b_ceiling").shift(1)).alias("c"),
                    _over(pl.col("b_floor").shift(1)).alias("f"),
                ])
            )
            .otherwise(None)
            .forward_fill()
        ).alias("the_hood")
    )

    # --- 5. Signal: break out of the hood ---
    hood_ok = pl.col("the_hood").is_not_null()
    # A. Had close < the_hood.f in last N days
    had_below = _rolling_any(
        pl.col("close") < pl.col("the_hood").struct.field("f"),
        had_below_floor_window,
        over,
    )
    # B. Today close > the_hood.c
    above_ceiling = pl.col("close") > pl.col("the_hood").struct.field("c")
    # C. Previous close <= the_hood.c (breakout today)
    prev_below_ceiling = _over(pl.col("close").shift(1)) <= pl.col("the_hood").struct.field("c")
    # D. Optional: yang line
    yang = pl.col("close") > pl.col("open") if require_yang_line else pl.lit(True)

    df = df.with_columns(
        (
            hood_ok
            & had_below
            & above_ceiling
            & prev_below_ceiling
            & yang
        ).fill_null(False).alias("signal_pure_topology")
    )

    return df


class FractalBoxScanner:
    """
    Scanner that runs capture_topological_final with configurable parameters.

    Use for batch scan: load daily data, call analyze(), then filter signal_pure_topology.
    """

    def __init__(
        self,
        had_below_floor_window: int = 10,
        require_yang_line: bool = True,
    ):
        """
        Initialize scanner.

        Args:
            had_below_floor_window: Lookback days for "had broken below hood floor".
            require_yang_line: If True, signal requires close > open (yang line).
        """
        self.had_below_floor_window = had_below_floor_window
        self.require_yang_line = require_yang_line

    def analyze(
        self,
        df: pl.DataFrame,
        ts_code_col: str = "ts_code",
    ) -> pl.DataFrame:
        """
        Run fractal box logic and add signal_pure_topology.

        Args:
            df: Daily OHLC DataFrame (columns: trade_date, open, high, low, close [, volume]).
            ts_code_col: Group column for multi-stock.

        Returns:
            DataFrame with block and signal columns added.
        """
        return capture_topological_final(
            df,
            ts_code_col=ts_code_col,
            had_below_floor_window=self.had_below_floor_window,
            require_yang_line=self.require_yang_line,
        )
