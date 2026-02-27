# -*- coding: utf-8 -*-
"""
Pure-price DRU (Down-Relay-Up) state and signal logic.

No moving averages: states are defined only by K-line high/low evolution and
spatial overlap (重心移动、包含关系、空间重合). Zero-lag, price-first topology.

- D (Down): Low < prev low, High < prev high (重心下移).
- U (Up): High > prev high, Low > prev low (重心上移).
- R (Relay): Neither; or spatial_overlap (rolling N-day shared price space).
- Signal 4: prev not up, prev spatial_overlap, is_turn_up (拐点), close > relay ceiling.
"""

import logging
import polars as pl

logger = logging.getLogger(__name__)


def analyze_pure_price_logic(
    df: pl.DataFrame,
    ts_code_col: str = "ts_code",
    rolling_n: int = 5,
    turn_lookback: int = 3,
) -> pl.DataFrame:
    """
    Pure-price DRU state and signal_4 without any moving averages.

    Expects columns: high, low, close (and optionally trade_date, open, volume).
    If ts_code_col exists, all rolling/over is per stock.

    Args:
        df: OHLC DataFrame.
        ts_code_col: Column name for stock id (optional).
        rolling_n: Window for spatial overlap (rolling_h_min / rolling_l_max).
        turn_lookback: Lookback for is_turn_up (low vs rolling_min(low)).

    Returns:
        DataFrame with:
        - higher_h, higher_l, lower_h, lower_l (vs prev bar)
        - is_up, is_down, is_relay, p_state (1=U, -1=D, 0=R)
        - rolling_h_min, rolling_l_max, spatial_overlap
        - is_turn_up (拐点向上)
        - signal_4 (D/R -> turn_up break above relay ceiling)
        - relay_box_high, relay_box_low (optional recursive relay box)
    """
    if len(df) < max(rolling_n, turn_lookback) + 1:
        return df

    over = [ts_code_col] if ts_code_col in df.columns else []

    def roll(expr: pl.Expr) -> pl.Expr:
        return expr.over(over) if over else expr

    # --- 1. 基础空间演进（相对前一根 K 线）---
    df = df.with_columns([
        (pl.col("high") > pl.col("high").shift(1)).alias("higher_h"),
        (pl.col("low") > pl.col("low").shift(1)).alias("higher_l"),
        (pl.col("high") < pl.col("high").shift(1)).alias("lower_h"),
        (pl.col("low") < pl.col("low").shift(1)).alias("lower_l"),
    ])

    # --- 2. 原子状态（纯价格重心）---
    is_up = pl.col("higher_h") & pl.col("higher_l")
    is_down = pl.col("lower_h") & pl.col("lower_l")
    is_relay = (~is_up) & (~is_down)

    df = df.with_columns([
        is_up.alias("is_up"),
        is_down.alias("is_down"),
        is_relay.alias("is_relay"),
        pl.when(is_up).then(pl.lit(1))
        .when(is_down).then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias("p_state"),
    ])

    # --- 3. 空间重合（大级别中继）---
    # 连续 rolling_n 天内：最低的高点 > 最高的低点 → 存在共同价格区间
    df = df.with_columns([
        roll(pl.col("high").rolling_min(rolling_n)).alias("rolling_h_min"),
        roll(pl.col("low").rolling_max(rolling_n)).alias("rolling_l_max"),
    ])
    df = df.with_columns(
        (pl.col("rolling_h_min") > pl.col("rolling_l_max")).alias("spatial_overlap")
    )

    # --- 4. 拐点先行（基于高低点，非均线）---
    # 低点不再创新低 且 收盘站上前一日最高 = 拐点向上
    prev_high = pl.col("high").shift(1)
    rolling_low_min_prev = roll(pl.col("low").rolling_min(turn_lookback)).shift(1)
    is_turn_up = (
        (pl.col("low") >= rolling_low_min_prev)
        & (pl.col("close") > prev_high)
    )
    df = df.with_columns(is_turn_up.alias("is_turn_up"))

    # --- 5. Signal 4：之前非上升、近期有空间重合、今日拐点向上且收盘穿透中继区间顶部 ---
    prev_state_not_up = pl.col("p_state").shift(1) <= 0
    prev_overlap = pl.col("spatial_overlap").shift(1)
    close_above_relay_ceiling = pl.col("close") > roll(pl.col("rolling_h_min").shift(1))

    signal_4 = (
        prev_state_not_up
        & prev_overlap.fill_null(False)
        & pl.col("is_turn_up")
        & close_above_relay_ceiling
    )
    df = df.with_columns(signal_4.alias("signal_4"))

    # --- 6. 递归中继箱体（空间重合区边界）---
    # 当 spatial_overlap 成立时记录 [rolling_l_max, rolling_h_min]，向前填充
    # 进阶可做：击穿箱体时置 null 再 forward_fill，由下一段 spatial_overlap 重新定界
    df = df.with_columns([
        roll(pl.when(pl.col("spatial_overlap")).then(pl.col("rolling_h_min")).otherwise(None).forward_fill()).alias("relay_box_high"),
        roll(pl.when(pl.col("spatial_overlap")).then(pl.col("rolling_l_max")).otherwise(None).forward_fill()).alias("relay_box_low"),
    ])

    return df


class PurePriceDRUScanner:
    """
    Pure-price DRU scanner: D/R/U states and signal_4 from high/low evolution only.

    No moving averages. States: 1=Up, -1=Down, 0=Relay.
    Signal 4: turn-up (拐点) break above relay space after spatial_overlap.
    """

    def __init__(
        self,
        rolling_n: int = 5,
        turn_lookback: int = 3,
    ):
        self.rolling_n = rolling_n
        self.turn_lookback = turn_lookback

    def analyze(
        self,
        df: pl.DataFrame,
        ts_code_col: str = "ts_code",
    ) -> pl.DataFrame:
        """
        Run pure-price DRU logic.

        Expects columns: high, low, close. Optional: trade_date, ts_code.

        Returns:
            DataFrame with p_state, spatial_overlap, is_turn_up, signal_4,
            relay_box_high, relay_box_low, and intermediate columns.
        """
        return analyze_pure_price_logic(
            df,
            ts_code_col=ts_code_col,
            rolling_n=self.rolling_n,
            turn_lookback=self.turn_lookback,
        )
