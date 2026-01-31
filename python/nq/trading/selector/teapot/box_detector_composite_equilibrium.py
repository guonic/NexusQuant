"""
Composite Equilibrium Box Detector for Teapot pattern recognition.

Multi-dimension serial filter (多级串联): only rows that satisfy ALL of
MA cohesion, quantile band width, price crossing MA10, and volume exhaustion
are marked as box candidates. No single indicator; composite logic only.
"""

import logging
from typing import Optional

import polars as pl

from nq.trading.selector.teapot.box_detector import BoxDetector

logger = logging.getLogger(__name__)


class CompositeEquilibriumDetector(BoxDetector):
    """
    Composite equilibrium detector (组合均衡检测器 / 终极均衡过滤器).

    Four dimensions, all required (AND):
    A. MA cohesion: MA5/10/20 tightly bound (ma_cohesion < threshold).
    B. Quantile band: (q80 - q20) / q20 < threshold (narrow core after removing spikes).
    C. Price penetration: price crosses MA10 at least N times in a window (equilibrium, not one-sided).
    D. Volume exhaustion: short-term avg volume < long-term avg * ratio (energy drying up).

    Rejects "fake big red box" (deep V + slow climb) and locks onto the true narrow equilibrium zone.
    """

    def __init__(
        self,
        box_window: int = 20,
        ma_cohesion_threshold: float = 0.015,
        quantile_width_threshold: float = 0.04,
        quantile_window: int = 20,
        cross_ma_period: int = 10,
        cross_count_min: int = 3,
        cross_window: int = 15,
        volume_short: int = 15,
        volume_long: int = 60,
        volume_ratio: float = 0.8,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize Composite Equilibrium Detector.

        Args:
            box_window: Default window for box (default: 20).
            ma_cohesion_threshold: Max std(MA5,10,20)/MA20 (default: 0.015, 1.5%).
            quantile_width_threshold: Max (q80-q20)/q20 (default: 0.04, 4%).
            quantile_window: Window for rolling quantiles (default: 20).
            cross_ma_period: MA period for cross check (default: 10, i.e. MA10).
            cross_count_min: Min number of crosses of MA in cross_window (default: 3).
            cross_window: Window to count crosses (default: 15).
            volume_short: Short volume mean window (default: 15).
            volume_long: Long volume mean window (default: 60).
            volume_ratio: Volume exhaustion: vol_short < vol_long * ratio (default: 0.8).
            smooth_window: Optional smoothing window (default: None).
            smooth_threshold: Optional smoothing threshold (default: None).
        """
        super().__init__(box_window=box_window, smooth_window=smooth_window, smooth_threshold=smooth_threshold)
        self.ma_cohesion_threshold = ma_cohesion_threshold
        self.quantile_width_threshold = quantile_width_threshold
        self.quantile_window = quantile_window
        self.cross_ma_period = cross_ma_period
        self.cross_count_min = cross_count_min
        self.cross_window = cross_window
        self.volume_short = volume_short
        self.volume_long = volume_long
        self.volume_ratio = volume_ratio

    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect boxes using composite logic: MA cohesion + quantile band + cross MA10 + volume exhaustion.

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low, volume.

        Returns:
            DataFrame with box_h, box_l, box_width, is_box_candidate and intermediate columns.
        """
        # 1. MA ribbon (均线维度)
        df = df.with_columns([
            pl.col("close").rolling_mean(window_size=5).over("ts_code").alias("ma5"),
            pl.col("close").rolling_mean(window_size=10).over("ts_code").alias("ma10"),
            pl.col("close").rolling_mean(window_size=20).over("ts_code").alias("ma20"),
        ])

        # 2. MA cohesion = horizontal std(ma5, ma10, ma20) / ma20
        ma_mean = (pl.col("ma5") + pl.col("ma10") + pl.col("ma20")) / 3
        ma_var = (
            (pl.col("ma5") - ma_mean).pow(2)
            + (pl.col("ma10") - ma_mean).pow(2)
            + (pl.col("ma20") - ma_mean).pow(2)
        ) / 3
        ma_std = ma_var.sqrt()
        df = df.with_columns([
            (ma_std / (pl.col("ma20") + 1e-10)).alias("ma_cohesion"),
        ])

        # 3. Space dimension: quantile band (箱体纯净度，剔除刺针)
        df = df.with_columns([
            pl.col("close")
            .rolling_quantile(quantile=0.8, window_size=self.quantile_window)
            .over("ts_code")
            .alias("q80"),
            pl.col("close")
            .rolling_quantile(quantile=0.2, window_size=self.quantile_window)
            .over("ts_code")
            .alias("q20"),
        ])
        df = df.with_columns(
            ((pl.col("q80") - pl.col("q20")) / (pl.col("q20") + 1e-10)).alias("quantile_width")
        )

        # 4. Price penetration: count crosses of MA10 in past cross_window days
        above_ma = (pl.col("close") > pl.col("ma10")).cast(pl.Int8)
        cross_count = (
            above_ma.diff().over("ts_code").abs()
            .rolling_sum(window_size=self.cross_window)
            .over("ts_code")
        )
        df = df.with_columns(cross_count.alias("ma_cross_count"))

        # 5. Volume exhaustion (need volume column)
        has_volume = "volume" in df.columns
        if has_volume:
            df = df.with_columns([
                pl.col("volume").rolling_mean(window_size=self.volume_short).over("ts_code").alias("vol_short"),
                pl.col("volume").rolling_mean(window_size=self.volume_long).over("ts_code").alias("vol_long"),
            ])
            volume_ok = pl.col("vol_short") < (pl.col("vol_long") * self.volume_ratio)
        else:
            volume_ok = pl.lit(True)

        # 6. Composite filter: all four conditions
        df = df.with_columns(
            (
                (pl.col("ma_cohesion") < self.ma_cohesion_threshold)
                & (pl.col("quantile_width") < self.quantile_width_threshold)
                & (pl.col("ma_cross_count") >= self.cross_count_min)
                & volume_ok
                & pl.col("ma20").is_not_null()
                & (pl.col("ma20") > 0)
                & pl.col("q20").is_not_null()
                & (pl.col("q20") > 0)
            ).alias("is_box_candidate")
        )

        # 7. Box bounds = quantile band (core range)
        df = df.with_columns([
            pl.col("q80").alias("box_h"),
            pl.col("q20").alias("box_l"),
            ((pl.col("q80") - pl.col("q20")) / (pl.col("q20") + 1e-10)).alias("box_width"),
        ])

        return self._apply_smoothing(df)
