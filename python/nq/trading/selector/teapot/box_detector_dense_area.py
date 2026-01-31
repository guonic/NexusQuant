"""
Dense Area Box Detector for Teapot pattern recognition.

Detects boxes by price distribution density: finds the narrow band where price
stays most of the time, ignoring outlier wicks that stretch H/L. Suited for
"core筹码均衡、上方允许突刺" type consolidation.
"""

import logging
from typing import Optional

import polars as pl

from nq.trading.selector.teapot.box_detector import BoxDetector

logger = logging.getLogger(__name__)


class DenseAreaBoxDetector(BoxDetector):
    """
    Dense-area box detector (筹码密集区检测器).

    Uses quantiles of close price (not H/L) to define the "core band" where
    price spends most of its time. Filters out spikes that would stretch
    box boundaries, so boxes can be offset (e.g. dense area in lower half)
    while still being a stable platform.

    Conditions:
    - Dense band is narrow (dense_width < dense_threshold).
    - Volatility is low (vol_stability < vol_stability_threshold).
    - Mid-line is flat over a short horizon (mid_slope check).
    """

    def __init__(
        self,
        box_window: int = 40,
        dense_threshold: float = 0.05,
        quantile_high: float = 0.8,
        quantile_low: float = 0.2,
        vol_stability_threshold: float = 0.03,
        mid_slope_threshold: float = 0.02,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize Dense Area Box Detector.

        Args:
            box_window: Window size for rolling stats (default: 40).
            dense_threshold: Max relative width of dense band (default: 0.05, 5%).
                Stricter (e.g. 0.03–0.04) for very tight platforms.
            quantile_high: Upper quantile for dense band (default: 0.8).
            quantile_low: Lower quantile for dense band (default: 0.2).
                Together 0.2–0.8 covers 60% of price distribution.
            vol_stability_threshold: Max rolling std / mid_line (default: 0.03).
            mid_slope_threshold: Max relative change of mid_line over 10 days
                (default: 0.02).
            smooth_window: Box filter smoothing window (default: None).
            smooth_threshold: Smoothing threshold (default: None).
        """
        super().__init__(box_window, smooth_window, smooth_threshold)
        self.dense_threshold = dense_threshold
        self.quantile_high = quantile_high
        self.quantile_low = quantile_low
        self.vol_stability_threshold = vol_stability_threshold
        self.mid_slope_threshold = mid_slope_threshold

    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect boxes using price distribution density (quantile-based core band).

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low.

        Returns:
            DataFrame with box_h, box_l, box_width, is_box_candidate and
            dense_h, dense_l, dense_width, vol_stability, mid_line.
        """
        # 1. Core band from close quantiles (not H/L) to ignore wicks
        df = df.with_columns([
            pl.col("close")
            .rolling_quantile(
                quantile=self.quantile_high,
                window_size=self.box_window,
            )
            .over("ts_code")
            .alias("dense_h"),
            pl.col("close")
            .rolling_quantile(
                quantile=self.quantile_low,
                window_size=self.box_window,
            )
            .over("ts_code")
            .alias("dense_l"),
            pl.col("close")
            .rolling_mean(window_size=self.box_window)
            .over("ts_code")
            .alias("mid_line"),
        ])

        # 2. Dense band width and volatility stability
        df = df.with_columns([
            ((pl.col("dense_h") - pl.col("dense_l")) / (pl.col("dense_l") + 1e-10)).alias("dense_width"),
            (
                pl.col("close")
                .rolling_std(window_size=self.box_window)
                .over("ts_code")
                / (pl.col("mid_line") + 1e-10)
            ).alias("vol_stability"),
        ])

        # 3. Mid-line slope (flat platform)
        mid_shifted = pl.col("mid_line").shift(10).over("ts_code")
        df = df.with_columns([
            ((pl.col("mid_line") - mid_shifted).abs() / (pl.col("mid_line") + 1e-10)).alias("mid_slope_10"),
        ])

        # 4. Box candidate: narrow dense band, low vol, flat mid-line
        df = df.with_columns(
            (
                (pl.col("dense_width") < self.dense_threshold)
                & (pl.col("vol_stability") < self.vol_stability_threshold)
                & pl.col("mid_slope_10").is_not_null()
                & (pl.col("mid_slope_10") < self.mid_slope_threshold)
                & pl.col("dense_h").is_not_null()
                & pl.col("dense_l").is_not_null()
                & (pl.col("dense_h") > pl.col("dense_l"))
            ).alias("is_box_candidate")
        )

        # 5. Output box bounds = dense band (for charts and compatibility)
        df = df.with_columns([
            pl.col("dense_h").alias("box_h"),
            pl.col("dense_l").alias("box_l"),
            ((pl.col("dense_h") - pl.col("dense_l")) / (pl.col("dense_l") + 1e-10)).alias("box_width"),
        ])

        return self._apply_smoothing(df)
