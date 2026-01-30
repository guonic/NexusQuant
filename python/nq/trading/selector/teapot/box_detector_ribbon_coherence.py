"""
Ribbon Coherence Box Detector for Teapot pattern recognition.

Identifies "true platform" zones where MA5, MA10, MA20 are intertwined and
price oscillates slightly around them — filtering out wide Max/Min boxes
that span deep V or slow climb (no real equilibrium).
"""

import logging
from typing import Optional

import polars as pl

from nq.trading.selector.teapot.box_detector import BoxDetector

logger = logging.getLogger(__name__)


class RibbonCoherenceDetector(BoxDetector):
    """
    Ribbon coherence detector (均线相干性/集束度检测器).

    Detects boxes by "MA ribbon coherence": MA5, MA10, MA20 must stay close
    (convergence), MA20 must be flat (no trend), and price must stay close
    to the ribbon. Rejects zones where the center of gravity is moving
    (e.g. deep V + slow climb) even if H/L look like a box.

    Conditions:
    - ma_cv: horizontal std of [MA5, MA10, MA20] / MA20 < convergence_threshold.
    - ma_slope: |MA20 - MA20_5d_ago| / MA20_5d_ago < ma_slope_threshold.
    - price_dev: |close - MA20| / MA20 < price_to_ma_threshold.
    - Past min_steady_days must have (min_steady_days - 2) or more stable points.
    """

    def __init__(
        self,
        box_window: int = 15,
        min_steady_days: Optional[int] = None,
        convergence_threshold: float = 0.015,
        ma_slope_threshold: float = 0.01,
        price_to_ma_threshold: float = 0.02,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize Ribbon Coherence Detector.

        Args:
            box_window: Used as min_steady_days when min_steady_days is None (default: 15).
            min_steady_days: Minimum number of days the ribbon must stay coherent.
                If None, uses box_window.
            convergence_threshold: Max MA spread (std of MA5/10/20 / MA20) (default: 0.015, 1.5%).
            ma_slope_threshold: Max relative change of MA20 over 5 days (default: 0.01, 1%).
            price_to_ma_threshold: Max |close - MA20| / MA20 (default: 0.02, 2%).
            smooth_window: Optional extra smoothing window (default: None).
            smooth_threshold: Optional smoothing threshold (default: None).
        """
        steady = min_steady_days if min_steady_days is not None else box_window
        super().__init__(box_window=steady, smooth_window=smooth_window, smooth_threshold=smooth_threshold)
        self.min_steady_days = steady
        self.convergence_threshold = convergence_threshold
        self.ma_slope_threshold = ma_slope_threshold
        self.price_to_ma_threshold = price_to_ma_threshold

    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect boxes using MA ribbon coherence (three MAs intertwined, price on ribbon).

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low.

        Returns:
            DataFrame with box_h, box_l, box_width, is_box_candidate and
            ma5, ma10, ma20, ma_cv, ma_slope, price_dev, is_stable_point.
        """
        # 1. MA ribbon
        df = df.with_columns([
            pl.col("close").rolling_mean(window_size=5).over("ts_code").alias("ma5"),
            pl.col("close").rolling_mean(window_size=10).over("ts_code").alias("ma10"),
            pl.col("close").rolling_mean(window_size=20).over("ts_code").alias("ma20"),
        ])

        # 2. Coherence: horizontal std of [ma5, ma10, ma20] / ma20 (no horizontal_std in Polars)
        ma_mean = (
            pl.col("ma5") + pl.col("ma10") + pl.col("ma20")
        ) / 3
        ma_var = (
            (pl.col("ma5") - ma_mean).pow(2)
            + (pl.col("ma10") - ma_mean).pow(2)
            + (pl.col("ma20") - ma_mean).pow(2)
        ) / 3
        ma_std = ma_var.sqrt()

        df = df.with_columns([
            (ma_std / (pl.col("ma20") + 1e-10)).alias("ma_cv"),
            (
                (pl.col("ma20") - pl.col("ma20").shift(5).over("ts_code")).abs()
                / (pl.col("ma20").shift(5).over("ts_code") + 1e-10)
            ).alias("ma_slope"),
            ((pl.col("close") - pl.col("ma20")).abs() / (pl.col("ma20") + 1e-10)).alias("price_dev"),
        ])

        # 3. Stable point: all three conditions
        df = df.with_columns(
            (
                (pl.col("ma_cv") < self.convergence_threshold)
                & (pl.col("ma_slope") < self.ma_slope_threshold)
                & (pl.col("price_dev") < self.price_to_ma_threshold)
                & pl.col("ma20").is_not_null()
                & (pl.col("ma20") > 0)
            ).alias("is_stable_point")
        )

        # 4. Box candidate: past min_steady_days have at least (min_steady_days - 2) stable points (~90%+)
        min_stable_count = max(1, self.min_steady_days - 2)
        df = df.with_columns(
            (
                pl.col("is_stable_point").cast(pl.Int32).rolling_sum(window_size=self.min_steady_days).over("ts_code")
                >= min_stable_count
            ).alias("is_box_candidate")
        )

        # 5. Box bounds from close range over the same window (stable zone only)
        df = df.with_columns([
            pl.col("close").rolling_max(window_size=self.min_steady_days).over("ts_code").alias("box_h"),
            pl.col("close").rolling_min(window_size=self.min_steady_days).over("ts_code").alias("box_l"),
        ])
        df = df.with_columns(
            ((pl.col("box_h") - pl.col("box_l")) / (pl.col("box_l") + 1e-10)).alias("box_width")
        )

        return self._apply_smoothing(df)
