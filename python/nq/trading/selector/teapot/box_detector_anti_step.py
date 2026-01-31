"""
Anti-Step Box Detector for Teapot pattern recognition.

Rejects "step-shaped" pseudo-boxes: left-high-right-low (cliff drop) or
single-sided trends. Uses rolling correlation (price vs time) and
close's relative position in the window to enforce "no directionality".
"""

import logging
from typing import Optional

import polars as pl

from nq.trading.selector.teapot.box_detector import BoxDetector

logger = logging.getLogger(__name__)


class AntiStepBoxDetector(BoxDetector):
    """
    Anti-step / anti-trend box detector (反阶梯/反趋势箱体检测器).

    Filters out boxes that are "welded" from two regimes (e.g. high zone + cliff drop).
    Two core filters:
    - R (rolling correlation of close vs time): |R| must be small (no strong trend).
    - Relative position: close must not sit at the extreme bottom/top of the window
      (no cliff-at-end).

    Equilibrium = narrow in space AND directionless in time.
    """

    def __init__(
        self,
        box_window: int = 20,
        r_threshold: float = 0.4,
        center_dev_threshold: float = 0.6,
        box_width_threshold: float = 0.15,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize Anti-Step Box Detector.

        Args:
            box_window: Rolling window size (default: 20).
            r_threshold: Max |rolling_corr(time, close)| (default: 0.4). Higher |R| = trend.
            center_dev_threshold: Max |close - rolling_mean(close)| / range (default: 0.6).
                Prevents close at the extreme bottom/top of the window.
            box_width_threshold: Max (box_h - box_l) / box_l to consider as box (default: 0.15).
            smooth_window: Optional smoothing window (default: None).
            smooth_threshold: Optional smoothing threshold (default: None).
        """
        super().__init__(box_window=box_window, smooth_window=smooth_window, smooth_threshold=smooth_threshold)
        self.r_threshold = r_threshold
        self.center_dev_threshold = center_dev_threshold
        self.box_width_threshold = box_width_threshold

    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect boxes by rejecting directional (step/trend) windows.

        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low.

        Returns:
            DataFrame with box_h, box_l, box_width, is_box_candidate, r_value, center_dev.
        """
        # 1. Per-stock row index (0, 1, 2, ...) for correlation with time
        df = df.with_columns(
            (pl.col("close").cum_count().over("ts_code") - 1).cast(pl.Int64).alias("_idx")
        )

        # 2. Rolling correlation: price vs time (within window)
        df = df.with_columns(
            pl.rolling_corr(pl.col("_idx"), pl.col("close"), window_size=self.box_window)
            .over("ts_code")
            .alias("r_value")
        )

        # 3. Box bounds and range (same window)
        df = df.with_columns([
            pl.col("high").rolling_max(window_size=self.box_window).over("ts_code").alias("box_h"),
            pl.col("low").rolling_min(window_size=self.box_window).over("ts_code").alias("box_l"),
            pl.col("close").rolling_mean(window_size=self.box_window).over("ts_code").alias("_close_mean"),
        ])
        df = df.with_columns(
            (pl.col("box_h") - pl.col("box_l")).alias("_range")
        )

        # 4. Center deviation: |close - rolling_mean(close)| / range (avoid cliff at end)
        df = df.with_columns(
            ((pl.col("close") - pl.col("_close_mean")).abs() / (pl.col("_range") + 1e-10)).alias("center_dev")
        )

        # 5. Box width (for compatibility and filter)
        df = df.with_columns(
            ((pl.col("box_h") - pl.col("box_l")) / (pl.col("box_l") + 1e-10)).alias("box_width")
        )

        # 6. Candidate: no strong trend (|R| < r_threshold), close not at extreme (center_dev < threshold), narrow box
        df = df.with_columns(
            (
                pl.col("r_value").is_not_null()
                & (pl.col("r_value").abs() < self.r_threshold)
                & (pl.col("center_dev") < self.center_dev_threshold)
                & (pl.col("box_width") < self.box_width_threshold)
                & (pl.col("box_width") > 0)
                & pl.col("box_h").is_not_null()
                & pl.col("box_l").is_not_null()
                & (pl.col("box_h") > pl.col("box_l"))
            ).alias("is_box_candidate")
        )

        # Drop temporary columns
        df = df.drop(["_idx", "_close_mean", "_range"])

        return self._apply_smoothing(df)
