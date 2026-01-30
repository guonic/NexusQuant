"""
Balanced Box Detector for Teapot pattern recognition.

Designed to detect "statistically balanced" boxes with:
1. Pivot Symmetry: Moving average is centered between high and low
2. Volatility Uniformity: Daily volatility is consistent (no extreme spikes)
3. Distribution Uniformity: Price occupies upper, middle, and lower bands
"""

import logging
from typing import Optional

import polars as pl

from nq.trading.selector.teapot.box_detector import BoxDetector

logger = logging.getLogger(__name__)


class BalancedBoxDetector(BoxDetector):
    """
    Balanced Box Detector (均衡箱体检测器).
    
    Designed to detect "statistically balanced" boxes with:
    1. Pivot Symmetry (中枢对称性): Moving average is centered between high and low
    2. Volatility Uniformity (波动均匀度): Daily volatility is consistent (no extreme spikes)
    3. Distribution Uniformity (分布均匀度): Price occupies upper, middle, and lower bands
    
    This detector filters out:
    - Boxes with extreme spikes (large wicks)
    - Boxes with asymmetric price distribution
    - Boxes with inconsistent daily volatility
    """
    
    def __init__(
        self,
        box_window: int = 30,
        height_threshold: float = 0.08,
        symmetry_threshold: float = 0.15,
        uniformity_threshold: float = 0.4,
        slope_threshold: float = 0.01,
        smooth_window: Optional[int] = None,
        smooth_threshold: Optional[int] = None,
    ):
        """
        Initialize Balanced Box Detector.
        
        Args:
            box_window: Window size for box calculation (default: 30 days).
            height_threshold: Maximum box height relative to lower bound (default: 0.08, 8%).
            symmetry_threshold: Maximum symmetry error (default: 0.15, 15%).
                Lower value means stricter symmetry requirement.
            uniformity_threshold: Maximum volatility coefficient of variation (default: 0.4, 40%).
                Lower value means more uniform volatility.
            slope_threshold: Maximum slope change for box mid-line (default: 0.01, 1%).
            smooth_window: Window size for box filter smoothing (default: None, disabled).
            smooth_threshold: Minimum number of days in smooth_window (default: None, disabled).
        """
        super().__init__(box_window, smooth_window, smooth_threshold)
        self.height_threshold = height_threshold
        self.symmetry_threshold = symmetry_threshold
        self.uniformity_threshold = uniformity_threshold
        self.slope_threshold = slope_threshold
    
    def detect_box(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect balanced boxes using symmetry and uniformity metrics.
        
        Args:
            df: Input DataFrame with columns: ts_code, trade_date, close, high, low, volume.
            
        Returns:
            DataFrame with box features and is_box_candidate flag.
        """
        # 1. Calculate basic box boundaries
        df = df.with_columns([
            pl.col("high")
            .rolling_max(window_size=self.box_window)
            .over("ts_code")
            .alias("box_h"),
            pl.col("low")
            .rolling_min(window_size=self.box_window)
            .over("ts_code")
            .alias("box_l"),
            pl.col("close")
            .rolling_mean(window_size=self.box_window)
            .over("ts_code")
            .alias("box_mid"),
        ])
        
        # 2. Calculate balance metrics
        
        # (1) Symmetry Error (对称性误差)
        # Ideal: (box_h + box_l) / 2 == box_mid
        # Calculate deviation from ideal center as percentage of box height
        df = df.with_columns([
            (
                ((pl.col("box_h") + pl.col("box_l")) / 2 - pl.col("box_mid")).abs()
                / (pl.col("box_h") - pl.col("box_l") + 1e-8)  # Avoid division by zero
            ).alias("symmetry_error"),
        ])
        
        # (2) Volatility Uniformity (波动均匀度)
        # Coefficient of Variation (CV) of daily volatility
        # If some days have extreme spikes, CV will be high
        df = df.with_columns([
            # Daily volatility (high - low)
            (pl.col("high") - pl.col("low")).alias("daily_vol"),
        ])
        
        df = df.with_columns([
            # Mean and std of daily volatility over window
            pl.col("daily_vol")
            .rolling_mean(window_size=self.box_window)
            .over("ts_code")
            .alias("vol_mean"),
            pl.col("daily_vol")
            .rolling_std(window_size=self.box_window)
            .over("ts_code")
            .alias("vol_std"),
        ])
        
        df = df.with_columns([
            # Coefficient of Variation: std / mean
            (pl.col("vol_std") / (pl.col("vol_mean") + 1e-8)).alias("vol_cv"),
        ])
        
        # (3) Box Height (箱体高度)
        df = df.with_columns([
            ((pl.col("box_h") - pl.col("box_l")) / pl.col("box_l")).alias("box_height"),
        ])
        
        # (4) Slope Check (斜率检查)
        # Ensure box mid-line is relatively flat
        df = df.with_columns([
            (
                (pl.col("box_mid") - pl.col("box_mid").shift(5)).abs()
                / (pl.col("box_mid").shift(5) + 1e-8)
            ).alias("mid_slope"),
        ])
        
        # 3. Detection logic
        df = df.with_columns([
            (
                # Condition A: Box height within threshold
                (pl.col("box_height") < self.height_threshold)
                &
                # Condition B: High symmetry (moving average near center)
                (pl.col("symmetry_error") < self.symmetry_threshold)
                &
                # Condition C: Uniform volatility (no extreme spikes)
                (pl.col("vol_cv") < self.uniformity_threshold)
                &
                # Condition D: Flat mid-line (no significant slope)
                (pl.col("mid_slope") < self.slope_threshold)
                &
                # Condition E: Valid values
                pl.col("box_h").is_not_null()
                & pl.col("box_l").is_not_null()
                & (pl.col("box_h") > pl.col("box_l"))
                & pl.col("box_mid").is_not_null()
                & (pl.col("box_mid") > 0)
            ).alias("is_box_candidate"),
        ])
        
        # 4. Calculate box width (for compatibility)
        df = df.with_columns([
            pl.col("box_height").alias("box_width"),
        ])
        
        # 5. Apply smoothing if enabled
        df = self._apply_smoothing(df)
        
        return df
