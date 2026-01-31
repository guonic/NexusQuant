# -*- coding: utf-8 -*-
"""
teapot_box_detector_evaluation.py

Description:
    Evaluate box detection algorithms for Teapot pattern recognition.
    Compares different box detectors and generates evaluation reports.

Usage:
    python teapot_box_detector_evaluation.py --start-date 2023-01-01 --end-date 2024-01-01
    python teapot_box_detector_evaluation.py --start-date 2023-01-01 --end-date 2024-01-01 --symbols 000001.SZ 000002.SZ
    python teapot_box_detector_evaluation.py --start-date 2023-01-01 --end-date 2024-01-01 --output outputs/teapot/evaluation

Arguments:
    --start-date      Start date for evaluation (YYYY-MM-DD)
    --end-date        End date for evaluation (YYYY-MM-DD)
    --symbols         Optional list of stock codes to evaluate (default: all stocks)
    --output          Output directory for reports (default: outputs/teapot/evaluation)
    --box-window      Box window size (default: 40)
    --use-cache       Use cached data if available (default: False)

Output:
    - Evaluation report CSV files
    - Comparison statistics
    - Box detection statistics per stock
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import polars as pl

from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from nq.trading.selector.teapot.box_detector import (
    BoxDetector,
    HybridBoxDetector,
    HybridBoxDetectorV2,
    MeanReversionBoxDetector,
    MeanReversionBoxDetectorV2,
    SimpleBoxDetector,
)
from nq.trading.selector.teapot.box_detector_ma_convergence import (
    MovingAverageConvergenceBoxDetector,
)
from nq.trading.selector.teapot.box_detector_dynamic_convergence import (
    DynamicConvergenceDetector,
)
from nq.trading.selector.teapot.box_detector_keltner_squeeze import (
    ExpansionAnchorBoxDetector,
    KeltnerSqueezeDetector,
)
from nq.trading.selector.teapot.box_detector_accurate import (
    AccurateBoxDetector,
)
from nq.trading.selector.teapot.box_detector_balanced import (
    BalancedBoxDetector,
)
from nq.trading.selector.teapot.box_detector_dense_area import (
    DenseAreaBoxDetector,
)
from nq.trading.selector.teapot.box_detector_ribbon_coherence import (
    RibbonCoherenceDetector,
)
from nq.trading.selector.teapot.box_detector_composite_equilibrium import (
    CompositeEquilibriumDetector,
)
from nq.trading.selector.teapot.box_detector_anti_step import (
    AntiStepBoxDetector,
)
from nq.trading.selector.teapot.box_visualizer import BoxVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BoxDetectorEvaluator:
    """
    Evaluator for box detection algorithms.

    Compares different box detectors and generates evaluation reports.
    """

    def __init__(
        self,
        db_config: DatabaseConfig,
        schema: str = "quant",
        use_cache: bool = False,
    ):
        """
        Initialize evaluator.

        Args:
            db_config: Database configuration.
            schema: Database schema name.
            use_cache: Whether to use cached data.
        """
        self.db_config = db_config
        self.schema = schema
        self.data_loader = TeapotDataLoader(
            db_config=db_config,
            schema=schema,
            use_cache=use_cache,
        )

    def evaluate_detector(
        self,
        detector: BoxDetector,
        df: pl.DataFrame,
        detector_name: str,
    ) -> Dict:
        """
        Evaluate a single box detector.

        Args:
            detector: Box detector instance.
            df: Input DataFrame with stock data.
            detector_name: Name of the detector.

        Returns:
            Dictionary with evaluation metrics.
        """
        logger.info(f"Evaluating {detector_name}...")

        # Detect boxes
        df_with_boxes = detector.detect_box(df)

        # Calculate statistics
        total_rows = len(df_with_boxes)
        box_candidates = df_with_boxes.filter(pl.col("is_box_candidate"))
        box_count = len(box_candidates)

        if box_count == 0:
            return {
                "detector_name": detector_name,
                "total_rows": total_rows,
                "box_count": 0,
                "box_ratio": 0.0,
                "avg_box_width": None,
                "min_box_width": None,
                "max_box_width": None,
                "stocks_with_boxes": 0,
                "total_stocks": df["ts_code"].n_unique(),
            }

        # Box width statistics
        box_width_mean = box_candidates["box_width"].mean()
        box_width_min = box_candidates["box_width"].min()
        box_width_max = box_candidates["box_width"].max()
        box_width_std = box_candidates["box_width"].std()

        # Stocks with boxes
        stocks_with_boxes = box_candidates["ts_code"].n_unique()
        total_stocks = df["ts_code"].n_unique()

        # Box duration statistics (consecutive days)
        box_durations = self._calculate_box_durations(df_with_boxes)

        return {
            "detector_name": detector_name,
            "total_rows": total_rows,
            "box_count": box_count,
            "box_ratio": box_count / total_rows if total_rows > 0 else 0.0,
            "avg_box_width": box_width_mean,
            "min_box_width": box_width_min,
            "max_box_width": box_width_max,
            "std_box_width": box_width_std,
            "stocks_with_boxes": stocks_with_boxes,
            "total_stocks": total_stocks,
            "stock_coverage": stocks_with_boxes / total_stocks if total_stocks > 0 else 0.0,
            "avg_box_duration": box_durations.get("mean") if box_durations else None,
            "max_box_duration": box_durations.get("max") if box_durations else None,
        }

    def _calculate_box_durations(self, df: pl.DataFrame) -> Optional[Dict]:
        """
        Calculate box duration statistics (consecutive days).

        Args:
            df: DataFrame with is_box_candidate column.

        Returns:
            Dictionary with duration statistics or None if no boxes.
        """
        durations = []

        for ts_code in df["ts_code"].unique():
            stock_df = (
                df.filter(pl.col("ts_code") == ts_code)
                .sort("trade_date")
                .with_columns(
                    [
                        (
                            pl.col("is_box_candidate")
                            != pl.col("is_box_candidate").shift(1)
                        ).alias("state_change"),
                    ]
                )
            )

            # Find consecutive box periods
            current_duration = 0
            for row in stock_df.iter_rows(named=True):
                if row["is_box_candidate"]:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                        current_duration = 0
            # Handle last period
            if current_duration > 0:
                durations.append(current_duration)

        if not durations:
            return None

        return {
            "mean": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
            "median": sorted(durations)[len(durations) // 2],
        }

    def evaluate_all_detectors(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
        box_window: int = 40,
    ) -> pl.DataFrame:
        """
        Evaluate all box detectors.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            symbols: Optional list of stock codes.
            box_window: Box window size.

        Returns:
            DataFrame with evaluation results for all detectors.
        """
        logger.info(f"Loading data: {start_date} to {end_date}")

        # Load data
        df = self.data_loader.load_daily_data(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
        )

        if df.is_empty():
            logger.warning("No data loaded")
            return pl.DataFrame()

        logger.info(f"Loaded {len(df)} rows for {df['ts_code'].n_unique()} stocks")

        # Define detectors to evaluate
        detectors = {
            # "simple": SimpleBoxDetector(
            #     box_window=box_window,
            #     box_width_threshold=0.15,
            # ),
            # "mean_reversion": MeanReversionBoxDetector(
            #     box_window=box_window,
            #     max_total_return=0.05,
            #     max_relative_box_height=0.08,
            # ),
            # "mean_reversion_smooth": MeanReversionBoxDetector(
            #     box_window=box_window,
            #     max_total_return=0.05,
            #     max_relative_box_height=0.08,
            #     smooth_window=5,
            #     smooth_threshold=4,
            # ),
            # "mean_reversion_v2": MeanReversionBoxDetectorV2(
            #     box_window=box_window,
            #     max_total_return=None,  # Use volatility-based
            #     max_relative_box_height=0.12,
            #     volatility_ratio=0.25,
            #     smooth_window=10,
            #     smooth_threshold=6,
            # ),
            # "hybrid": HybridBoxDetector(
            #     box_window=box_window,
            #     box_width_threshold=0.15,
            #     max_total_return=0.05,
            #     max_relative_box_height=0.08,
            # ),
            # "hybrid_v2": HybridBoxDetectorV2(
            #     box_window=box_window,
            #     box_width_threshold=0.15,
            #     max_total_return=None,  # Use volatility-based
            #     max_relative_box_height=0.12,
            #     volatility_ratio=0.25,
            #     score_threshold=0.5,
            #     weight_simple=0.3,
            #     weight_mean_reversion=0.4,
            #     weight_mean_reversion_v2=0.3,
            # ),
            # "ma_convergence": MovingAverageConvergenceBoxDetector(
            #     box_window=20,  # Shorter window for step patterns
            #     ma_periods=[5, 10, 20, 30],
            #     convergence_threshold=0.02,
            #     max_relative_height=0.10,
            #     smooth_window=5,
            #     smooth_threshold=3,
            # ),
            # "dynamic_convergence": DynamicConvergenceDetector(
            #     box_window=30,
            #     ma_periods=[5, 10, 20],
            #     max_tightness=0.08,
            #     volness_threshold=0.6,
            #     smooth_window=5,
            #     smooth_threshold=3,
            # ),
            # "keltner_squeeze": KeltnerSqueezeDetector(
            #     box_window=20,
            #     squeeze_threshold=0.06,
            #     slope_threshold=0.015,
            #     atr_multiplier=1.5,
            #     volume_decay_threshold=0.8,
            #     expansion_window=15,
            #     stability_window=20,
            #     stability_threshold=15,
            #     smooth_window=5,
            #     smooth_threshold=3,
            # ),
            # "expansion_anchor": ExpansionAnchorBoxDetector(
            #     box_window=40,
            #     squeeze_threshold=0.06,
            #     slope_threshold=0.015,
            #     atr_multiplier=1.5,
            #     lookback_window=60,
            #     expansion_window=20,
            #     stability_window=20,
            #     stability_threshold=15,
            #     smooth_window=5,
            #     smooth_threshold=3,
            # ),
            "accurate": AccurateBoxDetector(
                box_window=30,
                price_tol=0.09,  # Further increased to 0.12 (12%) for better recall
                # Based on debug analysis: 75th percentile upper_dist=0.1372, so 0.12 captures more boxes
                min_cross_count=4,
                smooth_window=4,
                smooth_threshold=3,
            ),
            "balanced": BalancedBoxDetector(
                box_window=30,
                height_threshold=0.12,  # Increased to 12% for better recall
                symmetry_threshold=0.15,  # Relaxed to 15% for better recall
                uniformity_threshold=0.45,  # Increased to 45% (based on median CV=0.4517)
                slope_threshold=0.015,  # Increased to 1.5% slope change
                smooth_window=5,
                smooth_threshold=3,
            ),
            "dense_area": DenseAreaBoxDetector(
                box_window=40,
                dense_threshold=0.05,  # 5% core band; use 0.03–0.04 for stricter
                quantile_high=0.8,
                quantile_low=0.2,
                vol_stability_threshold=0.03,
                mid_slope_threshold=0.02,
                smooth_window=5,
                smooth_threshold=3,
            ),
            "ribbon_coherence": RibbonCoherenceDetector(
                min_steady_days=15,
                convergence_threshold=0.015,
                ma_slope_threshold=0.01,
                price_to_ma_threshold=0.02,
                smooth_window=5,
                smooth_threshold=3,
            ),
            "composite_equilibrium": CompositeEquilibriumDetector(
                box_window=20,
                ma_cohesion_threshold=0.015,
                quantile_width_threshold=0.04,
                quantile_window=20,
                cross_count_min=3,
                cross_window=15,
                volume_short=15,
                volume_long=60,
                volume_ratio=0.8,
                smooth_window=5,
                smooth_threshold=3,
            ),
            "anti_step": AntiStepBoxDetector(
                box_window=20,
                r_threshold=0.4,
                center_dev_threshold=0.6,
                box_width_threshold=0.15,
                smooth_window=5,
                smooth_threshold=3,
            ),
            # "expansion_anchor": ExpansionAnchorBoxDetector(
            #     box_window=40,
            #     squeeze_threshold=0.06,
            #     slope_threshold=0.015,
            #     atr_multiplier=1.5,
            #     lookback_window=60,
            #     expansion_window=20,
            #     stability_window=20,
            #     stability_threshold=12,  # Reduced from 15 to 12 for better recall
            #     smooth_window=5,
            #     smooth_threshold=3,
            # ),
        }

        # Evaluate each detector
        results = []
        for name, detector in detectors.items():
            result = self.evaluate_detector(detector, df, name)
            results.append(result)

        return pl.DataFrame(results)

    def extract_boxes_with_dates(
        self,
        df: pl.DataFrame,
        detector: BoxDetector,
        detector_name: str,
        min_box_days: int = 5,
    ) -> pl.DataFrame:
        """
        Extract boxes with start and end dates from detection results.

        Args:
            df: DataFrame with K-line data.
            detector: Box detector instance.
            detector_name: Name of the detector.
            min_box_days: Minimum number of consecutive days for a valid box (default: 5).

        Returns:
            DataFrame with box information including box_id, start_date, end_date.
        """
        # Detect boxes
        df_with_boxes = detector.detect_box(df)

        # Find consecutive box periods per stock using Polars operations
        boxes = []

        for ts_code in df_with_boxes["ts_code"].unique():
            stock_df = (
                df_with_boxes.filter(pl.col("ts_code") == ts_code)
                .sort("trade_date")
                .with_row_index("row_idx")
            )

            # Create groups for consecutive box periods
            # Mark state changes: when is_box_candidate changes, create new group
            stock_df = stock_df.with_columns([
                (
                    (pl.col("is_box_candidate") != pl.col("is_box_candidate").shift(1).fill_null(False))
                    | (pl.col("row_idx") == 0)
                ).cast(pl.Int32).cum_sum().alias("group_id")
            ])

            # Filter only box candidate groups
            box_groups = stock_df.filter(pl.col("is_box_candidate"))

            if box_groups.is_empty():
                continue

            # Group by group_id to find consecutive periods
            for group_id in box_groups["group_id"].unique():
                group_df = box_groups.filter(pl.col("group_id") == group_id)
                
                if len(group_df) < min_box_days:
                    # Skip boxes that are too short
                    continue

                # Get box period dates
                box_start_date = group_df["trade_date"].min()
                box_end_date = group_df["trade_date"].max()
                
                # Calculate box bounds based on ACTUAL entity range (open/close) during box period
                # Use entity range instead of high/low to focus on actual trading body
                # Entity high = max(open, close), Entity low = min(open, close)
                group_df = group_df.with_columns([
                    pl.max_horizontal(["open", "close"]).alias("entity_high"),
                    pl.min_horizontal(["open", "close"]).alias("entity_low"),
                ])
                
                actual_entity_high_values = group_df["entity_high"].drop_nulls()
                actual_entity_low_values = group_df["entity_low"].drop_nulls()
                
                if actual_entity_high_values.is_empty() or actual_entity_low_values.is_empty():
                    continue
                
                # Use quantiles to filter outliers (95th percentile for high, 5th for low)
                # This gives a more stable box range that represents the actual entity trading range
                box_h = actual_entity_high_values.quantile(0.95)
                box_l = actual_entity_low_values.quantile(0.05)
                
                # Fallback to max/min if quantile is None
                if box_h is None:
                    box_h = actual_entity_high_values.max()
                if box_l is None:
                    box_l = actual_entity_low_values.min()
                
                # Ensure box_h > box_l
                if box_h <= box_l:
                    # Use max/min as fallback
                    box_h = actual_entity_high_values.max()
                    box_l = actual_entity_low_values.min()
                    if box_h <= box_l:
                        continue  # Skip invalid box
                
                if box_h and box_l and box_start_date and box_end_date:
                    box_id = f"{ts_code}.{box_start_date}.{box_end_date}"
                    boxes.append({
                        "box_id": box_id,
                        "ts_code": ts_code,
                        "box_start_date": box_start_date,
                        "box_end_date": box_end_date,
                        "box_h": float(box_h),
                        "box_l": float(box_l),
                        "detector_name": detector_name,
                        "box_duration_days": len(group_df),
                    })

        return pl.DataFrame(boxes) if boxes else pl.DataFrame()

    def evaluate_detector_overlap(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
        box_window: int = 40,
    ) -> pl.DataFrame:
        """
        Evaluate box detection overlap between different detectors.

        For each row, marks which detectors detected it as a box candidate.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            symbols: Optional list of stock codes.
            box_window: Box window size.

        Returns:
            DataFrame with detection results from all detectors, including
            columns indicating which detector(s) detected each box.
        """
        logger.info(f"Loading data: {start_date} to {end_date}")

        # Load data
        df = self.data_loader.load_daily_data(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
        )

        if df.is_empty():
            logger.warning("No data loaded")
            return pl.DataFrame()

        # Define detectors
        detectors = {
            # "simple": SimpleBoxDetector(
            #     box_window=box_window,
            #     box_width_threshold=0.15,
            # ),
            # "mean_reversion": MeanReversionBoxDetector(
            #     box_window=box_window,
            #     max_total_return=0.05,
            #     max_relative_box_height=0.08,
            # ),
            # "mean_reversion_v2": MeanReversionBoxDetectorV2(
            #     box_window=box_window,
            #     max_total_return=None,
            #     max_relative_box_height=0.12,
            #     volatility_ratio=0.25,
            # ),
            # "hybrid": HybridBoxDetector(
            #     box_window=box_window,
            #     box_width_threshold=0.15,
            #     max_total_return=0.05,
            #     max_relative_box_height=0.08,
            # ),
            "hybrid_v2": HybridBoxDetectorV2(
                box_window=box_window,
                box_width_threshold=0.15,
                max_total_return=None,
                max_relative_box_height=0.12,
                volatility_ratio=0.25,
                score_threshold=0.5,
            ),
            "ma_convergence": MovingAverageConvergenceBoxDetector(
                box_window=20,
                ma_periods=[5, 10, 20, 30],
                convergence_threshold=0.02,
                max_relative_height=0.10,
                smooth_window=5,
                smooth_threshold=3,
            ),
            "dynamic_convergence": DynamicConvergenceDetector(
                box_window=30,
                ma_periods=[5, 10, 20],
                max_tightness=0.08,
                volness_threshold=0.6,
                smooth_window=5,
                smooth_threshold=3,
            ),
            "keltner_squeeze": KeltnerSqueezeDetector(
                box_window=20,
                squeeze_threshold=0.06,
                slope_threshold=0.015,
                atr_multiplier=1.5,
                volume_decay_threshold=0.8,
                expansion_window=15,
                stability_window=20,
                stability_threshold=15,
                smooth_window=5,
                smooth_threshold=3,
            ),
            "expansion_anchor": ExpansionAnchorBoxDetector(
                box_window=40,
                squeeze_threshold=0.06,
                slope_threshold=0.015,
                atr_multiplier=1.5,
                lookback_window=60,
                expansion_window=20,
                stability_window=20,
                stability_threshold=12,  # Reduced from 15 to 12 for better recall
                smooth_window=5,
                smooth_threshold=3,
            ),
            "accurate": AccurateBoxDetector(
                box_window=30,
                price_tol=0.09,  # Match main evaluation parameters
                min_cross_count=4,
                min_touch_count=1,  # Require at least 1 touch on each boundary
                touch_tolerance=0.03,  # 3% tolerance for touching boundary
                smooth_window=4,
                smooth_threshold=3,
            ),
            "balanced": BalancedBoxDetector(
                box_window=30,
                height_threshold=0.12,  # Match main evaluation parameters
                symmetry_threshold=0.15,
                uniformity_threshold=0.45,
                slope_threshold=0.015,
                smooth_window=5,
                smooth_threshold=3,
            ),
            "dense_area": DenseAreaBoxDetector(
                box_window=40,
                dense_threshold=0.05,
                quantile_high=0.8,
                quantile_low=0.2,
                vol_stability_threshold=0.03,
                mid_slope_threshold=0.02,
                smooth_window=5,
                smooth_threshold=3,
            ),
            "ribbon_coherence": RibbonCoherenceDetector(
                min_steady_days=15,
                convergence_threshold=0.015,
                ma_slope_threshold=0.01,
                price_to_ma_threshold=0.02,
                smooth_window=5,
                smooth_threshold=3,
            ),
            "composite_equilibrium": CompositeEquilibriumDetector(
                box_window=20,
                ma_cohesion_threshold=0.015,
                quantile_width_threshold=0.04,
                quantile_window=20,
                cross_count_min=3,
                cross_window=15,
                volume_short=15,
                volume_long=60,
                volume_ratio=0.8,
                smooth_window=5,
                smooth_threshold=3,
            ),
            "anti_step": AntiStepBoxDetector(
                box_window=20,
                r_threshold=0.4,
                center_dev_threshold=0.6,
                box_width_threshold=0.15,
                smooth_window=5,
                smooth_threshold=3,
            ),
        }

        # Apply all detectors
        result_df = df.clone()
        detector_flags = []

        for name, detector in detectors.items():
            logger.info(f"Running {name} detector...")
            df_detected = detector.detect_box(df)
            result_df = result_df.with_columns([
                df_detected["is_box_candidate"].alias(f"is_box_{name}"),
            ])
            detector_flags.append(f"is_box_{name}")

        # Create summary: which detectors detected each box
        # Calculate detector count first
        result_df = result_df.with_columns([
            pl.sum_horizontal([
                pl.col(f"is_box_{name}").cast(pl.Int32)
                for name in detectors.keys()
            ]).alias("detector_count"),
        ])
        
        # Build detected_by string using map_elements
        detector_cols = [f"is_box_{name}" for name in detectors.keys()]
        result_df = result_df.with_columns([
            pl.struct(detector_cols)
            .map_elements(
                lambda x: ",".join([name for name in detectors.keys() if x.get(f"is_box_{name}", False)]),
                return_dtype=pl.Utf8,
            )
            .alias("detected_by"),
        ])

        # Filter to only box candidates (at least one detector detected)
        result_df = result_df.filter(pl.col("detector_count") > 0)

        return result_df

    def evaluate_per_stock(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
        box_window: int = 40,
        detector_type: str = "dynamic_convergence",
    ) -> pl.DataFrame:
        """
        Evaluate box detection per stock.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            symbols: Optional list of stock codes.
            box_window: Box window size.
            detector_type: Detector type to use.

        Returns:
            DataFrame with per-stock evaluation results.
        """
        logger.info(f"Loading data: {start_date} to {end_date}")

        # Load data
        df = self.data_loader.load_daily_data(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
        )

        if df.is_empty():
            logger.warning("No data loaded")
            return pl.DataFrame()

        # Create detector
        if detector_type == "simple":
            detector = SimpleBoxDetector(box_window=box_window, box_width_threshold=0.15)
        elif detector_type == "mean_reversion":
            detector = MeanReversionBoxDetector(
                box_window=box_window,
                max_total_return=0.05,
                max_relative_box_height=0.08,
            )
        elif detector_type == "hybrid":
            detector = HybridBoxDetector(
                box_window=box_window,
                box_width_threshold=0.15,
                max_total_return=0.05,
                max_relative_box_height=0.08,
            )
        elif detector_type == "mean_reversion_v2":
            detector = MeanReversionBoxDetectorV2(
                box_window=box_window,
                max_total_return=None,  # Use volatility-based
                max_relative_box_height=0.12,
                volatility_ratio=0.25,
            )
        elif detector_type == "hybrid_v2":
            detector = HybridBoxDetectorV2(
                box_window=box_window,
                box_width_threshold=0.15,
                max_total_return=None,  # Use volatility-based
                max_relative_box_height=0.12,
                volatility_ratio=0.25,
                score_threshold=0.5,
            )
        elif detector_type == "ma_convergence":
            detector = MovingAverageConvergenceBoxDetector(
                box_window=20,
                ma_periods=[5, 10, 20, 30],
                convergence_threshold=0.02,
                max_relative_height=0.10,
                smooth_window=5,
                smooth_threshold=3,
            )
        elif detector_type == "dynamic_convergence":
            detector = DynamicConvergenceDetector(
                box_window=30,
                ma_periods=[5, 10, 20],
                max_tightness=0.08,
                volness_threshold=0.6,
                smooth_window=5,
                smooth_threshold=3,
            )
        elif detector_type == "keltner_squeeze":
            detector = KeltnerSqueezeDetector(
                box_window=20,
                squeeze_threshold=0.06,
                slope_threshold=0.015,
                atr_multiplier=1.5,
                volume_decay_threshold=0.8,
                expansion_window=15,
                stability_window=20,
                stability_threshold=15,
                smooth_window=5,
                smooth_threshold=3,
            )
        elif detector_type == "expansion_anchor":
            detector = ExpansionAnchorBoxDetector(
                box_window=40,
                squeeze_threshold=0.06,
                slope_threshold=0.015,
                atr_multiplier=1.5,
                lookback_window=60,
                expansion_window=20,
                stability_window=20,
                stability_threshold=15,
                smooth_window=5,
                smooth_threshold=3,
            )
        elif detector_type == "accurate":
            detector = AccurateBoxDetector(
                box_window=30,
                price_tol=0.09,  # Match evaluation script parameters
                min_cross_count=4,
                min_touch_count=1,
                touch_tolerance=0.03,
                smooth_window=4,
                smooth_threshold=3,
            )
        elif detector_type == "balanced":
            detector = BalancedBoxDetector(
                box_window=30,
                height_threshold=0.12,
                symmetry_threshold=0.15,
                uniformity_threshold=0.45,
                slope_threshold=0.015,
                smooth_window=5,
                smooth_threshold=3,
            )
        elif detector_type == "dense_area":
            detector = DenseAreaBoxDetector(
                box_window=40,
                dense_threshold=0.05,
                quantile_high=0.8,
                quantile_low=0.2,
                vol_stability_threshold=0.03,
                mid_slope_threshold=0.02,
                smooth_window=5,
                smooth_threshold=3,
            )
        elif detector_type == "ribbon_coherence":
            detector = RibbonCoherenceDetector(
                min_steady_days=15,
                convergence_threshold=0.015,
                ma_slope_threshold=0.01,
                price_to_ma_threshold=0.02,
                smooth_window=5,
                smooth_threshold=3,
            )
        elif detector_type == "composite_equilibrium":
            detector = CompositeEquilibriumDetector(
                box_window=20,
                ma_cohesion_threshold=0.015,
                quantile_width_threshold=0.04,
                quantile_window=20,
                cross_count_min=3,
                cross_window=15,
                volume_short=15,
                volume_long=60,
                volume_ratio=0.8,
                smooth_window=5,
                smooth_threshold=3,
            )
        elif detector_type == "anti_step":
            detector = AntiStepBoxDetector(
                box_window=20,
                r_threshold=0.4,
                center_dev_threshold=0.6,
                box_width_threshold=0.15,
                smooth_window=5,
                smooth_threshold=3,
            )
        else:
            raise ValueError(f"Unknown detector_type: {detector_type}")

        # Detect boxes
        df_with_boxes = detector.detect_box(df)

        # Calculate per-stock statistics
        per_stock = (
            df_with_boxes.group_by("ts_code")
            .agg(
                [
                    pl.count().alias("total_days"),
                    pl.col("is_box_candidate").sum().alias("box_days"),
                    (
                        pl.col("is_box_candidate").sum() / pl.count()
                    ).alias("box_ratio"),
                    pl.col("box_width").mean().alias("avg_box_width"),
                    pl.col("box_width").min().alias("min_box_width"),
                    pl.col("box_width").max().alias("max_box_width"),
                    pl.col("box_width").std().alias("std_box_width"),
                ]
            )
            .sort("box_ratio", descending=True)
        )

        return per_stock


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate box detection algorithms for Teapot pattern recognition"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="*",
        help="Optional list of stock codes to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/teapot/evaluation",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--box-window",
        type=int,
        default=40,
        help="Box window size (default: 40)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached data if available",
    )
    parser.add_argument(
        "--per-stock",
        action="store_true",
        help="Generate per-stock evaluation report",
    )
    parser.add_argument(
        "--detector-overlap",
        action="store_true",
        help="Generate detector overlap analysis (which detectors detected each box)",
    )
    parser.add_argument(
        "--plot-boxes",
        action="store_true",
        help="Generate K-line charts for detected boxes",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=100,
        help="Maximum number of box charts to generate (default: 100)",
    )
    parser.add_argument(
        "--context-days",
        type=int,
        default=30,
        help="Number of days before and after box to show in chart (default: 30)",
    )
    parser.add_argument(
        "--min-box-days",
        type=int,
        default=5,
        help="Minimum number of consecutive days for a valid box (default: 5)",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.warning(f"Failed to load config: {e}, using defaults")
        db_config = DatabaseConfig()

    # Create evaluator
    evaluator = BoxDetectorEvaluator(
        db_config=db_config,
        schema="quant",
        use_cache=args.use_cache,
    )

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate all detectors
    logger.info("Evaluating all detectors...")
    comparison_results = evaluator.evaluate_all_detectors(
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=args.symbols,
        box_window=args.box_window,
    )

    # Save comparison results (summary in root directory)
    comparison_file = output_dir / f"detector_comparison_{args.start_date}_{args.end_date}.csv"
    comparison_results.write_csv(comparison_file)
    logger.info(f"Saved comparison results to {comparison_file}")

    # Save individual detector results in separate directories
    # Iterate directly over Polars DataFrame to avoid type conversion issues
    for row in comparison_results.iter_rows(named=True):
        detector_name = row.get("detector_name", "unknown")
        detector_dir = output_dir / detector_name
        detector_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual detector summary
        # Create DataFrame from dict, ensuring proper types
        detector_summary_file = detector_dir / f"summary_{args.start_date}_{args.end_date}.csv"
        detector_summary = pl.DataFrame([row])
        detector_summary.write_csv(detector_summary_file)
        logger.info(f"Saved {detector_name} summary to {detector_summary_file}")

    # Print summary with formatted numbers
    print("\n" + "=" * 80)
    print("Box Detector Evaluation Summary")
    print("=" * 80)
    df_summary = comparison_results.to_pandas()
    
    # Format numeric columns with thousand separators
    numeric_cols = ['total_rows', 'box_count', 'stocks_with_boxes', 'total_stocks']
    for col in numeric_cols:
        if col in df_summary.columns:
            df_summary[col] = df_summary[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "")
    
    # Format float columns with 2-4 decimal places and thousand separators
    float_cols = ['box_ratio', 'avg_box_width', 'min_box_width', 'max_box_width', 
                  'std_box_width', 'stock_coverage', 'avg_box_duration', 'max_box_duration']
    for col in float_cols:
        if col in df_summary.columns:
            def format_float(x):
                if pd.isna(x) or not isinstance(x, (int, float)):
                    return ""
                # For small numbers (< 1), don't use thousand separator, just format decimal places
                if abs(x) < 1:
                    return f"{x:.4f}"
                else:
                    return f"{x:,.4f}"
            df_summary[col] = df_summary[col].apply(format_float)
    
    print(df_summary.to_string(index=False))
    print("=" * 80)

    # Generate per-stock evaluation if requested
    if args.per_stock:
        # Get list of detector types from comparison results
        detector_types = df_summary["detector_name"].tolist() if not df_summary.empty else ["dynamic_convergence"]
        
        for detector_type in detector_types:
            logger.info(f"Generating per-stock evaluation for {detector_type}...")
            per_stock_results = evaluator.evaluate_per_stock(
                start_date=args.start_date,
                end_date=args.end_date,
                symbols=args.symbols,
                box_window=args.box_window,
                detector_type=detector_type,
            )

            # Create detector-specific directory
            detector_dir = output_dir / detector_type
            detector_dir.mkdir(parents=True, exist_ok=True)

            per_stock_file = detector_dir / f"per_stock_evaluation_{args.start_date}_{args.end_date}.csv"
            per_stock_results.write_csv(per_stock_file)
            logger.info(f"Saved per-stock results to {per_stock_file}")
            
            # Also save formatted version for readability
            per_stock_file_formatted = detector_dir / f"per_stock_evaluation_{args.start_date}_{args.end_date}_formatted.csv"
            df_top_formatted = per_stock_results.to_pandas()
            
            # Format numeric columns
            numeric_cols = ['total_days', 'box_days']
            for col in numeric_cols:
                if col in df_top_formatted.columns:
                    df_top_formatted[col] = df_top_formatted[col].apply(
                        lambda x: f"{int(x):,}" if pd.notna(x) else ""
                    )
            
            # Format float columns
            float_cols = ['box_ratio', 'avg_box_width', 'min_box_width', 'max_box_width', 'std_box_width']
            for col in float_cols:
                if col in df_top_formatted.columns:
                    def format_float(x):
                        if pd.isna(x) or not isinstance(x, (int, float)):
                            return ""
                        if abs(x) < 1:
                            return f"{x:.4f}"
                        else:
                            return f"{x:,.4f}"
                    df_top_formatted[col] = df_top_formatted[col].apply(format_float)
            
            df_top_formatted.to_csv(per_stock_file_formatted, index=False)
            logger.info(f"Saved formatted per-stock results to {per_stock_file_formatted}")

            # Print top 20 stocks by box ratio with formatted numbers
            print("\n" + "=" * 80)
            print(f"Top 20 Stocks by Box Ratio - {detector_type}")
            print("=" * 80)
            top_stocks = per_stock_results.head(20)
            df_top = top_stocks.to_pandas()
            
            # Format numeric columns
            numeric_cols = ['total_days', 'box_days']
            for col in numeric_cols:
                if col in df_top.columns:
                    df_top[col] = df_top[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "")
            
            # Format float columns
            float_cols = ['box_ratio', 'avg_box_width', 'min_box_width', 'max_box_width', 'std_box_width']
            for col in float_cols:
                if col in df_top.columns:
                    def format_float(x):
                        if pd.isna(x) or not isinstance(x, (int, float)):
                            return ""
                        # For small numbers (< 1), don't use thousand separator, just format decimal places
                        if abs(x) < 1:
                            return f"{x:.4f}"
                        else:
                            return f"{x:,.4f}"
                    df_top[col] = df_top[col].apply(format_float)
            
            print(df_top.to_string(index=False))
            print("=" * 80)

    # Generate detector overlap analysis if requested
    if args.detector_overlap:
        logger.info("Generating detector overlap analysis...")
        overlap_results = evaluator.evaluate_detector_overlap(
            start_date=args.start_date,
            end_date=args.end_date,
            symbols=args.symbols,
            box_window=args.box_window,
        )

        if not overlap_results.is_empty():
            overlap_file = output_dir / f"detector_overlap_{args.start_date}_{args.end_date}.csv"
            overlap_results.write_csv(overlap_file)
            logger.info(f"Saved detector overlap results to {overlap_file}")

            # Print summary statistics
            print("\n" + "=" * 80)
            print("Detector Overlap Summary")
            print("=" * 80)
            
            # Count boxes detected by each detector
            detector_names = ["simple", "mean_reversion", "mean_reversion_v2", "hybrid", "hybrid_v2"]
            overlap_stats = []
            for name in detector_names:
                count = overlap_results.filter(pl.col(f"is_box_{name}")).height
                overlap_stats.append({
                    "detector": name,
                    "boxes_detected": count,
                })
            
            # Count boxes detected by multiple detectors
            multi_detector = overlap_results.filter(pl.col("detector_count") > 1).height
            overlap_stats.append({
                "detector": "multiple_detectors",
                "boxes_detected": multi_detector,
            })
            
            df_overlap = pl.DataFrame(overlap_stats).to_pandas()
            df_overlap["boxes_detected"] = df_overlap["boxes_detected"].apply(lambda x: f"{int(x):,}")
            print(df_overlap.to_string(index=False))
            print("=" * 80)
            
            # Show sample boxes with detector info
            print("\nSample boxes with detector information:")
            print("=" * 80)
            sample = overlap_results.select([
                "ts_code", "trade_date", "detected_by", "detector_count"
            ]).head(10).to_pandas()
            print(sample.to_string(index=False))
            print("=" * 80)

    # Generate box charts if requested
    if args.plot_boxes:
        logger.info("=" * 80)
        logger.info("Generating box charts...")
        logger.info("=" * 80)
        
        try:
            from datetime import timedelta
            from nq.trading.selector.teapot.box_detector import MeanReversionBoxDetectorV2
            from nq.trading.selector.teapot.box_visualizer import BoxVisualizer
            
            # Check if plotly is available
            try:
                import plotly.graph_objects as go
                logger.info("✓ Plotly is available")
            except ImportError:
                logger.error("✗ Plotly is not installed. Please install: pip install plotly")
                logger.error("Skipping chart generation.")
                return
            
            # Load data for visualization (need extended date range for context)
            start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
            extended_start = (start_dt - timedelta(days=args.context_days + 60)).strftime("%Y-%m-%d")
            extended_end = (end_dt + timedelta(days=args.context_days)).strftime("%Y-%m-%d")
            
            logger.info(f"Loading extended data range: {extended_start} to {extended_end}")
            extended_data = evaluator.data_loader.load_daily_data(
                start_date=extended_start,
                end_date=extended_end,
                symbols=args.symbols,
            )
            
            if extended_data.is_empty():
                logger.warning("✗ No extended data loaded for visualization")
                logger.warning("Please check:")
                logger.warning("  1. Database connection")
                logger.warning("  2. Date range has data")
                logger.warning("  3. Cache is available (if using --use-cache)")
            else:
                logger.info(f"✓ Loaded {len(extended_data)} rows for {extended_data['ts_code'].n_unique()} stocks")
                
                # Get list of detector types from comparison results
                detector_types = df_summary["detector_name"].tolist() if not df_summary.empty else ["dynamic_convergence"]
                
                # Generate charts for each detector
                for detector_type in detector_types:
                    logger.info(f"\n{'=' * 80}")
                    logger.info(f"Generating charts for {detector_type} detector...")
                    logger.info(f"{'=' * 80}")
                    
                    # Create detector based on type
                    detector = None
                    if detector_type == "ma_convergence":
                        detector = MovingAverageConvergenceBoxDetector(
                            box_window=20,
                            ma_periods=[5, 10, 20, 30],
                            convergence_threshold=0.02,
                            max_relative_height=0.10,
                            smooth_window=5,
                            smooth_threshold=3,
                        )
                    elif detector_type == "dynamic_convergence":
                        detector = DynamicConvergenceDetector(
                            box_window=30,
                            ma_periods=[5, 10, 20],
                            max_tightness=0.08,
                            volness_threshold=0.6,
                            smooth_window=5,
                            smooth_threshold=3,
                        )
                    elif detector_type == "keltner_squeeze":
                        detector = KeltnerSqueezeDetector(
                            box_window=20,
                            squeeze_threshold=0.06,
                            slope_threshold=0.015,
                            atr_multiplier=1.5,
                            volume_decay_threshold=0.8,
                            expansion_window=15,
                            stability_window=20,
                            stability_threshold=15,
                            smooth_window=5,
                            smooth_threshold=3,
                        )
                    elif detector_type == "expansion_anchor":
                        detector = ExpansionAnchorBoxDetector(
                            box_window=40,
                            squeeze_threshold=0.06,
                            slope_threshold=0.015,
                            atr_multiplier=1.5,
                            lookback_window=60,
                            expansion_window=20,
                            stability_window=20,
                            stability_threshold=12,  # Match evaluation script parameters
                            smooth_window=5,
                            smooth_threshold=3,
                        )
                    elif detector_type == "accurate":
                        detector = AccurateBoxDetector(
                            box_window=30,
                            price_tol=0.09,  # Match evaluation script parameters
                            min_cross_count=4,
                            min_touch_count=1,  # Require at least 1 touch on each boundary
                            touch_tolerance=0.03,  # 3% tolerance for touching boundary
                            smooth_window=4,  # Match evaluation script parameters
                            smooth_threshold=3,
                        )
                    elif detector_type == "balanced":
                        detector = BalancedBoxDetector(
                            box_window=30,
                            height_threshold=0.12,  # Match evaluation script parameters
                            symmetry_threshold=0.15,
                            uniformity_threshold=0.45,
                            slope_threshold=0.015,
                            smooth_window=5,
                            smooth_threshold=3,
                        )
                    elif detector_type == "dense_area":
                        detector = DenseAreaBoxDetector(
                            box_window=40,
                            dense_threshold=0.05,
                            quantile_high=0.8,
                            quantile_low=0.2,
                            vol_stability_threshold=0.03,
                            mid_slope_threshold=0.02,
                            smooth_window=5,
                            smooth_threshold=3,
                        )
                    elif detector_type == "ribbon_coherence":
                        detector = RibbonCoherenceDetector(
                            min_steady_days=15,
                            convergence_threshold=0.015,
                            ma_slope_threshold=0.01,
                            price_to_ma_threshold=0.02,
                            smooth_window=5,
                            smooth_threshold=3,
                        )
                    elif detector_type == "composite_equilibrium":
                        detector = CompositeEquilibriumDetector(
                            box_window=20,
                            ma_cohesion_threshold=0.015,
                            quantile_width_threshold=0.04,
                            quantile_window=20,
                            cross_count_min=3,
                            cross_window=15,
                            volume_short=15,
                            volume_long=60,
                            volume_ratio=0.8,
                            smooth_window=5,
                            smooth_threshold=3,
                        )
                    elif detector_type == "anti_step":
                        detector = AntiStepBoxDetector(
                            box_window=20,
                            r_threshold=0.4,
                            center_dev_threshold=0.6,
                            box_width_threshold=0.15,
                            smooth_window=5,
                            smooth_threshold=3,
                        )
                    else:
                        logger.warning(f"✗ Unknown detector type: {detector_type}, skipping...")
                        continue
                    
                    logger.info(f"Running {detector_type} detector...")
                    logger.info(f"Extracting boxes (min_box_days={args.min_box_days})...")
                    boxes_df = evaluator.extract_boxes_with_dates(
                        df=extended_data,
                        detector=detector,
                        detector_name=detector_type,
                        min_box_days=args.min_box_days,
                    )
                    
                    logger.info(f"Extracted {len(boxes_df)} boxes with min_box_days={args.min_box_days}")
                    
                    if boxes_df.is_empty():
                        logger.warning("✗ No boxes extracted with current min_box_days")
                        logger.warning(f"Trying with min_box_days=3 to find available boxes...")
                        
                        # Try with lower min_box_days to find boxes
                        boxes_df = evaluator.extract_boxes_with_dates(
                            df=extended_data,
                            detector=detector,
                            detector_name=detector_type,  # Fixed: use detector_type instead of hardcoded name
                            min_box_days=3,
                        )
                        
                        if not boxes_df.is_empty():
                            logger.info(f"✓ Found {len(boxes_df)} boxes with min_box_days=3")
                            
                            # Show box duration statistics
                            if "box_duration_days" in boxes_df.columns:
                                duration_stats = boxes_df["box_duration_days"]
                                logger.info(f"\nBox duration statistics:")
                                logger.info(f"  Min: {duration_stats.min()} days")
                                logger.info(f"  Max: {duration_stats.max()} days")
                                logger.info(f"  Mean: {duration_stats.mean():.1f} days")
                                logger.info(f"  Median: {duration_stats.median():.1f} days")
                                
                                # Count boxes by duration ranges
                                logger.info(f"\nBox count by duration:")
                                logger.info(f"  3-5 days: {boxes_df.filter((pl.col('box_duration_days') >= 3) & (pl.col('box_duration_days') <= 5)).height}")
                                logger.info(f"  6-10 days: {boxes_df.filter((pl.col('box_duration_days') >= 6) & (pl.col('box_duration_days') <= 10)).height}")
                                logger.info(f"  11-20 days: {boxes_df.filter((pl.col('box_duration_days') >= 11) & (pl.col('box_duration_days') <= 20)).height}")
                                logger.info(f"  >20 days: {boxes_df.filter(pl.col('box_duration_days') > 20).height}")
                                
                                logger.warning(f"\nNote: Using min_box_days=3 instead of {args.min_box_days}")
                                logger.warning(f"Consider using --min-box-days 3 for better results")
                        else:
                            logger.error("✗ No boxes found even with min_box_days=3")
                            logger.error("Box detection may need parameter adjustment")
                            boxes_df = pl.DataFrame()  # Ensure it's empty
                    
                    # Generate charts if we have boxes
                    if not boxes_df.is_empty():
                        logger.info(f"✓ Extracted {len(boxes_df)} boxes")
                        if "box_duration_days" in boxes_df.columns:
                            avg_duration = boxes_df["box_duration_days"].mean()
                            logger.info(f"  Average box duration: {avg_duration:.1f} days")
                        
                        # Create detector-specific directory for charts
                        detector_dir = output_dir / detector_type
                        detector_dir.mkdir(parents=True, exist_ok=True)
                        chart_dir = detector_dir / "charts"
                        
                        logger.info(f"Initializing visualizer (output: {chart_dir})...")
                        visualizer = BoxVisualizer(
                            output_dir=chart_dir,
                        )
                        
                        # Generate charts
                        logger.info(f"Generating charts (max: {args.max_plots})...")
                        chart_results = visualizer.batch_plot_boxes(
                            boxes_df=boxes_df,
                            kline_df=extended_data,
                            context_days=args.context_days,
                            max_boxes=args.max_plots,
                        )
                        
                        if chart_results.is_empty():
                            logger.warning("✗ No chart results generated")
                        else:
                            # Save chart results in detector-specific directory
                            chart_results_file = detector_dir / f"box_charts_{args.start_date}_{args.end_date}.csv"
                            chart_results.write_csv(chart_results_file)
                            logger.info(f"✓ Saved chart results to {chart_results_file}")
                            
                            # Create summary HTML with all chart images
                            summary_html = visualizer.create_summary_html(
                                results_df=chart_results,
                                output_file=detector_dir / f"summary_{args.start_date}_{args.end_date}.html",
                                title=f"Box Detection Charts Summary - {detector_type}",
                            )
                            if summary_html:
                                logger.info(f"✓ Created summary HTML: {summary_html}")
                            
                            # Print summary
                            success_count = chart_results.filter(pl.col("status") == "success").height
                            failed_count = chart_results.filter(pl.col("status") != "success").height
                            
                            print("\n" + "=" * 80)
                            print(f"Box Chart Generation Summary - {detector_type}")
                            print("=" * 80)
                            print(f"Total boxes extracted: {len(boxes_df)}")
                            print(f"Charts generated (success): {success_count}")
                            print(f"Charts failed: {failed_count}")
                            print(f"Charts saved to: {chart_dir}")
                            if summary_html:
                                print(f"Summary HTML: {summary_html}")
                            print("=" * 80)
                            
                            if success_count == 0:
                                logger.warning("No charts were successfully generated!")
                                logger.warning("Check chart_results CSV for error details")
                            else:
                                logger.info(f"✓ Successfully generated {success_count} charts for {detector_type}")
                    else:
                        logger.warning(f"✗ No boxes found for {detector_type} detector")
                            
        except ImportError as e:
            logger.error(f"✗ Failed to import visualization modules: {e}")
            logger.error("Please install plotly: pip install plotly")
        except Exception as e:
            logger.error(f"✗ Failed to generate box charts: {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
