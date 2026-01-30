# -*- coding: utf-8 -*-
"""
Debug script to analyze why box detectors are not detecting boxes.

This script helps diagnose why certain detectors (e.g., AccurateBoxDetector, 
ExpansionAnchorBoxDetector) are not finding any boxes by:
1. Checking intermediate conditions
2. Showing statistics for each condition
3. Identifying which conditions are too strict
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import polars as pl

from nq.config import DatabaseConfig, load_config
from nq.data.processor.teapot import TeapotDataLoader
from nq.trading.selector.teapot.box_detector_accurate import AccurateBoxDetector
from nq.trading.selector.teapot.box_detector_keltner_squeeze import (
    ExpansionAnchorBoxDetector,
    KeltnerSqueezeDetector,
)
from nq.trading.selector.teapot.box_detector_balanced import BalancedBoxDetector
from nq.trading.selector.teapot.box_detector_dense_area import DenseAreaBoxDetector
from nq.trading.selector.teapot.box_detector_ribbon_coherence import (
    RibbonCoherenceDetector,
)
from nq.trading.selector.teapot.box_detector_composite_equilibrium import (
    CompositeEquilibriumDetector,
)
from nq.trading.selector.teapot.box_detector_anti_step import AntiStepBoxDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def debug_accurate_detector(
    df: pl.DataFrame,
    box_window: int = 30,
    price_tol: float = 0.1372,  # Updated to match evaluation script
    min_cross_count: int = 3,
) -> Dict:
    """Debug AccurateBoxDetector to see why no boxes are detected."""
    logger.info("=" * 80)
    logger.info("Debugging AccurateBoxDetector")
    logger.info("=" * 80)
    
    detector = AccurateBoxDetector(
        box_window=box_window,
        price_tol=price_tol,
        min_cross_count=min_cross_count,
    )
    
    # Run detection
    df_result = detector.detect_box(df)
    
    # Check intermediate conditions
    total_rows = len(df_result)
    
    # Condition 1: Cross count
    cross_count_ok = df_result.filter(pl.col("cross_count") >= min_cross_count)
    logger.info(f"Condition 1: Cross count >= {min_cross_count}")
    logger.info(f"  Satisfied: {len(cross_count_ok)} / {total_rows} ({len(cross_count_ok)/total_rows*100:.2f}%)")
    if len(cross_count_ok) > 0:
        logger.info(f"  Cross count stats: min={cross_count_ok['cross_count'].min()}, "
                   f"max={cross_count_ok['cross_count'].max()}, "
                   f"mean={cross_count_ok['cross_count'].mean():.2f}")
    
    # Condition 2: Upper distance
    upper_dist_ok = df_result.filter(pl.col("upper_dist") < price_tol)
    logger.info(f"Condition 2: Upper distance < {price_tol}")
    logger.info(f"  Satisfied: {len(upper_dist_ok)} / {total_rows} ({len(upper_dist_ok)/total_rows*100:.2f}%)")
    if len(upper_dist_ok) > 0:
        logger.info(f"  Upper dist stats: min={upper_dist_ok['upper_dist'].min():.4f}, "
                   f"max={upper_dist_ok['upper_dist'].max():.4f}, "
                   f"mean={upper_dist_ok['upper_dist'].mean():.4f}")
    
    # Condition 3: Lower distance
    lower_dist_ok = df_result.filter(pl.col("lower_dist") < price_tol)
    logger.info(f"Condition 3: Lower distance < {price_tol}")
    logger.info(f"  Satisfied: {len(lower_dist_ok)} / {total_rows} ({len(lower_dist_ok)/total_rows*100:.2f}%)")
    if len(lower_dist_ok) > 0:
        logger.info(f"  Lower dist stats: min={lower_dist_ok['lower_dist'].min():.4f}, "
                   f"max={lower_dist_ok['lower_dist'].max():.4f}, "
                   f"mean={lower_dist_ok['lower_dist'].mean():.4f}")
    
    # Combined conditions
    all_conditions = df_result.filter(
        (pl.col("cross_count") >= min_cross_count)
        & (pl.col("upper_dist") < price_tol)
        & (pl.col("lower_dist") < price_tol)
        & pl.col("pivot_line").is_not_null()
        & (pl.col("pivot_line") > 0)
    )
    logger.info(f"All conditions satisfied (is_confirmed):")
    logger.info(f"  Count: {len(all_conditions)} / {total_rows} ({len(all_conditions)/total_rows*100:.2f}%)")
    
    # Final result (after back-propagation)
    final_boxes = df_result.filter(pl.col("is_box_candidate"))
    logger.info(f"Final boxes (after back-propagation):")
    logger.info(f"  Count: {len(final_boxes)} / {total_rows} ({len(final_boxes)/total_rows*100:.2f}%)")
    
    # Show distribution of intermediate values
    logger.info(f"Intermediate value distributions:")
    logger.info(f"  Cross count: min={df_result['cross_count'].min()}, "
               f"max={df_result['cross_count'].max()}, "
               f"mean={df_result['cross_count'].mean():.2f}, "
               f"median={df_result['cross_count'].median():.2f}")
    logger.info(f"  Upper dist: min={df_result['upper_dist'].min():.4f}, "
               f"max={df_result['upper_dist'].max():.4f}, "
               f"mean={df_result['upper_dist'].mean():.4f}, "
               f"median={df_result['upper_dist'].median():.4f}")
    logger.info(f"  Lower dist: min={df_result['lower_dist'].min():.4f}, "
               f"max={df_result['lower_dist'].max():.4f}, "
               f"mean={df_result['lower_dist'].mean():.4f}, "
               f"median={df_result['lower_dist'].median():.4f}")
    
    # Suggest parameter adjustments
    logger.info(f"" + "=" * 80)
    logger.info("Parameter Adjustment Suggestions:")
    logger.info("=" * 80)
    
    # Check percentiles
    cross_p50 = df_result['cross_count'].quantile(0.5)
    cross_p75 = df_result['cross_count'].quantile(0.75)
    upper_p50 = df_result['upper_dist'].quantile(0.5)
    upper_p75 = df_result['upper_dist'].quantile(0.75)
    lower_p50 = df_result['lower_dist'].quantile(0.5)
    lower_p75 = df_result['lower_dist'].quantile(0.75)
    
    logger.info(f"If you want to capture ~50% of data:")
    logger.info(f"  min_cross_count: {int(cross_p50)} (current: {min_cross_count})")
    logger.info(f"  price_tol: {upper_p50:.4f} (current: {price_tol})")
    
    logger.info(f"If you want to capture ~75% of data:")
    logger.info(f"  min_cross_count: {int(cross_p75)} (current: {min_cross_count})")
    logger.info(f"  price_tol: {max(upper_p75, lower_p75):.4f} (current: {price_tol})")
    
    return {
        "total_rows": total_rows,
        "cross_count_ok": len(cross_count_ok),
        "upper_dist_ok": len(upper_dist_ok),
        "lower_dist_ok": len(lower_dist_ok),
        "all_conditions_ok": len(all_conditions),
        "final_boxes": len(final_boxes),
    }


def debug_expansion_anchor_detector(
    df: pl.DataFrame,
    box_window: int = 40,
    squeeze_threshold: float = 0.06,
    slope_threshold: float = 0.015,
    stability_threshold: int = 15,
    stability_window: int = 20,
) -> Dict:
    """Debug ExpansionAnchorBoxDetector to see why no boxes are detected."""
    logger.info("=" * 80)
    logger.info("Debugging ExpansionAnchorBoxDetector")
    logger.info("=" * 80)
    
    detector = ExpansionAnchorBoxDetector(
        box_window=box_window,
        squeeze_threshold=squeeze_threshold,
        slope_threshold=slope_threshold,
        stability_threshold=stability_threshold,
        stability_window=stability_window,
    )
    
    # Run detection
    df_result = detector.detect_box(df)
    
    total_rows = len(df_result)
    
    # Condition 1: Squeeze ratio
    squeeze_ok = df_result.filter(pl.col("squeeze_ratio") < squeeze_threshold)
    logger.info(f"Condition 1: Squeeze ratio < {squeeze_threshold}")
    logger.info(f"  Satisfied: {len(squeeze_ok)} / {total_rows} ({len(squeeze_ok)/total_rows*100:.2f}%)")
    if len(squeeze_ok) > 0:
        logger.info(f"  Squeeze ratio stats: min={squeeze_ok['squeeze_ratio'].min():.4f}, "
                   f"max={squeeze_ok['squeeze_ratio'].max():.4f}, "
                   f"mean={squeeze_ok['squeeze_ratio'].mean():.4f}")
    
    # Condition 2: Slope
    slope_ok = df_result.filter(pl.col("mid_slope") < slope_threshold)
    logger.info(f"Condition 2: Mid slope < {slope_threshold}")
    logger.info(f"  Satisfied: {len(slope_ok)} / {total_rows} ({len(slope_ok)/total_rows*100:.2f}%)")
    if len(slope_ok) > 0:
        logger.info(f"  Mid slope stats: min={slope_ok['mid_slope'].min():.4f}, "
                   f"max={slope_ok['mid_slope'].max():.4f}, "
                   f"mean={slope_ok['mid_slope'].mean():.4f}")
    
    # Condition 3: Price in channel
    price_in_channel = df_result.filter(
        (pl.col("close") < pl.col("box_h")) & (pl.col("close") > pl.col("box_l"))
    )
    logger.info(f"Condition 3: Price in channel")
    logger.info(f"  Satisfied: {len(price_in_channel)} / {total_rows} ({len(price_in_channel)/total_rows*100:.2f}%)")
    
    # Condition 4: Stability count
    stability_ok = df_result.filter(pl.col("stability_count") >= stability_threshold)
    logger.info(f"Condition 4: Stability count >= {stability_threshold}")
    logger.info(f"  Satisfied: {len(stability_ok)} / {total_rows} ({len(stability_ok)/total_rows*100:.2f}%)")
    if len(stability_ok) > 0:
        logger.info(f"  Stability count stats: min={stability_ok['stability_count'].min()}, "
                   f"max={stability_ok['stability_count'].max()}, "
                   f"mean={stability_ok['stability_count'].mean():.2f}")
    
    # Combined conditions
    all_conditions = df_result.filter(
        (pl.col("squeeze_ratio") < squeeze_threshold)
        & (pl.col("mid_slope") < slope_threshold)
        & (pl.col("close") < pl.col("box_h"))
        & (pl.col("close") > pl.col("box_l"))
        & (pl.col("stability_count") >= stability_threshold)
        & pl.col("box_h").is_not_null()
        & pl.col("box_l").is_not_null()
        & (pl.col("box_h") > pl.col("box_l"))
    )
    logger.info(f"All conditions satisfied (is_squeeze_detected):")
    logger.info(f"  Count: {len(all_conditions)} / {total_rows} ({len(all_conditions)/total_rows*100:.2f}%)")
    
    # Final result (after back-propagation)
    final_boxes = df_result.filter(pl.col("is_box_candidate"))
    logger.info(f"Final boxes (after back-propagation):")
    logger.info(f"  Count: {len(final_boxes)} / {total_rows} ({len(final_boxes)/total_rows*100:.2f}%)")
    
    # Show distribution of intermediate values
    logger.info(f"Intermediate value distributions:")
    logger.info(f"  Squeeze ratio: min={df_result['squeeze_ratio'].min():.4f}, "
               f"max={df_result['squeeze_ratio'].max():.4f}, "
               f"mean={df_result['squeeze_ratio'].mean():.4f}, "
               f"median={df_result['squeeze_ratio'].median():.4f}")
    logger.info(f"  Mid slope: min={df_result['mid_slope'].min():.4f}, "
               f"max={df_result['mid_slope'].max():.4f}, "
               f"mean={df_result['mid_slope'].mean():.4f}, "
               f"median={df_result['mid_slope'].median():.4f}")
    logger.info(f"  Stability count: min={df_result['stability_count'].min()}, "
               f"max={df_result['stability_count'].max()}, "
               f"mean={df_result['stability_count'].mean():.2f}, "
               f"median={df_result['stability_count'].median():.2f}")
    
    # Suggest parameter adjustments
    logger.info(f"" + "=" * 80)
    logger.info("Parameter Adjustment Suggestions:")
    logger.info("=" * 80)
    
    squeeze_p50 = df_result['squeeze_ratio'].quantile(0.5)
    squeeze_p75 = df_result['squeeze_ratio'].quantile(0.75)
    slope_p50 = df_result['mid_slope'].quantile(0.5)
    slope_p75 = df_result['mid_slope'].quantile(0.75)
    stability_p50 = df_result['stability_count'].quantile(0.5)
    stability_p75 = df_result['stability_count'].quantile(0.75)
    
    logger.info(f"If you want to capture ~50% of data:")
    logger.info(f"  squeeze_threshold: {squeeze_p50:.4f} (current: {squeeze_threshold})")
    logger.info(f"  slope_threshold: {slope_p50:.4f} (current: {slope_threshold})")
    logger.info(f"  stability_threshold: {int(stability_p50)} (current: {stability_threshold})")
    
    logger.info(f"If you want to capture ~75% of data:")
    logger.info(f"  squeeze_threshold: {squeeze_p75:.4f} (current: {squeeze_threshold})")
    logger.info(f"  slope_threshold: {slope_p75:.4f} (current: {slope_threshold})")
    logger.info(f"  stability_threshold: {int(stability_p75)} (current: {stability_threshold})")
    
    return {
        "total_rows": total_rows,
        "squeeze_ok": len(squeeze_ok),
        "slope_ok": len(slope_ok),
        "price_in_channel": len(price_in_channel),
        "stability_ok": len(stability_ok),
        "all_conditions_ok": len(all_conditions),
        "final_boxes": len(final_boxes),
    }


def debug_balanced_detector(
    df: pl.DataFrame,
    box_window: int = 30,
    height_threshold: float = 0.07,
    symmetry_threshold: float = 0.1,
    uniformity_threshold: float = 0.35,
    slope_threshold: float = 0.01,
) -> Dict:
    """Debug BalancedBoxDetector to see why no boxes are detected."""
    logger.info("=" * 80)
    logger.info("Debugging BalancedBoxDetector")
    logger.info("=" * 80)
    
    detector = BalancedBoxDetector(
        box_window=box_window,
        height_threshold=height_threshold,
        symmetry_threshold=symmetry_threshold,
        uniformity_threshold=uniformity_threshold,
        slope_threshold=slope_threshold,
    )
    
    # Run detection
    df_result = detector.detect_box(df)
    
    total_rows = len(df_result)
    
    # Condition 1: Box height
    height_ok = df_result.filter(pl.col("box_height") < height_threshold)
    logger.info(f"Condition 1: Box height < {height_threshold}")
    logger.info(f"  Satisfied: {len(height_ok)} / {total_rows} ({len(height_ok)/total_rows*100:.2f}%)")
    if len(height_ok) > 0:
        logger.info(f"  Box height stats: min={height_ok['box_height'].min():.4f}, "
                   f"max={height_ok['box_height'].max():.4f}, "
                   f"mean={height_ok['box_height'].mean():.4f}")
    
    # Condition 2: Symmetry error
    symmetry_ok = df_result.filter(pl.col("symmetry_error") < symmetry_threshold)
    logger.info(f"Condition 2: Symmetry error < {symmetry_threshold}")
    logger.info(f"  Satisfied: {len(symmetry_ok)} / {total_rows} ({len(symmetry_ok)/total_rows*100:.2f}%)")
    if len(symmetry_ok) > 0:
        logger.info(f"  Symmetry error stats: min={symmetry_ok['symmetry_error'].min():.4f}, "
                   f"max={symmetry_ok['symmetry_error'].max():.4f}, "
                   f"mean={symmetry_ok['symmetry_error'].mean():.4f}")
    
    # Condition 3: Volatility uniformity
    uniformity_ok = df_result.filter(pl.col("vol_cv") < uniformity_threshold)
    logger.info(f"Condition 3: Volatility CV < {uniformity_threshold}")
    logger.info(f"  Satisfied: {len(uniformity_ok)} / {total_rows} ({len(uniformity_ok)/total_rows*100:.2f}%)")
    if len(uniformity_ok) > 0:
        logger.info(f"  Volatility CV stats: min={uniformity_ok['vol_cv'].min():.4f}, "
                   f"max={uniformity_ok['vol_cv'].max():.4f}, "
                   f"mean={uniformity_ok['vol_cv'].mean():.4f}")
    
    # Condition 4: Slope
    slope_ok = df_result.filter(pl.col("mid_slope") < slope_threshold)
    logger.info(f"Condition 4: Mid slope < {slope_threshold}")
    logger.info(f"  Satisfied: {len(slope_ok)} / {total_rows} ({len(slope_ok)/total_rows*100:.2f}%)")
    if len(slope_ok) > 0:
        logger.info(f"  Mid slope stats: min={slope_ok['mid_slope'].min():.4f}, "
                   f"max={slope_ok['mid_slope'].max():.4f}, "
                   f"mean={slope_ok['mid_slope'].mean():.4f}")
    
    # Combined conditions
    all_conditions = df_result.filter(
        (pl.col("box_height") < height_threshold)
        & (pl.col("symmetry_error") < symmetry_threshold)
        & (pl.col("vol_cv") < uniformity_threshold)
        & (pl.col("mid_slope") < slope_threshold)
        & pl.col("box_h").is_not_null()
        & pl.col("box_l").is_not_null()
        & (pl.col("box_h") > pl.col("box_l"))
        & pl.col("box_mid").is_not_null()
        & (pl.col("box_mid") > 0)
    )
    logger.info(f"All conditions satisfied (is_box_candidate):")
    logger.info(f"  Count: {len(all_conditions)} / {total_rows} ({len(all_conditions)/total_rows*100:.2f}%)")
    
    # Final result (after smoothing if enabled)
    final_boxes = df_result.filter(pl.col("is_box_candidate"))
    logger.info(f"Final boxes (after smoothing):")
    logger.info(f"  Count: {len(final_boxes)} / {total_rows} ({len(final_boxes)/total_rows*100:.2f}%)")
    
    # Show distribution of intermediate values
    logger.info(f"Intermediate value distributions:")
    logger.info(f"  Box height: min={df_result['box_height'].min():.4f}, "
               f"max={df_result['box_height'].max():.4f}, "
               f"mean={df_result['box_height'].mean():.4f}, "
               f"median={df_result['box_height'].median():.4f}")
    logger.info(f"  Symmetry error: min={df_result['symmetry_error'].min():.4f}, "
               f"max={df_result['symmetry_error'].max():.4f}, "
               f"mean={df_result['symmetry_error'].mean():.4f}, "
               f"median={df_result['symmetry_error'].median():.4f}")
    logger.info(f"  Volatility CV: min={df_result['vol_cv'].min():.4f}, "
               f"max={df_result['vol_cv'].max():.4f}, "
               f"mean={df_result['vol_cv'].mean():.4f}, "
               f"median={df_result['vol_cv'].median():.4f}")
    logger.info(f"  Mid slope: min={df_result['mid_slope'].min():.4f}, "
               f"max={df_result['mid_slope'].max():.4f}, "
               f"mean={df_result['mid_slope'].mean():.4f}, "
               f"median={df_result['mid_slope'].median():.4f}")
    
    # Suggest parameter adjustments
    logger.info(f"" + "=" * 80)
    logger.info("Parameter Adjustment Suggestions:")
    logger.info("=" * 80)
    
    height_p50 = df_result['box_height'].quantile(0.5)
    height_p75 = df_result['box_height'].quantile(0.75)
    symmetry_p50 = df_result['symmetry_error'].quantile(0.5)
    symmetry_p75 = df_result['symmetry_error'].quantile(0.75)
    vol_cv_p50 = df_result['vol_cv'].quantile(0.5)
    vol_cv_p75 = df_result['vol_cv'].quantile(0.75)
    slope_p50 = df_result['mid_slope'].quantile(0.5)
    slope_p75 = df_result['mid_slope'].quantile(0.75)
    
    logger.info(f"If you want to capture ~50% of data:")
    logger.info(f"  height_threshold: {height_p50:.4f} (current: {height_threshold})")
    logger.info(f"  symmetry_threshold: {symmetry_p50:.4f} (current: {symmetry_threshold})")
    logger.info(f"  uniformity_threshold: {vol_cv_p50:.4f} (current: {uniformity_threshold})")
    logger.info(f"  slope_threshold: {slope_p50:.4f} (current: {slope_threshold})")
    
    logger.info(f"If you want to capture ~75% of data:")
    logger.info(f"  height_threshold: {height_p75:.4f} (current: {height_threshold})")
    logger.info(f"  symmetry_threshold: {symmetry_p75:.4f} (current: {symmetry_threshold})")
    logger.info(f"  uniformity_threshold: {vol_cv_p75:.4f} (current: {uniformity_threshold})")
    logger.info(f"  slope_threshold: {slope_p75:.4f} (current: {slope_threshold})")
    
    return {
        "total_rows": total_rows,
        "height_ok": len(height_ok),
        "symmetry_ok": len(symmetry_ok),
        "uniformity_ok": len(uniformity_ok),
        "slope_ok": len(slope_ok),
        "all_conditions_ok": len(all_conditions),
        "final_boxes": len(final_boxes),
    }


def debug_dense_area_detector(
    df: pl.DataFrame,
    box_window: int = 40,
    dense_threshold: float = 0.05,
    quantile_high: float = 0.8,
    quantile_low: float = 0.2,
    vol_stability_threshold: float = 0.03,
    mid_slope_threshold: float = 0.02,
) -> Dict:
    """Debug DenseAreaBoxDetector to see why no boxes are detected."""
    logger.info("=" * 80)
    logger.info("Debugging DenseAreaBoxDetector")
    logger.info("=" * 80)

    detector = DenseAreaBoxDetector(
        box_window=box_window,
        dense_threshold=dense_threshold,
        quantile_high=quantile_high,
        quantile_low=quantile_low,
        vol_stability_threshold=vol_stability_threshold,
        mid_slope_threshold=mid_slope_threshold,
    )

    df_result = detector.detect_box(df)
    total_rows = len(df_result)

    # Condition 1: Dense width
    dense_ok = df_result.filter(pl.col("dense_width") < dense_threshold)
    logger.info(f"Condition 1: Dense width < {dense_threshold}")
    logger.info(f"  Satisfied: {len(dense_ok)} / {total_rows} ({len(dense_ok)/total_rows*100:.2f}%)")
    if len(dense_ok) > 0:
        logger.info(f"  Dense width stats: min={dense_ok['dense_width'].min():.4f}, "
                   f"max={dense_ok['dense_width'].max():.4f}, "
                   f"mean={dense_ok['dense_width'].mean():.4f}")

    # Condition 2: Vol stability
    vol_ok = df_result.filter(pl.col("vol_stability") < vol_stability_threshold)
    logger.info(f"Condition 2: Vol stability < {vol_stability_threshold}")
    logger.info(f"  Satisfied: {len(vol_ok)} / {total_rows} ({len(vol_ok)/total_rows*100:.2f}%)")
    if len(vol_ok) > 0:
        logger.info(f"  Vol stability stats: min={vol_ok['vol_stability'].min():.4f}, "
                   f"max={vol_ok['vol_stability'].max():.4f}, "
                   f"mean={vol_ok['vol_stability'].mean():.4f}")

    # Condition 3: Mid slope
    slope_ok = df_result.filter(pl.col("mid_slope_10") < mid_slope_threshold)
    logger.info(f"Condition 3: Mid slope 10d < {mid_slope_threshold}")
    logger.info(f"  Satisfied: {len(slope_ok)} / {total_rows} ({len(slope_ok)/total_rows*100:.2f}%)")
    if len(slope_ok) > 0:
        logger.info(f"  Mid slope stats: min={slope_ok['mid_slope_10'].min():.4f}, "
                   f"max={slope_ok['mid_slope_10'].max():.4f}, "
                   f"mean={slope_ok['mid_slope_10'].mean():.4f}")

    all_conditions = df_result.filter(
        (pl.col("dense_width") < dense_threshold)
        & (pl.col("vol_stability") < vol_stability_threshold)
        & (pl.col("mid_slope_10") < mid_slope_threshold)
        & pl.col("dense_h").is_not_null()
        & pl.col("dense_l").is_not_null()
        & (pl.col("dense_h") > pl.col("dense_l"))
    )
    logger.info("All conditions satisfied (is_box_candidate):")
    logger.info(f"  Count: {len(all_conditions)} / {total_rows} ({len(all_conditions)/total_rows*100:.2f}%)")

    final_boxes = df_result.filter(pl.col("is_box_candidate"))
    logger.info("Final boxes (after smoothing):")
    logger.info(f"  Count: {len(final_boxes)} / {total_rows} ({len(final_boxes)/total_rows*100:.2f}%)")

    logger.info("Intermediate value distributions:")
    logger.info(f"  Dense width: min={df_result['dense_width'].min():.4f}, "
               f"max={df_result['dense_width'].max():.4f}, "
               f"mean={df_result['dense_width'].mean():.4f}, "
               f"median={df_result['dense_width'].median():.4f}")
    logger.info(f"  Vol stability: min={df_result['vol_stability'].min():.4f}, "
               f"max={df_result['vol_stability'].max():.4f}, "
               f"mean={df_result['vol_stability'].mean():.4f}, "
               f"median={df_result['vol_stability'].median():.4f}")
    logger.info(f"  Mid slope 10d: min={df_result['mid_slope_10'].min():.4f}, "
               f"max={df_result['mid_slope_10'].max():.4f}, "
               f"mean={df_result['mid_slope_10'].mean():.4f}, "
               f"median={df_result['mid_slope_10'].median():.4f}")

    logger.info("=" * 80)
    logger.info("Parameter Adjustment Suggestions:")
    logger.info("=" * 80)
    dense_p50 = df_result["dense_width"].quantile(0.5)
    dense_p75 = df_result["dense_width"].quantile(0.75)
    vol_p50 = df_result["vol_stability"].quantile(0.5)
    vol_p75 = df_result["vol_stability"].quantile(0.75)
    slope_p50 = df_result["mid_slope_10"].quantile(0.5)
    slope_p75 = df_result["mid_slope_10"].quantile(0.75)
    logger.info(f"For ~50%% recall: dense_threshold={dense_p50:.4f}, vol_stability_threshold={vol_p50:.4f}, mid_slope_threshold={slope_p50:.4f}")
    logger.info(f"For ~75%% recall: dense_threshold={dense_p75:.4f}, vol_stability_threshold={vol_p75:.4f}, mid_slope_threshold={slope_p75:.4f}")

    return {
        "total_rows": total_rows,
        "dense_ok": len(dense_ok),
        "vol_ok": len(vol_ok),
        "slope_ok": len(slope_ok),
        "all_conditions_ok": len(all_conditions),
        "final_boxes": len(final_boxes),
    }


def debug_ribbon_coherence_detector(
    df: pl.DataFrame,
    min_steady_days: int = 15,
    convergence_threshold: float = 0.015,
    ma_slope_threshold: float = 0.01,
    price_to_ma_threshold: float = 0.02,
) -> Dict:
    """Debug RibbonCoherenceDetector: MA ribbon coherence and stable-point ratio."""
    logger.info("=" * 80)
    logger.info("Debugging RibbonCoherenceDetector")
    logger.info("=" * 80)

    detector = RibbonCoherenceDetector(
        min_steady_days=min_steady_days,
        convergence_threshold=convergence_threshold,
        ma_slope_threshold=ma_slope_threshold,
        price_to_ma_threshold=price_to_ma_threshold,
    )

    df_result = detector.detect_box(df)
    total_rows = len(df_result)

    # Condition 1: ma_cv (MA convergence)
    ma_cv_ok = df_result.filter(pl.col("ma_cv") < convergence_threshold)
    logger.info(f"Condition 1: ma_cv < {convergence_threshold}")
    logger.info(f"  Satisfied: {len(ma_cv_ok)} / {total_rows} ({len(ma_cv_ok)/total_rows*100:.2f}%)")
    if len(ma_cv_ok) > 0:
        logger.info(f"  ma_cv stats: min={ma_cv_ok['ma_cv'].min():.4f}, "
                   f"max={ma_cv_ok['ma_cv'].max():.4f}, mean={ma_cv_ok['ma_cv'].mean():.4f}")

    # Condition 2: ma_slope
    ma_slope_ok = df_result.filter(pl.col("ma_slope") < ma_slope_threshold)
    logger.info(f"Condition 2: ma_slope < {ma_slope_threshold}")
    logger.info(f"  Satisfied: {len(ma_slope_ok)} / {total_rows} ({len(ma_slope_ok)/total_rows*100:.2f}%)")
    if len(ma_slope_ok) > 0:
        logger.info(f"  ma_slope stats: min={ma_slope_ok['ma_slope'].min():.4f}, "
                   f"max={ma_slope_ok['ma_slope'].max():.4f}, mean={ma_slope_ok['ma_slope'].mean():.4f}")

    # Condition 3: price_dev
    price_dev_ok = df_result.filter(pl.col("price_dev") < price_to_ma_threshold)
    logger.info(f"Condition 3: price_dev < {price_to_ma_threshold}")
    logger.info(f"  Satisfied: {len(price_dev_ok)} / {total_rows} ({len(price_dev_ok)/total_rows*100:.2f}%)")
    if len(price_dev_ok) > 0:
        logger.info(f"  price_dev stats: min={price_dev_ok['price_dev'].min():.4f}, "
                   f"max={price_dev_ok['price_dev'].max():.4f}, mean={price_dev_ok['price_dev'].mean():.4f}")

    # is_stable_point (all three)
    stable = df_result.filter(pl.col("is_stable_point"))
    logger.info("is_stable_point (all three conditions):")
    logger.info(f"  Count: {len(stable)} / {total_rows} ({len(stable)/total_rows*100:.2f}%)")

    # Final boxes (rolling stable count >= min_steady_days - 2)
    final_boxes = df_result.filter(pl.col("is_box_candidate"))
    logger.info("Final boxes (rolling stable count >= min_steady_days - 2):")
    logger.info(f"  Count: {len(final_boxes)} / {total_rows} ({len(final_boxes)/total_rows*100:.2f}%)")

    logger.info("Intermediate value distributions:")
    logger.info(f"  ma_cv: min={df_result['ma_cv'].min():.4f}, max={df_result['ma_cv'].max():.4f}, "
               f"mean={df_result['ma_cv'].mean():.4f}, median={df_result['ma_cv'].median():.4f}")
    logger.info(f"  ma_slope: min={df_result['ma_slope'].min():.4f}, max={df_result['ma_slope'].max():.4f}, "
               f"mean={df_result['ma_slope'].mean():.4f}, median={df_result['ma_slope'].median():.4f}")
    logger.info(f"  price_dev: min={df_result['price_dev'].min():.4f}, max={df_result['price_dev'].max():.4f}, "
               f"mean={df_result['price_dev'].mean():.4f}, median={df_result['price_dev'].median():.4f}")

    logger.info("=" * 80)
    logger.info("Parameter Adjustment Suggestions:")
    logger.info("=" * 80)
    cv_p50 = df_result["ma_cv"].quantile(0.5)
    cv_p75 = df_result["ma_cv"].quantile(0.75)
    slope_p50 = df_result["ma_slope"].quantile(0.5)
    slope_p75 = df_result["ma_slope"].quantile(0.75)
    dev_p50 = df_result["price_dev"].quantile(0.5)
    dev_p75 = df_result["price_dev"].quantile(0.75)
    logger.info(f"For ~50%% recall: convergence_threshold={cv_p50:.4f}, ma_slope_threshold={slope_p50:.4f}, price_to_ma_threshold={dev_p50:.4f}")
    logger.info(f"For ~75%% recall: convergence_threshold={cv_p75:.4f}, ma_slope_threshold={slope_p75:.4f}, price_to_ma_threshold={dev_p75:.4f}")

    return {
        "total_rows": total_rows,
        "ma_cv_ok": len(ma_cv_ok),
        "ma_slope_ok": len(ma_slope_ok),
        "price_dev_ok": len(price_dev_ok),
        "stable_points": len(stable),
        "final_boxes": len(final_boxes),
    }


def debug_composite_equilibrium_detector(
    df: pl.DataFrame,
    ma_cohesion_threshold: float = 0.015,
    quantile_width_threshold: float = 0.04,
    cross_count_min: int = 3,
    volume_ratio: float = 0.8,
) -> Dict:
    """Debug CompositeEquilibriumDetector: four-dimension composite logic."""
    logger.info("=" * 80)
    logger.info("Debugging CompositeEquilibriumDetector")
    logger.info("=" * 80)

    detector = CompositeEquilibriumDetector(
        box_window=20,
        ma_cohesion_threshold=ma_cohesion_threshold,
        quantile_width_threshold=quantile_width_threshold,
        cross_count_min=cross_count_min,
        volume_ratio=volume_ratio,
    )

    df_result = detector.detect_box(df)
    total_rows = len(df_result)

    # A. MA cohesion
    a_ok = df_result.filter(pl.col("ma_cohesion") < ma_cohesion_threshold)
    logger.info(f"Condition A (ma_cohesion < {ma_cohesion_threshold}): {len(a_ok)} / {total_rows} ({len(a_ok)/total_rows*100:.2f}%%)")

    # B. Quantile width
    b_ok = df_result.filter(pl.col("quantile_width") < quantile_width_threshold)
    logger.info(f"Condition B (quantile_width < {quantile_width_threshold}): {len(b_ok)} / {total_rows} ({len(b_ok)/total_rows*100:.2f}%%)")

    # C. Cross count
    c_ok = df_result.filter(pl.col("ma_cross_count") >= cross_count_min)
    logger.info(f"Condition C (ma_cross_count >= {cross_count_min}): {len(c_ok)} / {total_rows} ({len(c_ok)/total_rows*100:.2f}%%)")

    # D. Volume (if present)
    if "vol_short" in df_result.columns and "vol_long" in df_result.columns:
        d_ok = df_result.filter(pl.col("vol_short") < pl.col("vol_long") * volume_ratio)
        logger.info(f"Condition D (vol_short < vol_long * {volume_ratio}): {len(d_ok)} / {total_rows} ({len(d_ok)/total_rows*100:.2f}%%)")
    else:
        d_ok = df_result
        logger.info("Condition D (volume): skipped (no volume column)")

    final = df_result.filter(pl.col("is_box_candidate"))
    logger.info(f"Final (all four AND): {len(final)} / {total_rows} ({len(final)/total_rows*100:.2f}%%)")

    logger.info("Intermediate distributions:")
    logger.info(f"  ma_cohesion: min={df_result['ma_cohesion'].min():.4f}, max={df_result['ma_cohesion'].max():.4f}, median={df_result['ma_cohesion'].median():.4f}")
    logger.info(f"  quantile_width: min={df_result['quantile_width'].min():.4f}, max={df_result['quantile_width'].max():.4f}, median={df_result['quantile_width'].median():.4f}")
    logger.info(f"  ma_cross_count: min={df_result['ma_cross_count'].min()}, max={df_result['ma_cross_count'].max()}, median={df_result['ma_cross_count'].median():.0f}")

    return {
        "total_rows": total_rows,
        "a_ok": len(a_ok),
        "b_ok": len(b_ok),
        "c_ok": len(c_ok),
        "d_ok": len(d_ok) if "vol_short" in df_result.columns else total_rows,
        "final_boxes": len(final),
    }


def debug_anti_step_detector(
    df: pl.DataFrame,
    box_window: int = 20,
    r_threshold: float = 0.4,
    center_dev_threshold: float = 0.6,
    box_width_threshold: float = 0.15,
) -> Dict:
    """Debug AntiStepBoxDetector: R (correlation) and center deviation."""
    logger.info("=" * 80)
    logger.info("Debugging AntiStepBoxDetector")
    logger.info("=" * 80)

    detector = AntiStepBoxDetector(
        box_window=box_window,
        r_threshold=r_threshold,
        center_dev_threshold=center_dev_threshold,
        box_width_threshold=box_width_threshold,
    )

    df_result = detector.detect_box(df)
    total_rows = len(df_result)

    r_ok = df_result.filter(pl.col("r_value").abs() < r_threshold)
    logger.info(f"Condition |r_value| < {r_threshold}: {len(r_ok)} / {total_rows} ({len(r_ok)/total_rows*100:.2f}%%)")

    dev_ok = df_result.filter(pl.col("center_dev") < center_dev_threshold)
    logger.info(f"Condition center_dev < {center_dev_threshold}: {len(dev_ok)} / {total_rows} ({len(dev_ok)/total_rows*100:.2f}%%)")

    width_ok = df_result.filter(pl.col("box_width") < box_width_threshold)
    logger.info(f"Condition box_width < {box_width_threshold}: {len(width_ok)} / {total_rows} ({len(width_ok)/total_rows*100:.2f}%%)")

    final = df_result.filter(pl.col("is_box_candidate"))
    logger.info(f"Final (all AND): {len(final)} / {total_rows} ({len(final)/total_rows*100:.2f}%%)")

    logger.info("Intermediate distributions:")
    logger.info(f"  r_value: min={df_result['r_value'].min():.4f}, max={df_result['r_value'].max():.4f}, median={df_result['r_value'].median():.4f}")
    logger.info(f"  center_dev: min={df_result['center_dev'].min():.4f}, max={df_result['center_dev'].max():.4f}, median={df_result['center_dev'].median():.4f}")

    return {
        "total_rows": total_rows,
        "r_ok": len(r_ok),
        "dev_ok": len(dev_ok),
        "width_ok": len(width_ok),
        "final_boxes": len(final),
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Debug box detectors to understand why no boxes are detected"
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
        help="Optional list of stock codes to debug (default: all stocks)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["accurate", "expansion_anchor", "balanced", "dense_area", "ribbon_coherence", "composite_equilibrium", "anti_step", "both"],
        default="both",
        help="Which detector to debug (default: both)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached data if available",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config()
        db_config = config.database
    except Exception as e:
        logger.warning(f"Failed to load config: {e}, using defaults")
        db_config = DatabaseConfig()
    
    # Load data
    data_loader = TeapotDataLoader(
        db_config=db_config,
        schema="quant",
        use_cache=args.use_cache,
    )
    
    logger.info(f"Loading data: {args.start_date} to {args.end_date}")
    df = data_loader.load_daily_data(
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=args.symbols,
    )
    
    if df.is_empty():
        logger.error("No data loaded!")
        return
    
    logger.info(f"Loaded {len(df)} rows for {df['ts_code'].n_unique()} stocks")
    
    # Debug detectors
    if args.detector in ["accurate", "both"]:
        debug_accurate_detector(df)
    
    if args.detector in ["expansion_anchor", "both"]:
        debug_expansion_anchor_detector(df)
    
    if args.detector in ["balanced", "both"]:
        debug_balanced_detector(df)

    if args.detector in ["dense_area", "both"]:
        debug_dense_area_detector(df)

    if args.detector in ["ribbon_coherence", "both"]:
        debug_ribbon_coherence_detector(df)

    if args.detector in ["composite_equilibrium", "both"]:
        debug_composite_equilibrium_detector(df)

    if args.detector in ["anti_step", "both"]:
        debug_anti_step_detector(df)

    logger.info("\n" + "=" * 80)
    logger.info("Debug complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
