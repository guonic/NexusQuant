# -*- coding: utf-8 -*-
"""
Teapot K-line pattern recognition (multi-scale CNN + DTW golden patterns).

- MultiScaleCNN: few-shot binary classifier (positive vs negative slices).
- PatternMatcher: DTW-based similarity to golden templates (shape alignment).
"""

from nq.ai.teapot_pattern.dataset import KLineDataset
from nq.ai.teapot_pattern.dtw_matcher import PatternMatcher
from nq.ai.teapot_pattern.model import MultiScaleCNN

__all__ = ["MultiScaleCNN", "KLineDataset", "PatternMatcher"]
