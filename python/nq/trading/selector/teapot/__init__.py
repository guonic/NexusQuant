"""
Teapot pattern recognition selector module.

Provides pattern recognition and signal generation for Teapot strategy.
"""

from nq.trading.selector.teapot.base import TeapotSelector
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
from nq.trading.selector.teapot.features import TeapotFeatures
from nq.trading.selector.teapot.step_breakout_scanner import (
    StepBreakoutResult,
    StepBreakoutScanner,
)
from nq.trading.selector.teapot.topological_trend_scanner import (
    TopologicalTrendScanner,
)
from nq.trading.selector.teapot.pure_price_dru import (
    PurePriceDRUScanner,
    analyze_pure_price_logic,
)
from nq.trading.selector.teapot.fractal_box_scanner import (
    FractalBoxScanner,
    capture_topological_final,
)
from nq.trading.selector.teapot.vertical_piercing_scanner import (
    VerticalPiercingScanner,
    capture_vertical_piercing,
)
from nq.trading.selector.teapot.tangle_reversal_scanner import (
    TangleReversalScanner,
    capture_tangle_reversal,
)
from nq.trading.selector.teapot.filters import TeapotFilters
from nq.trading.selector.teapot.state_machine import TeapotStateMachine
from nq.trading.selector.teapot.market_regime import analyze_market_regime

__all__ = [
    "TeapotSelector",
    "TeapotFeatures",
    "TeapotStateMachine",
    "TeapotFilters",
    "BoxDetector",
    "SimpleBoxDetector",
    "MeanReversionBoxDetector",
    "MeanReversionBoxDetectorV2",
    "HybridBoxDetector",
    "HybridBoxDetectorV2",
    "MovingAverageConvergenceBoxDetector",
    "DynamicConvergenceDetector",
    "KeltnerSqueezeDetector",
    "ExpansionAnchorBoxDetector",
    "AccurateBoxDetector",
    "BalancedBoxDetector",
    "DenseAreaBoxDetector",
    "RibbonCoherenceDetector",
    "CompositeEquilibriumDetector",
    "AntiStepBoxDetector",
    "StepBreakoutScanner",
    "StepBreakoutResult",
    "TopologicalTrendScanner",
    "PurePriceDRUScanner",
    "analyze_pure_price_logic",
    "FractalBoxScanner",
    "capture_topological_final",
    "VerticalPiercingScanner",
    "capture_vertical_piercing",
    "TangleReversalScanner",
    "capture_tangle_reversal",
    "analyze_market_regime",
]
