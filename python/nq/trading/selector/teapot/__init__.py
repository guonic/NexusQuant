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
from nq.trading.selector.teapot.filters import TeapotFilters
from nq.trading.selector.teapot.state_machine import TeapotStateMachine

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
]
