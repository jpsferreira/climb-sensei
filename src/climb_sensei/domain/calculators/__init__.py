"""Metrics calculators - Plugin system for climbing analysis.

Each calculator is responsible for a specific domain of metrics.
This enables:
- Single Responsibility: Each calculator has one job
- Open/Closed: Add new calculators without modifying existing ones
- Testability: Test each calculator independently
- Composability: Combine calculators as needed
"""

from .base import FrameContext, MetricsCalculator
from .efficiency import EfficiencyCalculator
from .fatigue import FatigueCalculator
from .joint_angles import JointAngleCalculator
from .progress import ProgressCalculator
from .stability import StabilityCalculator
from .technique import TechniqueCalculator

__all__ = [
    "FrameContext",
    "MetricsCalculator",
    "StabilityCalculator",
    "ProgressCalculator",
    "EfficiencyCalculator",
    "TechniqueCalculator",
    "JointAngleCalculator",
    "FatigueCalculator",
]
