"""Metrics calculators - Plugin system for climbing analysis.

Each calculator is responsible for a specific domain of metrics.
This enables:
- Single Responsibility: Each calculator has one job
- Open/Closed: Add new calculators without modifying existing ones
- Testability: Test each calculator independently
- Composability: Combine calculators as needed
"""

from .base import MetricsCalculator
from .stability import StabilityCalculator
from .progress import ProgressCalculator
from .efficiency import EfficiencyCalculator
from .technique import TechniqueCalculator
from .joint_angles import JointAngleCalculator

__all__ = [
    "MetricsCalculator",
    "StabilityCalculator",
    "ProgressCalculator",
    "EfficiencyCalculator",
    "TechniqueCalculator",
    "JointAngleCalculator",
]
