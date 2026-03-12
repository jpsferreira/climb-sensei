"""Climbing analysis metrics — legacy module.

The ClimbingAnalyzer and AdvancedClimbingMetrics classes have been removed.
Use ClimbingAnalysisService with composable calculators instead:

    >>> from climb_sensei.services import ClimbingAnalysisService
    >>> service = ClimbingAnalysisService()
    >>> analysis = service.analyze(landmarks_sequence, fps=30.0)

See ``climb_sensei.domain.calculators`` for the individual calculator plugins
and ``climb_sensei.models`` for result dataclasses.
"""
