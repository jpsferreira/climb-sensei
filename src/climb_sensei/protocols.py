"""Protocol definitions for extensibility.

This module defines protocols (interfaces) that allow different implementations
while maintaining type safety and enabling dependency injection.
"""

from typing import Protocol, runtime_checkable, List, Dict, Any
import numpy as np


@runtime_checkable
class PoseDetector(Protocol):
    """Protocol for pose detection engines.

    Implementations can use MediaPipe, OpenPose, MMPose, or any other
    pose estimation framework as long as they conform to this interface.
    """

    def process(self, frame: np.ndarray) -> Any:
        """Detect pose landmarks in a frame.

        Args:
            frame: Image frame as numpy array (BGR format)

        Returns:
            Pose detection results (implementation-specific)
        """
        ...

    def extract_landmarks(self, results: Any) -> List[Dict[str, float]]:
        """Extract landmark coordinates from detection results.

        Args:
            results: Detection results from process()

        Returns:
            List of landmarks with x, y, z, visibility keys
        """
        ...

    def close(self) -> None:
        """Release resources and cleanup."""
        ...


@runtime_checkable
class AnalysisRepository(Protocol):
    """Protocol for storing and retrieving analysis results.

    Implementations can use JSON, CSV, databases, or cloud storage.
    """

    def save(self, analysis: Any, path: str) -> None:
        """Save analysis results to storage.

        Args:
            analysis: Analysis results to save
            path: Storage location
        """
        ...

    def load(self, path: str) -> Any:
        """Load analysis results from storage.

        Args:
            path: Storage location

        Returns:
            Loaded analysis results
        """
        ...
