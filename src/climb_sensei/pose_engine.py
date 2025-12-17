"""Pose estimation engine using MediaPipe.

This module provides a wrapper around MediaPipe's pose detection
functionality for extracting human pose landmarks from images.
"""

from typing import Optional, List, Dict, Any
import urllib.request
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


# Default MediaPipe pose model URL
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def _get_model_path() -> str:
    """Download and cache the MediaPipe pose model.
    
    Returns:
        Path to the cached model file.
    """
    # Use a cache directory in the user's home
    cache_dir = Path.home() / ".cache" / "climb_sensei"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = cache_dir / "pose_landmarker_lite.task"
    
    # Download if not cached
    if not model_path.exists():
        print(f"Downloading pose model to {model_path}...")
        try:
            urllib.request.urlretrieve(_MODEL_URL, model_path)
            print("Model downloaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to download pose model: {e}")
    
    return str(model_path)


class PoseEngine:
    """Pose estimation engine using MediaPipe PoseLandmarker.
    
    This class wraps MediaPipe's PoseLandmarker to detect human pose
    landmarks in images. It provides a clean interface for processing
    individual frames and extracting landmark coordinates.
    
    Attributes:
        min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for
            pose detection to be considered successful.
        min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for
            pose tracking to be considered successful.
        landmarker: MediaPipe PoseLandmarker instance.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[str] = None
    ) -> None:
        """Initialize the pose engine.
        
        Args:
            min_detection_confidence: Minimum confidence for detection (0.0-1.0).
            min_tracking_confidence: Minimum confidence for tracking (0.0-1.0).
            model_path: Optional path to the pose model file. If None, downloads
                       the default model automatically.
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Get model path (download if needed)
        if model_path is None:
            model_path = _get_model_path()
        
        # Create PoseLandmarker options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_results = None

    def process(self, image: np.ndarray) -> Optional[Any]:
        """Process an image and detect pose landmarks.
        
        Args:
            image: Input image in BGR format (OpenCV convention).
        
        Returns:
            MediaPipe pose detection results object containing landmarks,
            or None if no pose was detected.
        """
        # Convert BGR to RGB (MediaPipe expects RGB)
        image_rgb = np.ascontiguousarray(image[:, :, ::-1])
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect pose landmarks
        results = self.landmarker.detect(mp_image)
        
        # Store results for later use
        self._last_results = results if results.pose_landmarks else None
        
        return self._last_results

    def extract_landmarks(self, results: Any = None) -> List[Dict[str, float]]:
        """Extract landmark coordinates from pose detection results.
        
        Args:
            results: MediaPipe pose detection results object. If None, uses
                    the last processed results.
        
        Returns:
            List of dictionaries containing x, y, z coordinates and
            visibility for each landmark. Coordinates are normalized
            to [0.0, 1.0] range.
        """
        if results is None:
            results = self._last_results
        
        if not results or not results.pose_landmarks:
            return []
        
        # Get first pose (PoseLandmarker can detect multiple poses)
        pose_landmarks = results.pose_landmarks[0]
        
        landmarks = []
        for landmark in pose_landmarks:
            landmarks.append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            })
        
        return landmarks

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.landmarker.close()

    def __enter__(self) -> "PoseEngine":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
