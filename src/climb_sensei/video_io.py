"""Video input/output handling module.

This module provides classes for reading from and writing to video files
using OpenCV.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


class VideoReader:
    """Read video frames from a file or camera source.

    Attributes:
        path: Path to video file or camera index.
        cap: OpenCV VideoCapture object.
        fps: Frames per second of the video.
        frame_count: Total number of frames in the video.
        width: Width of video frames in pixels.
        height: Height of video frames in pixels.
    """

    def __init__(self, path: str | int) -> None:
        """Initialize the video reader.

        Args:
            path: Path to video file or camera index (0 for default camera).

        Raises:
            ValueError: If the video source cannot be opened.
        """
        self.path = path
        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {path}")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame from the video.

        Returns:
            A tuple of (success, frame) where success is True if frame was read
            successfully, and frame is the image data as a numpy array.
        """
        success, frame = self.cap.read()
        return success, frame if success else None

    def release(self) -> None:
        """Release the video capture resource."""
        self.cap.release()

    def __enter__(self) -> "VideoReader":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.release()


class VideoWriter:
    """Write video frames to a file.

    Attributes:
        path: Output file path.
        fourcc: FourCC codec code.
        fps: Frames per second for output video.
        width: Width of output frames in pixels.
        height: Height of output frames in pixels.
        writer: OpenCV VideoWriter object.
    """

    def __init__(
        self, path: str, fps: int, width: int, height: int, fourcc: str = "mp4v"
    ) -> None:
        """Initialize the video writer.

        Args:
            path: Output file path.
            fps: Frames per second for the output video.
            width: Width of output frames in pixels.
            height: Height of output frames in pixels.
            fourcc: FourCC codec code (default: "mp4v").

        Raises:
            ValueError: If the video writer cannot be initialized.
        """
        self.path = path
        self.fps = fps
        self.width = width
        self.height = height
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)

        self.writer = cv2.VideoWriter(path, self.fourcc, fps, (width, height))

        if not self.writer.isOpened():
            raise ValueError(f"Cannot open video writer for: {path}")

    def write(self, frame: np.ndarray) -> None:
        """Write a frame to the video file.

        Args:
            frame: Image data as a numpy array (BGR format).
        """
        self.writer.write(frame)

    def release(self) -> None:
        """Release the video writer resource."""
        self.writer.release()

    def __enter__(self) -> "VideoWriter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.release()
