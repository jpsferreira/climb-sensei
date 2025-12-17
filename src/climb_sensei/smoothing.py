"""Smoothing and filtering algorithms for pose landmark stabilization.

This module provides various filtering techniques to reduce jitter and
noise in pose tracking data while preserving motion dynamics.
"""

from typing import Optional, List, Dict
import time
import numpy as np


class OneEuroFilter:
    """One Euro Filter for adaptive smoothing of time-series data.
    
    The One Euro Filter is an adaptive low-pass filter that adjusts its
    cutoff frequency based on the speed of change. This makes it effective
    at reducing jitter during slow movements while remaining responsive
    during fast movements.
    
    Reference:
        Casiez, G., Roussel, N., & Vogel, D. (2012).
        1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems.
    
    Attributes:
        min_cutoff: Minimum cutoff frequency (Hz). Lower values = more smoothing.
        beta: Speed coefficient. Higher values = more adaptation to speed changes.
        d_cutoff: Cutoff frequency for derivative. Controls derivative smoothing.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ) -> None:
        """Initialize the One Euro Filter.
        
        Args:
            min_cutoff: Minimum cutoff frequency in Hz (default: 1.0).
                       Lower values provide more smoothing but increase lag.
            beta: Speed coefficient (default: 0.007).
                 Controls how much the filter adapts to speed changes.
                 Higher values = more responsive to speed.
            d_cutoff: Derivative cutoff frequency in Hz (default: 1.0).
                     Controls smoothing of the derivative signal.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        # Internal state
        self.x_prev: Optional[float] = None
        self.dx_prev: Optional[float] = None
        self.t_prev: Optional[float] = None

    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        """Calculate the smoothing factor (alpha) for the low-pass filter.
        
        Args:
            t_e: Sampling period (time since last update).
            cutoff: Cutoff frequency in Hz.
        
        Returns:
            Smoothing factor alpha in range [0, 1].
        """
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        """Apply the One Euro Filter to a new data point.
        
        Args:
            x: New input value to filter.
            t: Timestamp in seconds. If None, uses current time.
        
        Returns:
            Filtered value.
        """
        if t is None:
            t = time.time()
        
        # Initialize on first call
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x
        
        # Calculate time difference
        t_e = t - self.t_prev
        
        # Avoid division by zero
        if t_e <= 0:
            return self.x_prev
        
        # Calculate derivative (speed of change)
        dx = (x - self.x_prev) / t_e
        
        # Smooth the derivative
        alpha_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        
        # Calculate adaptive cutoff frequency based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Smooth the signal with adaptive cutoff
        alpha = self._smoothing_factor(t_e, cutoff)
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        
        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat

    def reset(self) -> None:
        """Reset the filter state."""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class LandmarkSmoother:
    """Smooth pose landmarks using filtering algorithms.
    
    This class applies smoothing filters to pose landmarks to reduce
    jitter and noise while preserving natural motion dynamics.
    """

    def __init__(
        self,
        filter_type: str = "one_euro",
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ) -> None:
        """Initialize the landmark smoother.
        
        Args:
            filter_type: Type of filter to use. Currently supports "one_euro".
            min_cutoff: Minimum cutoff frequency for One Euro Filter.
            beta: Speed coefficient for One Euro Filter.
            d_cutoff: Derivative cutoff for One Euro Filter.
        """
        self.filter_type = filter_type
        
        # Filters for each landmark coordinate (x, y, z)
        # Key: (landmark_index, coordinate) -> filter instance
        self.filters: Dict[tuple, OneEuroFilter] = {}
        
        # Filter parameters
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

    def _get_filter(self, landmark_idx: int, coord: str) -> OneEuroFilter:
        """Get or create a filter for a specific landmark coordinate.
        
        Args:
            landmark_idx: Index of the landmark (0-32).
            coord: Coordinate name ('x', 'y', 'z', or 'visibility').
        
        Returns:
            OneEuroFilter instance for this landmark coordinate.
        """
        key = (landmark_idx, coord)
        if key not in self.filters:
            self.filters[key] = OneEuroFilter(
                min_cutoff=self.min_cutoff,
                beta=self.beta,
                d_cutoff=self.d_cutoff
            )
        return self.filters[key]

    def smooth(
        self,
        landmarks: List[Dict[str, float]],
        timestamp: Optional[float] = None
    ) -> List[Dict[str, float]]:
        """Apply smoothing to a list of landmarks.
        
        Args:
            landmarks: List of landmark dictionaries with 'x', 'y', 'z', 'visibility' keys.
            timestamp: Optional timestamp in seconds. If None, uses current time.
        
        Returns:
            List of smoothed landmarks with same structure as input.
        """
        if not landmarks:
            return landmarks
        
        if timestamp is None:
            timestamp = time.time()
        
        smoothed = []
        for idx, landmark in enumerate(landmarks):
            smoothed_landmark = {}
            
            # Smooth each coordinate
            for coord in ['x', 'y', 'z', 'visibility']:
                if coord in landmark:
                    filter_instance = self._get_filter(idx, coord)
                    smoothed_landmark[coord] = filter_instance(
                        landmark[coord],
                        timestamp
                    )
                else:
                    smoothed_landmark[coord] = landmark.get(coord, 0.0)
            
            smoothed.append(smoothed_landmark)
        
        return smoothed

    def reset(self) -> None:
        """Reset all filters to initial state."""
        for filter_instance in self.filters.values():
            filter_instance.reset()
        self.filters.clear()


class ExponentialMovingAverage:
    """Simple Exponential Moving Average (EMA) filter.
    
    A lightweight alternative to One Euro Filter with fixed smoothing.
    Good for quick smoothing but less adaptive to movement speed.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        """Initialize EMA filter.
        
        Args:
            alpha: Smoothing factor in range (0, 1].
                  Lower values = more smoothing, higher lag.
                  Higher values = less smoothing, more responsive.
        """
        self.alpha = alpha
        self.value: Optional[float] = None

    def __call__(self, x: float) -> float:
        """Apply EMA to new value.
        
        Args:
            x: New input value.
        
        Returns:
            Smoothed value.
        """
        if self.value is None:
            self.value = x
            return x
        
        self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        """Reset filter state."""
        self.value = None
