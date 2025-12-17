"""Tests for smoothing algorithms."""

import pytest
import numpy as np
from climb_sensei.smoothing import OneEuroFilter, LandmarkSmoother, ExponentialMovingAverage


class TestOneEuroFilter:
    """Test One Euro Filter implementation."""
    
    def test_initialization(self):
        """Test filter initialization."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
        assert f.min_cutoff == 1.0
        assert f.beta == 0.007
        assert f.d_cutoff == 1.0
        assert f.x_prev is None
    
    def test_first_value_passthrough(self):
        """Test that first value passes through unchanged."""
        f = OneEuroFilter()
        result = f(10.0, t=0.0)
        assert result == 10.0
    
    def test_smoothing_reduces_noise(self):
        """Test that filter reduces noise in signal."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        
        # Generate noisy signal around 10.0
        np.random.seed(42)
        noisy_values = [10.0 + np.random.normal(0, 2.0) for _ in range(100)]
        
        # Apply filter
        smoothed_values = []
        for i, val in enumerate(noisy_values):
            smoothed_values.append(f(val, t=i * 0.033))  # 30 fps
        
        # Smoothed signal should have less variance (after warm-up)
        original_std = np.std(noisy_values[20:])
        smoothed_std = np.std(smoothed_values[20:])
        assert smoothed_std < original_std
    
    def test_responsiveness_to_fast_movement(self):
        """Test that filter responds quickly to fast movements."""
        f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
        
        # Slow movement then sudden jump
        values = [10.0] * 10 + [20.0] * 10
        
        smoothed_values = []
        for i, val in enumerate(values):
            smoothed_values.append(f(val, t=i * 0.033))
        
        # After the jump, should reach near target relatively quickly
        # Check that by frame 15 (5 frames after jump), we're closer to 20 than 10
        assert smoothed_values[15] > 15.0
    
    def test_reset(self):
        """Test filter reset functionality."""
        f = OneEuroFilter()
        f(10.0, t=0.0)
        f(11.0, t=1.0)
        
        assert f.x_prev is not None
        
        f.reset()
        assert f.x_prev is None
        assert f.dx_prev is None
        assert f.t_prev is None


class TestLandmarkSmoother:
    """Test LandmarkSmoother class."""
    
    def test_initialization(self):
        """Test smoother initialization."""
        smoother = LandmarkSmoother(
            filter_type="one_euro",
            min_cutoff=1.0,
            beta=0.007
        )
        assert smoother.filter_type == "one_euro"
        assert smoother.min_cutoff == 1.0
        assert smoother.beta == 0.007
        assert len(smoother.filters) == 0
    
    def test_empty_landmarks(self):
        """Test handling of empty landmark list."""
        smoother = LandmarkSmoother()
        result = smoother.smooth([])
        assert result == []
    
    def test_smooth_single_landmark(self):
        """Test smoothing a single landmark."""
        smoother = LandmarkSmoother(min_cutoff=1.0, beta=0.0)
        
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9}]
        result = smoother.smooth(landmarks, timestamp=0.0)
        
        assert len(result) == 1
        assert "x" in result[0]
        assert "y" in result[0]
        assert "z" in result[0]
        assert "visibility" in result[0]
    
    def test_smooth_multiple_landmarks(self):
        """Test smoothing multiple landmarks."""
        smoother = LandmarkSmoother(min_cutoff=1.0, beta=0.0)
        
        # Create 3 landmarks
        landmarks = [
            {"x": 0.1, "y": 0.1, "z": 0.0, "visibility": 0.9},
            {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.8},
            {"x": 0.9, "y": 0.9, "z": 0.0, "visibility": 0.95},
        ]
        
        # Apply smoothing over multiple frames
        for t in range(10):
            result = smoother.smooth(landmarks, timestamp=t * 0.033)
        
        assert len(result) == 3
        # Each landmark should have all coordinates
        for lm in result:
            assert "x" in lm
            assert "y" in lm
            assert "z" in lm
            assert "visibility" in lm
    
    def test_noise_reduction(self):
        """Test that smoothing reduces noise in landmark positions."""
        smoother = LandmarkSmoother(min_cutoff=1.0, beta=0.0)
        
        np.random.seed(42)
        
        # Generate noisy landmark sequence
        x_values = []
        smoothed_x_values = []
        
        for t in range(50):
            # Add noise to base position
            noisy_x = 0.5 + np.random.normal(0, 0.05)
            landmarks = [{"x": noisy_x, "y": 0.5, "z": 0.0, "visibility": 0.9}]
            
            result = smoother.smooth(landmarks, timestamp=t * 0.033)
            
            x_values.append(noisy_x)
            smoothed_x_values.append(result[0]["x"])
        
        # Smoothed values should have less variance (after warm-up)
        original_std = np.std(x_values[20:])
        smoothed_std = np.std(smoothed_x_values[20:])
        assert smoothed_std < original_std
    
    def test_reset(self):
        """Test smoother reset functionality."""
        smoother = LandmarkSmoother()
        
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9}]
        smoother.smooth(landmarks, timestamp=0.0)
        smoother.smooth(landmarks, timestamp=1.0)
        
        assert len(smoother.filters) > 0
        
        smoother.reset()
        assert len(smoother.filters) == 0


class TestExponentialMovingAverage:
    """Test EMA filter implementation."""
    
    def test_initialization(self):
        """Test EMA initialization."""
        ema = ExponentialMovingAverage(alpha=0.3)
        assert ema.alpha == 0.3
        assert ema.value is None
    
    def test_first_value_passthrough(self):
        """Test that first value passes through."""
        ema = ExponentialMovingAverage(alpha=0.3)
        result = ema(10.0)
        assert result == 10.0
        assert ema.value == 10.0
    
    def test_smoothing(self):
        """Test EMA smoothing behavior."""
        ema = ExponentialMovingAverage(alpha=0.5)
        
        # Feed values
        result1 = ema(10.0)  # First value
        result2 = ema(20.0)  # Should be 0.5 * 20 + 0.5 * 10 = 15
        
        assert result1 == 10.0
        assert result2 == 15.0
    
    def test_reset(self):
        """Test EMA reset."""
        ema = ExponentialMovingAverage(alpha=0.3)
        ema(10.0)
        ema(20.0)
        
        assert ema.value is not None
        
        ema.reset()
        assert ema.value is None
