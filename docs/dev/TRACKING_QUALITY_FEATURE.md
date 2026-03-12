# Tracking Quality Analysis - Implementation Summary

## Overview

Added skeleton/pose tracking quality analysis to climb-sensei with dual interface:

1. **From video files** - Full pose detection + quality analysis
2. **From landmarks** - Efficient analysis using already-extracted coordinates

The landmarks-based approach is ideal for `analyze_climb` workflows where pose data is already available.

## What Was Added

### 1. Core Module: `src/climb_sensei/tracking_quality.py`

**Classes:**

- `TrackingQualityAnalyzer` - Main analyzer with configurable thresholds
- `TrackingQualityReport` - Dataclass containing quality assessment results

**Key Methods:**

- `analyze_video(video_path)` - Analyze from video file (runs pose detection)
- `analyze_from_landmarks(landmarks_sequence)` - Analyze from pre-extracted landmarks ⚡
- `_analyze_frames_from_video()` - Internal video processing
- `_analyze_landmarks_sequence()` - Internal landmarks processing
- `_calculate_smoothness()` - Landmark jitter calculation
- `_determine_quality_level()` - Overall quality assessment

**Convenience Functions:**

- `analyze_tracking_quality(video_path)` - Quick video analysis
- `analyze_tracking_from_landmarks(landmarks_sequence)` - Quick landmarks analysis

### 2. Quality Metrics

**Detection Quality:**

- **Detection rate**: % of frames with pose detected (default min: 70%)
- **Landmark confidence**: Average confidence scores (default min: 0.5)
- **Visibility score**: % of visible landmarks (default min: 60%)

**Tracking Quality:**

- **Smoothness**: Based on landmark jitter, 0-1 scale (default min: 0.6)
- **Tracking loss events**: Times tracking was lost then regained (default max: 5)

**Quality Levels:**

- `excellent`: All metrics significantly exceed thresholds
- `good`: All metrics meet or exceed thresholds
- `acceptable`: Meets minimum requirements with some warnings
- `poor`: Falls below minimum requirements

### 3. Test Suite: `tests/test_tracking_quality.py`

**27 tests total:**

- 16 tests for core analyzer functionality
- 11 tests for landmarks-based analysis

**Test Coverage:**

- Initialization and custom thresholds
- Smoothness calculation (smooth vs jittery)
- Quality level determination
- Video-based analysis (with mocks)
- **Landmarks-based analysis:**
  - High/low quality sequences
  - Intermittent detection
  - Smooth/jittery tracking
  - Empty sequences and edge cases
  - Sample rate handling
  - Custom file paths
  - Convenience functions

### 4. CLI Integration: `scripts/check_video_quality.py`

Added `--tracking` flag to existing video quality checker:

```bash
# Tracking quality analysis
python scripts/check_video_quality.py video.mp4 --tracking

# With sample rate (analyze every Nth frame)
python scripts/check_video_quality.py video.mp4 --tracking --sample-rate 10

# Combined analysis
python scripts/check_video_quality.py video.mp4 --deep --tracking --json quality.json
```

**Note:** CLI uses video-based analysis (not landmarks) since it's standalone.

### 5. Example Code: `examples/analyze_with_tracking.py`

Complete example showing efficient integration:

- Process video once with pose detection
- Store landmarks for both climbing analysis and tracking quality
- Generate combined report
- Export JSON results

### 6. Documentation Updates

**Updated files:**

- `docs/usage.md` - Added comprehensive "Tracking Quality Analysis" section
- `docs/api.md` - Added API reference for tracking quality classes
- `src/climb_sensei/__init__.py` - Exported all tracking quality components

**New documentation sections:**

- From video file analysis
- From landmarks analysis (more efficient)
- Using TrackingQualityAnalyzer class
- Tracking quality metrics explained
- CLI tool with tracking flag

## Usage Examples

### Efficient Integration (Recommended)

```python
from climb_sensei import (
    ClimbingAnalysis,
    PoseEngine,
    VideoReader,
    analyze_tracking_from_landmarks,
)

analyzer = ClimbingAnalysis(window_size=30, fps=30)
landmarks_history = []

# Process video once
with PoseEngine() as engine:
    with VideoReader('climbing.mp4') as reader:
        while True:
            success, frame = reader.read()
            if not success:
                break

            results = engine.process(frame)
            if results:
                landmarks = engine.extract_landmarks(results)
                if landmarks:
                    analyzer.analyze_frame(landmarks)
                    landmarks_history.append(landmarks)
                else:
                    landmarks_history.append(None)
            else:
                landmarks_history.append(None)

# Get both results
climbing_summary = analyzer.get_summary()
tracking_report = analyze_tracking_from_landmarks(landmarks_history)

# Check if results are reliable
if tracking_report.is_trackable:
    print(f"✓ Reliable results - {tracking_report.quality_level} tracking")
    print(f"  Detection rate: {tracking_report.detection_rate}%")
    print(f"  Smoothness: {tracking_report.tracking_smoothness:.3f}")
else:
    print("⚠️  Results may be unreliable due to poor tracking")
    print(f"  Issues: {tracking_report.issues}")
```

### From Video (Standalone)

```python
from climb_sensei import analyze_tracking_quality

report = analyze_tracking_quality('climbing.mp4', sample_rate=5)

if report.is_trackable:
    print(f"Detection rate: {report.detection_rate}%")
    print(f"Quality: {report.quality_level}")
else:
    print("Poor tracking:", report.issues)
```

### Custom Thresholds

```python
from climb_sensei import TrackingQualityAnalyzer

analyzer = TrackingQualityAnalyzer(
    min_detection_rate=80.0,
    min_avg_confidence=0.6,
    min_visibility=70.0,
    min_smoothness=0.7,
    sample_rate=5,
)

# From video
report = analyzer.analyze_video('video.mp4')

# Or from landmarks
report = analyzer.analyze_from_landmarks(landmarks_sequence)
```

## Performance Comparison

### Video-based Analysis

- Runs full pose detection on video
- Slower (processes all frames)
- Use when: Standalone quality checks, no existing landmarks

### Landmarks-based Analysis ⚡

- Uses pre-extracted landmarks
- **Much faster** (no pose detection overhead)
- Use when: During analyze_climb, landmarks already available
- **Recommended for production workflows**

## Quality Thresholds

### Detection Rate

- **Minimum:** 70% of frames with pose detected
- **Good:** 85%+
- **Excellent:** 95%+

### Landmark Confidence

- **Minimum:** 0.5 average confidence
- **Good:** 0.65+
- **Excellent:** 0.8+

### Visibility

- **Minimum:** 60% of landmarks visible
- **Good:** 75%+
- **Excellent:** 85%+

### Smoothness

- **Minimum:** 0.6 (some jitter acceptable)
- **Good:** 0.7+
- **Excellent:** 0.8+ (very smooth)

### Tracking Losses

- **Maximum:** 5 events
- **Good:** ≤2 events
- **Excellent:** 0 events

## Integration Benefits

1. **Efficient**: Reuse landmarks from analysis pipeline
2. **Real-time feedback**: Know if tracking is poor during processing
3. **Quality assurance**: Flag unreliable results automatically
4. **Backend ready**: Validate uploads before expensive processing
5. **Debugging**: Identify problematic videos quickly

## API Exports

All tracking quality components are now publicly available:

```python
from climb_sensei import (
    TrackingQualityAnalyzer,
    TrackingQualityReport,
    analyze_tracking_quality,
    analyze_tracking_from_landmarks,  # ⚡ Recommended
)
```

## Test Results

All 27 tests passing:

- ✅ 16 core analyzer tests
- ✅ 11 landmarks-based analysis tests
- ✅ Documentation builds successfully
- ✅ All imports working correctly

## Files Modified/Created

**Created:**

- `src/climb_sensei/tracking_quality.py` (450+ lines)
- `tests/test_tracking_quality.py` (450+ lines, 27 tests)
- `examples/analyze_with_tracking.py` (200+ lines)

**Modified:**

- `src/climb_sensei/__init__.py` - Added tracking quality exports
- `docs/usage.md` - Added tracking quality section
- `docs/api.md` - Added API reference
- `scripts/check_video_quality.py` - Added --tracking flag (optional future work)

## Next Steps

Suggested enhancements:

1. Add tracking quality report to ClimbingAnalysis.get_summary()
2. Auto-warn when tracking quality is poor
3. Add tracking quality visualization
4. Export tracking quality to JSON/CSV
5. Add confidence scores per landmark (if available from MediaPipe)
6. Per-landmark visibility tracking
7. Temporal smoothness analysis (frame-to-frame consistency)

## Production Ready

The tracking quality analysis system is production-ready:

- ✅ Dual interface (video + landmarks)
- ✅ Comprehensive test coverage (27 tests)
- ✅ Complete documentation
- ✅ Example integration code
- ✅ Efficient landmarks-based approach
- ✅ API exports
- ✅ All tests passing
