# climb-sensei

A Python pose estimation tool for analyzing climbing footage. Extract vertical movement metrics, calculate biomechanics, visualize technique with animated dashboards, and analyze climbing performance using computer vision.

[![CI](https://github.com/jpsferreira/climb-sensei/workflows/CI/badge.svg)](https://github.com/jpsferreira/climb-sensei/actions)
[![codecov](https://codecov.io/gh/jpsferreira/climb-sensei/branch/main/graph/badge.svg)](https://codecov.io/gh/jpsferreira/climb-sensei)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- 🎯 **Pose Detection**: Real-time pose estimation using MediaPipe
- 📊 **Performance Analysis**: 25+ climbing metrics (velocity, stability, efficiency)
- 📐 **Biomechanics**: Joint angles, reach distances, center of mass
- 📹 **Video Processing**: Annotated videos with side-by-side dashboards
- 🎨 **Visualization**: Pose landmarks and real-time metric graphs
- ✅ **Video Quality Validation**: Pre-processing quality checks for backend APIs
- 🧪 **Tested**: Comprehensive test suite with high code coverage

## Quick Start

### Installation

```bash
# Install uv (if not already installed)
pip install uv

# Install climb-sensei
uv sync
```

### Analyze a Video

```bash
# Quick terminal analysis
python scripts/analyze_climb.py climbing_video.mp4

# Create annotated video with dashboard
python scripts/analyze_climb.py climbing_video.mp4 --video output.mp4

# Export JSON data
python scripts/analyze_climb.py climbing_video.mp4 --json analysis.json
```

### Python API

```python
from climb_sensei import PoseEngine, VideoReader, ClimbingAnalyzer

analyzer = ClimbingAnalyzer(window_size=30, fps=30)

with PoseEngine() as engine:
    with VideoReader('climbing_video.mp4') as video:
        while True:
            success, frame = video.read()
            if not success:
                break

            results = engine.process(frame)
            if results:
                landmarks = engine.extract_landmarks(results)
                metrics = analyzer.analyze_frame(landmarks)

                print(f"Velocity: {metrics['com_velocity']:.4f}")
                print(f"Stability: {metrics['com_sway']:.4f}")

summary = analyzer.get_summary()
print(f"Total progress: {summary['total_vertical_progress']:.3f}")
```

### Video Quality Validation

```python
from climb_sensei import check_video_quality

# Validate video before processing
report = check_video_quality('video.mp4', deep_check=True)

if report.is_valid:
    # Process with climb-sensei
    pass
else:
    print("Quality issues:", report.issues)
```

Or use the CLI:

```bash
# Check video quality
python scripts/check_video_quality.py video.mp4 --deep --json report.json
```

## Documentation

📚 **Full documentation**: https://jpsferreira.github.io/climb-sensei

- [Installation Guide](https://jpsferreira.github.io/climb-sensei/installation/)
- [Quick Start Tutorial](https://jpsferreira.github.io/climb-sensei/quickstart/)
- [Usage Examples](https://jpsferreira.github.io/climb-sensei/usage/)
- [Metrics Reference](https://jpsferreira.github.io/climb-sensei/metrics/) - Complete guide to all 25+ metrics
- [API Reference](https://jpsferreira.github.io/climb-sensei/api/)
- [Architecture Overview](https://jpsferreira.github.io/climb-sensei/architecture/)
- [Development Guide](https://jpsferreira.github.io/climb-sensei/development/)

## Available Metrics

climb-sensei tracks 25+ climbing metrics:

- **Movement**: Velocity, sway, jerk, body angle, hand/foot span, vertical progress
- **Efficiency**: Movement economy, lock-off detection, rest positions
- **Biomechanics**: 8 joint angles (elbows, shoulders, knees, hips)
- **Fatigue**: Quality degradation scoring

See the [Metrics Reference](https://jpsferreira.github.io/climb-sensei/metrics/) for complete documentation.

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
make test

# Run linting and formatting
make check

# Build documentation
uv run mkdocs serve
```

See the [Development Guide](https://jpsferreira.github.io/climb-sensei/development/) for detailed instructions.

## Requirements

- Python 3.12+
- mediapipe >= 0.10.30
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- tqdm >= 4.66.0

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see the [Development Guide](https://jpsferreira.github.io/climb-sensei/development/) for guidelines.

---

**Documentation**: https://jpsferreira.github.io/climb-sensei
**Source Code**: https://github.com/jpsferreira/climb-sensei
**Bug Reports**: https://github.com/jpsferreira/climb-sensei/issues
