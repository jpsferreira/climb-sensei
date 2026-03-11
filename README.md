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
- ✅ **Quality Validation**: Video and tracking quality checks
- 🔌 **Modular Services**: Production-ready REST API architecture
- 🧪 **Tested**: Comprehensive test suite with high code coverage

## Architecture

**Service-Oriented Design** for production-ready climbing analysis:

- 🏭 **Independent Services** - VideoQuality, TrackingQuality, ClimbingAnalysis
- 🔌 **Pluggable Calculators** - Stability, Progress, Efficiency, Technique, JointAngles
- 🧪 **Fully Tested** - 209 tests, high coverage
- 📦 **Production Ready** - Async support, clean separation, easy to scale

📖 **See [API_GUIDE.md](API_GUIDE.md) for detailed examples**

## Quick Start

### Installation

```bash
# Install uv (if not already installed)
pip install uv

# Install climb-sensei
uv sync
```

### Web App (Easiest!)

Launch the web interface to upload videos and view results in your browser:

```bash
python run_app.py
```

Then open `http://localhost:8000` and upload a climbing video!

**Features:**

- � **User Authentication** - Email/password or Google OAuth sign-in
- 📤 Drag-and-drop video upload
- ⚙️ Select which analyses to run (metrics, video, quality checks)
- 📊 Interactive dashboard with charts
- 🎯 Goals and progress tracking
- 📈 Performance metrics over time
- 💾 Session management
- 🎥 View annotated videos in browser

See [app/README.md](app/README.md) for details.

**Authentication Options:**

- Email/password registration (works immediately)
- Google OAuth sign-in (requires [setup](docs/GOOGLE_OAUTH_SETUP.md))

See [OAUTH_QUICK_START.md](docs/OAUTH_QUICK_START.md) to get started.

### Command Line

```bash
# Quick terminal analysis
python scripts/analyze_climb.py climbing_video.mp4

# Create annotated video with dashboard
python scripts/analyze_climb.py climbing_video.mp4 --video output.mp4

# Export JSON data
python scripts/analyze_climb.py climbing_video.mp4 --json analysis.json
```

### Python API

**Use the modern service-oriented architecture for all new code:**

```python
from climb_sensei.services import (
    VideoQualityService,
    TrackingQualityService,
    ClimbingAnalysisService,
)
from climb_sensei.pose_engine import PoseEngine
from climb_sensei.video_io import VideoReader

# Initialize services (stateless, reusable)
video_quality = VideoQualityService()
tracking_quality = TrackingQualityService()
climbing = ClimbingAnalysisService()

# Validate video first (fast fail)
quality_report = video_quality.analyze_sync("video.mp4")
if not quality_report.is_valid:
    print("Invalid video:", quality_report.issues)
    exit(1)

# Extract landmarks once
landmarks = []
pose_engine = PoseEngine()
with VideoReader("video.mp4") as reader:
    fps = reader.fps
    while True:
        success, frame = reader.read()
        if not success:
            break
        result = pose_engine.process(frame)
        if result and result.pose_landmarks:
            landmarks.append(pose_engine.extract_landmarks(result))
        else:
            landmarks.append(None)
pose_engine.close()

# Use services independently (compose as needed)
tracking_report = tracking_quality.analyze_from_landmarks(landmarks)
analysis = climbing.analyze(landmarks, fps=fps)

# Access results
print(f"Max height: {analysis.summary.max_height:.2f}")
print(f"Quality: {tracking_report.quality_level}")
```

**Key advantages:**

- ✅ Independent services (use only what you need)
- ✅ Perfect for REST APIs and microservices
- ✅ Async support for I/O operations
- ✅ Easy to test and mock
- ✅ Plugin-based calculator system
- ✅ Production-ready architecture

**See [API_GUIDE.md](API_GUIDE.md) for detailed examples.**

**Two-Phase Pattern: For efficient video generation**

When you need to generate multiple outputs from one video (metrics + annotated video), extract landmarks once:

```python
from climb_sensei.services import ClimbingAnalysisService
from climb_sensei.pose_engine import PoseEngine
from climb_sensei.video_io import VideoReader

# Phase 1: Extract landmarks once (expensive)
landmarks = []
pose_results = []
pose_engine = PoseEngine()

with VideoReader("video.mp4") as reader:
    fps = reader.fps
    while True:
        success, frame = reader.read()
        if not success:
            break
        result = pose_engine.process(frame)
        pose_results.append(result)  # Save for video generation
        if result and result.pose_landmarks:
            landmarks.append(pose_engine.extract_landmarks(result))
        else:
            landmarks.append(None)

pose_engine.close()

# Phase 2: Use landmarks multiple times (fast, parallel!)
service = ClimbingAnalysisService()
analysis = service.analyze(landmarks, fps=fps)

# Reuse pose_results for video generation (no re-processing!)
# See scripts/analyze_climb.py for complete example
```

**Benefits:**

- 50% faster when generating video output
- Enables parallel processing
- Perfect for backend APIs with multiple outputs

**Custom Workflows: Direct component access**

```python
from climb_sensei import PoseEngine, VideoReader
from climb_sensei.domain.calculators import StabilityCalculator, ProgressCalculator

# Use only specific components you need
stability_calc = StabilityCalculator(fps=30.0)
progress_calc = ProgressCalculator(fps=30.0)

pose_engine = PoseEngine()
with VideoReader('climbing_video.mp4') as video:
    for frame in video:
        result = pose_engine.process(frame)
        if result and result.pose_landmarks:
            landmarks = pose_engine.extract_landmarks(result)

            stability_metrics = stability_calc.calculate(landmarks)
            progress_metrics = progress_calc.calculate(landmarks)

            print(f"Sway: {stability_metrics['com_sway']:.4f}")
            print(f"Progress: {progress_metrics['vertical_progress']:.3f}")

pose_engine.close()
```

.services import VideoQualityService

service = VideoQualityService()

# Validate before processing

report = service.analyze_sync('video.mp4'
from climb_sensei import check_video_quality

# Validate video before processing

report = check_video_quality('video.mp4', deep_check=True)

if report.is_valid: # Process with climb-sensei
pass
else:
print("Quality issues:", report.issues)

````

Or use the CLI:

```bash
# Check video quality
python scripts/check_video_quality.py video.mp4 --deep --json report.json
````

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
