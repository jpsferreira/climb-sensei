# Quick Start

## CLI Usage

The fastest way to analyze a climbing video is using the command-line interface:

### Basic Analysis

```bash
# Quick terminal summary (fast)
python scripts/analyze_climb.py climbing_video.mp4
```

This outputs climbing metrics directly to the terminal.

### Export Data

```bash
# Export detailed JSON data
python scripts/analyze_climb.py climbing_video.mp4 --json analysis.json
```

The JSON file contains all frame-by-frame metrics and summary statistics.

### Create Annotated Video

```bash
# Create video with metrics dashboard on the side (default: no overlay)
python scripts/analyze_climb.py climbing_video.mp4 --video output.mp4

# Customize dashboard position (left or right)
python scripts/analyze_climb.py climbing_video.mp4 --video output.mp4 --position left

# Use overlay mode instead of side-by-side
python scripts/analyze_climb.py climbing_video.mp4 --video output.mp4 --overlay

# Add text overlay with current metric values
python scripts/analyze_climb.py climbing_video.mp4 --video output.mp4 --show-text
```

### Combined Export

```bash
# Export both JSON and video in one pass
python scripts/analyze_climb.py climbing_video.mp4 --json data.json --video output.mp4
```

## Python API

### Basic Usage

```python
from climb_sensei import PoseEngine, VideoReader, ClimbingAnalyzer

# Initialize analyzer
analyzer = ClimbingAnalyzer(window_size=30, fps=30)

with PoseEngine() as engine:
    with VideoReader('climbing_video.mp4') as video:
        while True:
            success, frame = video.read()
            if not success:
                break

            # Detect pose
            results = engine.process(frame)
            if results:
                # Extract landmarks
                landmarks = engine.extract_landmarks(results)

                # Analyze frame - get all metrics
                metrics = analyzer.analyze_frame(landmarks)
                print(f"Velocity: {metrics['com_velocity']:.4f}")
                print(f"Stability: {metrics['com_sway']:.4f}")
                print(f"Progress: {metrics['vertical_progress']:.3f}")

# Get summary statistics
summary = analyzer.get_summary()
print(f"Average speed: {summary['avg_velocity']:.4f}")
print(f"Total vertical progress: {summary['total_vertical_progress']:.3f}")
```

### Running the Demo

```bash
# Run the interactive demo
uv run python -m climb_sensei
```

The demo will:

1. Open your webcam
2. Detect your pose in real-time
3. Display pose landmarks and basic metrics
4. Press 'q' to quit

## Next Steps

- Read the [Usage Guide](usage.md) for detailed examples
- Check the [Metrics Reference](metrics.md) to understand all available metrics
- Explore the [API Reference](api.md) for complete documentation
