# Scripts

This directory contains utility scripts for working with climb-sensei.

## process_video.py

Process a video file through the pose estimator and generate an output video with pose landmarks drawn on it.

### Usage

Basic usage:
```bash
python scripts/process_video.py input_video.mp4 output_video.mp4
```

With custom confidence thresholds:
```bash
python scripts/process_video.py input.mp4 output.mp4 --detection-conf 0.7 --tracking-conf 0.6
```

Quiet mode (no progress updates):
```bash
python scripts/process_video.py input.mp4 output.mp4 --quiet
```

With climbing metrics overlay on video:
```bash
python scripts/process_video.py input.mp4 output.mp4 --show-metrics
```

Save metrics to JSON file:
```bash
python scripts/process_video.py input.mp4 output.mp4 --metrics
```

Full analysis (metrics overlay on video AND JSON file):
```bash
python scripts/process_video.py input.mp4 output.mp4 --show-metrics --metrics
```

### Arguments

- `input`: Path to input video file (required)
- `output`: Path to save output video file (required)
- `--detection-conf`: Minimum detection confidence, 0.0-1.0 (default: 0.5)
- `--tracking-conf`: Minimum tracking confidence, 0.0-1.0 (default: 0.5)
- `--show-metrics`: Overlay real-time metrics on the video
- `--metrics`: Calculate and save metrics to JSON file
- `--metrics-output`: Path for metrics JSON file (default: same name as output video)
- `--include-head`: Include head keypoints (default: False, climbing-focused)
- `--quiet`: Suppress progress updates

**Note:** Temporal smoothing is always enabled via MediaPipe's built-in VIDEO mode filtering, which reduces jitter while maintaining responsiveness.

### Example

```bash
# Download a sample climbing video
# Process it through the pose estimator
python scripts/process_video.py climbing_video.mp4 climbing_analyzed.mp4

# Use higher confidence for better quality
python scripts/process_video.py climbing_video.mp4 climbing_analyzed.mp4 --detection-conf 0.8
```

### Output

The script will:
1. Read the input video frame by frame
2. Detect pose landmarks in each frame with built-in temporal smoothing (VIDEO mode)
3. Draw the landmarks and skeleton connections on each frame
4. (Optional) Overlay real-time metrics on the video showing:
   - Current frame joint angles (elbows, knees, hip)
   - Maximum reach distance
   - Lock-off detection (when arm angle < 90°)
   - Running averages of all metrics
5. (Optional) Save detailed per-frame metrics to JSON file
6. Save the annotated video

### Metrics Overlay

When using `--show-metrics`, the video will display a real-time overlay showing:

**Current Frame:**
- Frame number
- Left/Right elbow angles
- Left/Right knee angles
- Maximum reach distance
- Lock-off indicators (red text when detected)

**Running Averages:**
- Average left/right elbow angles
- Average maximum reach
- Average body extension

This makes it easy to analyze climbing technique while watching the video!
4. Save the annotated frames to the output video
5. Display progress updates and final statistics

The output video will have the same resolution and frame rate as the input video.
