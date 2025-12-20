# Scripts

This directory contains the command-line interface for climb-sensei.

## analyze_climb.py

Unified script for analyzing climbing videos. Processes video to extract performance metrics, with options to export JSON data and/or create annotated videos with animated dashboards.

### Usage

**Text summary only (fast):**

```bash
python scripts/analyze_climb.py climbing.mp4
```

**Export detailed JSON data:**

```bash
python scripts/analyze_climb.py climbing.mp4 --json analysis.json
```

**Create annotated video with 8 real-time metric plots:**

```bash
python scripts/analyze_climb.py climbing.mp4 --video output.mp4
```

**Customize dashboard:**

```bash
# Dashboard position
python scripts/analyze_climb.py climbing.mp4 --video output.mp4 --position left

# Add text overlay
python scripts/analyze_climb.py climbing.mp4 --video output.mp4 --show-text
```

**Both JSON and video in one pass:**

```bash
python scripts/analyze_climb.py climbing.mp4 --json data.json --video output.mp4
```
