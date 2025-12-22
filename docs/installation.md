# Installation

## Using uv (Recommended)

This project uses `uv` for fast, reliable package management:

```bash
# Install uv (if not already installed)
pip install uv

# Install the package
uv sync

# Or install with development dependencies
uv sync --extra dev
```

## Using pip

```bash
pip install climb-sensei
```

## From Source

```bash
# Clone the repository
git clone https://github.com/jpsferreira/climb-sensei.git
cd climb-sensei

# Install with uv
uv sync

# Or with pip
pip install -e .
```

## Development Installation

For development, install with all extras:

```bash
# Using uv
uv sync --extra dev

# Or using pip
pip install -e ".[dev]"
```

This includes:

- pytest & pytest-cov for testing
- ruff for linting
- black for code formatting
- pre-commit for git hooks
- mkdocs-material for documentation

## Verify Installation

```bash
# Run the demo
uv run python -m climb_sensei

# Or if installed with pip
python -m climb_sensei
```

## Requirements

- Python 3.12 or higher
- macOS, Linux, or Windows
- Webcam or video files for analysis

### Core Dependencies

- **mediapipe** >= 0.10.30: Pose estimation
- **opencv-python** >= 4.8.0: Video I/O and visualization
- **numpy** >= 1.24.0: Numerical computations
- **tqdm** >= 4.66.0: Progress bars

### Development Dependencies

- **pytest** >= 9.0.0: Testing framework
- **pytest-cov** >= 7.0.0: Coverage reporting
- **black** >= 24.0.0: Code formatting
- **ruff** >= 0.9.0: Linting
- **pre-commit** >= 4.0.0: Git hooks
- **mkdocs-material** >= 9.0.0: Documentation
