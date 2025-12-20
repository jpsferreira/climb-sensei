# Development Guide

## Setup

### Prerequisites

- Python 3.12 or 3.13
- [uv](https://github.com/astral-sh/uv) package manager
- Git

### Clone and Install

```bash
# Clone repository
git clone https://github.com/jpsferreira/climb-sensei.git
cd climb-sensei

# Install with dev dependencies
uv sync --extra dev

# Install pre-commit hooks
make pre-commit-install
```

## Development Tools

### Makefile Targets

The project includes a comprehensive Makefile with 19 targets:

```bash
# Install dependencies
make install          # Production dependencies only
make install-dev      # Include dev dependencies

# Code Quality
make check           # Run all checks (ruff + black)
make lint            # Run ruff linter
make lint-fix        # Auto-fix linting issues
make format          # Format with black
make format-check    # Check formatting without changes

# Testing
make test            # Run all tests
make test-fast       # Run without coverage
make coverage        # Generate coverage report
make test-watch      # Run tests in watch mode

# Pre-commit
make pre-commit-install    # Install git hooks
make pre-commit           # Run all hooks manually

# Maintenance
make clean           # Remove build artifacts
make update          # Update dependencies
make lock            # Update uv.lock

# Build
make build           # Build distribution packages
make qa              # Quality assurance (check + test)

# Run
make run             # Run the demo
make analyze         # Analyze a video (VIDEO=path/to/video.mp4)
```

### Using Makefile

```bash
# Quick quality check
make qa

# Full development cycle
make install-dev
make pre-commit-install
make test
make check

# Before committing
make qa
```

## Code Quality

### Linting with Ruff

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Or use make
make lint-fix
```

### Formatting with Black

```bash
# Check formatting
uv run black --check .

# Format code
uv run black .

# Or use make
make format
```

### Pre-commit Hooks

Automatically run checks before each commit:

```bash
# Install hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files

# Or use make
make pre-commit
```

Configured hooks:

- **ruff**: Linting and auto-fixes
- **black**: Code formatting
- **prettier**: YAML/Markdown formatting
- **yaml-check**: YAML syntax validation
- **toml-check**: TOML syntax validation
- **trailing-whitespace**: Remove trailing spaces
- **end-of-file-fixer**: Ensure newline at EOF

## Testing

### Running Tests

```bash
# All tests with coverage
uv run pytest tests/ --cov

# Verbose output
uv run pytest tests/ -v

# Specific test file
uv run pytest tests/test_biomechanics.py

# Or use make
make test
```

### Test Organization

```
tests/
├── __init__.py
├── test_biomechanics.py      # Pure function tests
├── test_metrics.py           # ClimbingAnalyzer tests
├── test_video_io.py          # Video I/O tests
├── test_pose_engine.py       # Pose detection tests
├── test_viz.py               # Visualization tests
└── test_patterns.py          # Design pattern tests
```

### Coverage Reports

```bash
# Terminal coverage report
make coverage

# HTML coverage report
uv run pytest tests/ --cov --cov-report=html
open htmlcov/index.html
```

Current coverage: **82%** (164 tests)

### Writing Tests

Follow existing patterns:

```python
import pytest
from climb_sensei.biomechanics import calculate_joint_angle

def test_joint_angle_calculation():
    """Test basic angle calculation."""
    point_a = (0, 0)
    point_b = (0, 1)
    point_c = (1, 1)

    angle = calculate_joint_angle(point_a, point_b, point_c)
    assert abs(angle - 90.0) < 0.01

def test_joint_angle_edge_cases():
    """Test edge cases."""
    # Collinear points
    angle = calculate_joint_angle((0,0), (1,1), (2,2))
    assert abs(angle - 180.0) < 0.01
```

## Documentation

### Building Docs

```bash
# Install mkdocs (included in dev dependencies)
uv sync --extra dev

# Serve docs locally
uv run mkdocs serve

# Build static site
uv run mkdocs build

# Deploy to GitHub Pages
uv run mkdocs gh-deploy
```

### Writing Documentation

Documentation uses MkDocs with Material theme.

**File locations**:

- `docs/index.md`: Homepage
- `docs/installation.md`: Installation guide
- `docs/usage.md`: Usage examples
- `docs/api.md`: API reference
- `docs/metrics.md`: Metrics documentation
- `docs/architecture.md`: Architecture overview
- `docs/development.md`: This file

**Editing tips**:

````markdown
# Use admonitions for notes

!!! note
This is a note

!!! warning
This is a warning

# Code blocks with syntax highlighting

```python
from climb_sensei import PoseEngine
```
````

# Math equations with KaTeX

Inline: $x^2 + y^2 = z^2$
Block: $$\frac{vertical\_progress}{total\_distance}$$

````

## CI/CD

### GitHub Actions Workflows

Four automated workflows:

#### 1. CI (`ci.yaml`)
- **Triggers**: Push to main, pull requests
- **Jobs**:
  - Quality: Ruff + Black checks
  - Test: Python 3.12 & 3.13
  - Coverage: Upload to Codecov

#### 2. Pre-commit (`pre-commit.yaml`)
- **Triggers**: Push, pull requests
- **Action**: Run all pre-commit hooks

#### 3. Release (`release.yaml`)
- **Triggers**: GitHub release published
- **Action**: Build and publish to PyPI

#### 4. Codecov Validation (`validate-codecov.yaml`)
- **Triggers**: Changes to codecov.yaml
- **Action**: Validate configuration

### Local CI Simulation

```bash
# Run quality checks
make check

# Run tests (both Python versions locally)
uv run pytest tests/

# Run pre-commit hooks
make pre-commit
````

## Release Process

1. **Update Version**

   ```bash
   # Update version in pyproject.toml
   version = "0.2.0"
   ```

2. **Update Changelog**

   ```bash
   # Add release notes
   git commit -m "Release v0.2.0"
   ```

3. **Create Git Tag**

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

4. **Create GitHub Release**

   - Go to GitHub Releases
   - Create new release from tag
   - Add release notes
   - Publish release

5. **Automatic PyPI Upload**
   - GitHub Actions automatically builds and publishes
   - Uses trusted publishing (no API key needed)

## Code Style Guidelines

### Type Hints

Always use type hints:

```python
def calculate_velocity(
    current_pos: Tuple[float, float],
    previous_pos: Tuple[float, float],
    fps: int
) -> float:
    """Calculate velocity between positions."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def analyze_frame(self, landmarks: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Analyze a single frame of pose landmarks.

    Args:
        landmarks: List of (x, y) coordinate tuples for 33 pose landmarks.

    Returns:
        Dictionary containing all computed metrics for this frame.

    Raises:
        ValueError: If landmarks list is invalid.
    """
    ...
```

### Naming Conventions

- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Private**: `_leading_underscore`

### Code Organization

1. **Imports** (standard → third-party → local)
2. **Constants**
3. **Classes/Functions**
4. **Main block** (if script)

## Troubleshooting

### Common Issues

**Issue**: Pre-commit hooks fail with "command not found"

```bash
# Solution: Reinstall dev dependencies
uv sync --extra dev
make pre-commit-install
```

**Issue**: Tests fail with import errors

```bash
# Solution: Install package in editable mode
uv sync --extra dev
```

**Issue**: Coverage report shows missing lines

```bash
# Solution: Check if code is actually executed
uv run pytest tests/ --cov --cov-report=term-missing
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run quality checks: `make qa`
5. Commit changes: `git commit -m "Add amazing feature"`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open pull request

### Pull Request Checklist

- [ ] Code follows style guidelines (ruff + black)
- [ ] Added tests for new functionality
- [ ] All tests pass (`make test`)
- [ ] Documentation updated (if needed)
- [ ] Pre-commit hooks pass
- [ ] Coverage maintained or improved

## Resources

- [uv Documentation](https://github.com/astral-sh/uv)
- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
- [Ruff](https://docs.astral.sh/ruff/)
- [pytest](https://docs.pytest.org/)
