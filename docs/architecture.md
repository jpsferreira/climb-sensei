# Architecture

climb-sensei follows strict **Separation of Concerns** principles for maintainability and testability.

## Design Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Stateless Functions**: Pure functions in biomechanics module
3. **Dependency Injection**: Context managers for resource management
4. **Type Safety**: Comprehensive type hints throughout
5. **Test Coverage**: 82% coverage with 164 unit tests

## Module Overview

### config.py

**Purpose**: Application-wide configuration and constants

- Defines MediaPipe confidence thresholds
- Video encoding parameters
- Metric calculation constants
- Color schemes and visualization settings

**Key Exports**: Configuration dataclasses and constants

---

### video_io.py

**Purpose**: Video input/output operations

**Responsibilities**:

- Read video files with `VideoReader`
- Write video files with `VideoWriter`
- Expose frame properties (fps, dimensions)
- Handle codec and format conversion

**Key Classes**:

- `VideoReader`: Context manager for reading videos
- `VideoWriter`: Context manager for writing videos

**No Business Logic**: Pure I/O operations only

---

### pose_engine.py

**Purpose**: MediaPipe pose detection wrapper

**Responsibilities**:

- Initialize MediaPipe Pose model
- Process frames for pose detection
- Extract normalized landmark coordinates
- Manage model lifecycle

**Key Class**: `PoseEngine`

**Single Responsibility**: Wraps MediaPipe, nothing else

---

### biomechanics.py

**Purpose**: Pure mathematical calculations

**Characteristics**:

- All functions are **stateless**
- No side effects
- Fully testable with simple inputs
- No dependencies on other modules

**Functions**:

- `calculate_joint_angle()`: 3-point angle calculation
- `calculate_reach_distance()`: Euclidean distance
- `calculate_center_of_mass()`: Weighted average position
- `calculate_limb_angles()`: All 8 joint angles
- `calculate_total_distance_traveled()`: Path length

**Philosophy**: If it's math, it goes here

---

### metrics.py

**Purpose**: Temporal analysis and stateful tracking

**Key Class**: `ClimbingAnalyzer`

**Responsibilities**:

- Track metrics over time (stateful)
- Calculate moving averages
- Detect patterns (lock-offs, rest positions)
- Compute summary statistics
- Maintain frame history

**Design Pattern**: Object-oriented state management

**Dependencies**: Uses `biomechanics.py` for calculations

---

### metrics_viz.py

**Purpose**: Metrics dashboard visualization

**Responsibilities**:

- Create time-series plots
- Generate dashboard layouts
- Compose side-by-side visualizations
- Format metric displays

**Functions**:

- `create_metrics_dashboard()`: Generate plots
- `compose_frame_with_dashboard()`: Side-by-side layout

**Presentation Layer**: Purely visual output

---

### viz.py

**Purpose**: Pose rendering and annotation

**Responsibilities**:

- Draw pose landmarks on frames
- Render skeleton connections
- Add text overlays
- Dashboard overlay composition

**Functions**:

- `draw_pose_landmarks()`: Render pose visualization
- `overlay_metrics_dashboard()`: Alpha-blend dashboard

**Presentation Layer**: Visual pose feedback

---

## Data Flow

```
Input Video
    ↓
VideoReader → Frame
    ↓
PoseEngine → Landmarks (33 points)
    ↓
biomechanics.py → Joint Angles, Distances
    ↓
ClimbingAnalyzer → Metrics Dictionary
    ↓
    ├→ metrics_viz.py → Dashboard
    ├→ viz.py → Annotated Frame
    └→ Summary Statistics
    ↓
VideoWriter → Output Video
```

## Testing Strategy

### Unit Tests (164 tests, 82% coverage)

**biomechanics.py**: Pure function testing

- Input/output validation
- Edge cases (zero distances, collinear points)
- Mathematical correctness

**metrics.py**: State management testing

- Moving average windows
- Pattern detection accuracy
- Summary calculation correctness

**video_io.py**: I/O testing

- File handling
- Property exposure
- Codec compatibility

**pose_engine.py**: Integration testing

- MediaPipe initialization
- Landmark extraction
- Resource cleanup

### Test Organization

```
tests/
├── test_biomechanics.py       # Pure functions
├── test_metrics.py            # ClimbingAnalyzer
├── test_video_io.py          # Video I/O
├── test_pose_engine.py       # Pose detection
├── test_viz.py               # Visualization
└── test_patterns.py          # Design patterns
```

## Design Patterns

### Facade Pattern

**Location**: `__init__.py`

Simplifies the API by exposing only essential classes:

```python
from climb_sensei import PoseEngine, ClimbingAnalyzer
# vs
from climb_sensei.pose_engine import PoseEngine
from climb_sensei.metrics import ClimbingAnalyzer
```

### Builder Pattern

**Location**: `metrics.py`

`ClimbingAnalyzer` builds complex state over time:

```python
analyzer = ClimbingAnalyzer(window_size=30, fps=30)
for landmarks in frames:
    metrics = analyzer.analyze_frame(landmarks)  # Builds internal state
summary = analyzer.get_summary()  # Final product
```

### Context Manager Pattern

**Location**: `video_io.py`, `pose_engine.py`

Automatic resource management:

```python
with VideoReader('input.mp4') as video:
    # Resources automatically cleaned up
```

### Repository Pattern

**Location**: `metrics.py`

`ClimbingAnalyzer` stores and retrieves historical data:

```python
history = analyzer.get_history()  # Retrieve all stored data
summary = analyzer.get_summary()  # Aggregate view
```

## Extensibility

### Adding New Metrics

1. **Pure Calculation**: Add function to `biomechanics.py`
2. **Temporal Tracking**: Extend `ClimbingAnalyzer` in `metrics.py`
3. **Visualization**: Add plot to `metrics_viz.py`
4. **Tests**: Add test coverage

Example:

```python
# 1. biomechanics.py
def calculate_leg_extension(landmarks: List[Tuple[float, float]]) -> float:
    """Pure calculation"""
    return distance

# 2. metrics.py
class ClimbingAnalyzer:
    def analyze_frame(self, landmarks):
        leg_extension = calculate_leg_extension(landmarks)
        self._leg_extensions.append(leg_extension)
        return {'leg_extension': leg_extension, ...}

# 3. metrics_viz.py
def create_metrics_dashboard(history, ...):
    ax.plot(history['leg_extensions'])
```

### Adding New Visualization

Create new function in `viz.py` or `metrics_viz.py`:

```python
def draw_custom_overlay(frame, data):
    # Custom visualization logic
    return annotated_frame
```

## Performance Considerations

- **MediaPipe**: GPU acceleration when available
- **NumPy**: Vectorized operations for biomechanics
- **OpenCV**: Hardware video decoding
- **Moving Averages**: O(1) amortized with deque

## Future Architecture Plans

- [ ] Plugin system for custom metrics
- [ ] Async video processing pipeline
- [ ] Real-time streaming support
- [ ] Multi-person tracking
- [ ] 3D pose reconstruction
