# Metrics Reference

Complete documentation of all 25+ climbing metrics available in climb-sensei.

## Core Movement Metrics

### Hip Height

**Key**: `hip_height`
**Type**: `float`
**Units**: Normalized coordinates (0.0 - 1.0)

Current vertical position of the hips. In normalized MediaPipe coordinates where 0.0 is top of frame and 1.0 is bottom.

**Interpretation**:

- Lower values = higher position in frame
- Decreasing over time = climbing upward
- Used as the primary vertical position indicator

---

### Center of Mass Velocity

**Key**: `com_velocity`
**Type**: `float`
**Units**: Normalized units per second

Speed of movement calculated from center of mass displacement over time.

**Interpretation**:

- Higher values = faster movement
- Typical climbing: 0.01 - 0.1
- Very fast moves: > 0.15
- Near stationary: < 0.005

---

### Center of Mass Sway

**Key**: `com_sway`
**Type**: `float`
**Units**: Normalized units

Lateral instability measure - standard deviation of horizontal position over recent frames.

**Interpretation**:

- Lower values = more stable
- Excellent stability: < 0.01
- Good stability: 0.01 - 0.02
- Poor stability: > 0.03
- Indicates control and technique quality

---

### Jerk

**Key**: `jerk`
**Type**: `float`
**Units**: Normalized units per second²

Rate of change of acceleration - measures movement smoothness.

**Interpretation**:

- Lower values = smoother movement
- Smooth climbing: < 0.001
- Jerky movement: > 0.005
- Indicates fatigue or poor technique when high

---

### Body Angle

**Key**: `body_angle`
**Type**: `float`
**Units**: Degrees (0° - 90°)

Lean angle from vertical, measured from hip-shoulder axis.

**Calculation**: `arctan(abs(dx) / abs(dy))` where dx is horizontal distance and dy is vertical distance between shoulders and hips.

**Interpretation**:

- 0° = Perfectly vertical (straight up)
- 45° = 45-degree lean
- 90° = Horizontal (laying out)
- Typical rest position: 5-15°
- Steep overhang climbing: 30-60°

---

### Hand Span

**Key**: `hand_span`
**Type**: `float`
**Units**: Normalized units

Distance between left and right wrists.

**Interpretation**:

- Larger values = wider reach
- Changing rapidly = dynamic movement
- Small values = hands close together (mantling, jamming)

---

### Foot Span

**Key**: `foot_span`
**Type**: `float`
**Units**: Normalized units

Distance between left and right ankles.

**Interpretation**:

- Larger values = wider stance
- Very small values = feet together (stemming, crack climbing)

---

### Vertical Progress

**Key**: `vertical_progress`
**Type**: `float`
**Units**: Normalized units (cumulative)

Total vertical distance climbed from starting position.

**Interpretation**:

- Cumulative metric (increases over time)
- Measures total height gained
- Typical boulder problem: 0.3 - 0.6
- Full rope length: 2.0 - 3.0

---

## Efficiency & Technique Metrics

### Movement Economy

**Key**: `movement_economy`
**Type**: `float`
**Units**: Ratio (0.0 - 1.0)

Efficiency calculated as vertical progress divided by total distance traveled.

**Calculation**: `vertical_progress / total_distance_traveled`

**Interpretation**:

- 1.0 = Perfect efficiency (straight up)
- 0.7 - 0.9 = Excellent efficiency
- 0.5 - 0.7 = Good efficiency
- < 0.5 = Poor efficiency (too much lateral movement)

---

### Lock-off Detection

**Keys**: `is_lock_off`, `left_lock_off`, `right_lock_off`
**Type**: `bool`
**Units**: Boolean

Detects static bent-arm positions indicating lock-off holds.

**Detection Criteria**:

- Elbow angle < 110° (bent arm)
- Low velocity (< threshold)
- Sustained for multiple frames

**Interpretation**:

- `True` = Currently in lock-off
- `left_lock_off` / `right_lock_off` = Per-arm detection
- High lock-off percentage indicates strength-intensive climbing

---

### Rest Position Detection

**Key**: `is_rest_position`
**Type**: `bool`
**Units**: Boolean

Identifies low-stress vertical positions where climber can recover.

**Detection Criteria**:

- Body angle < 20° (nearly vertical)
- Low velocity (near stationary)

**Interpretation**:

- `True` = Currently resting
- High rest percentage = good route reading
- Low rest percentage on hard routes = may indicate pumped climbing

---

## Joint Angles

All joint angles are measured in degrees using the three-point angle calculation.

### Elbow Angles

**Keys**: `left_elbow`, `right_elbow`
**Type**: `float`
**Units**: Degrees (0° - 180°)

Angle formed by shoulder-elbow-wrist.

**Interpretation**:

- 180° = Fully extended (straight arm)
- 90° = 90-degree bend
- < 110° = Lock-off position
- Typical climbing: 120° - 160°

---

### Shoulder Angles

**Keys**: `left_shoulder`, `right_shoulder`
**Type**: `float`
**Units**: Degrees (0° - 180°)

Angle formed by hip-shoulder-elbow.

**Interpretation**:

- 90° = Arm perpendicular to body
- > 90° = Arm raised above shoulder
- < 90° = Arm below shoulder
- High angles (>120°) = reaching overhead

---

### Knee Angles

**Keys**: `left_knee`, `right_knee`
**Type**: `float`
**Units**: Degrees (0° - 180°)

Angle formed by hip-knee-ankle.

**Interpretation**:

- 180° = Fully extended leg
- 90° = Deep squat
- Typical climbing: 100° - 170°
- < 100° = High step or knee bar

---

### Hip Angles

**Keys**: `left_hip`, `right_hip`
**Type**: `float`
**Units**: Degrees (0° - 180°)

Angle formed by shoulder-hip-knee.

**Interpretation**:

- 180° = Fully extended body
- 90° = Bent at waist
- Typical climbing: 140° - 170°
- Low angles indicate hip flexibility moves

---

## Summary Statistics

Available via `analyzer.get_summary()`:

### Velocity Statistics

- **`avg_velocity`**: Mean velocity over entire climb
- **`max_velocity`**: Peak velocity reached
- **`min_velocity`**: Minimum velocity (excluding stops)

### Progress Metrics

- **`total_vertical_progress`**: Total height gained
- **`avg_movement_economy`**: Average efficiency ratio

### Technique Counts

- **`lock_off_count`**: Number of lock-off moments detected
- **`lock_off_percentage`**: Percentage of frames in lock-off
- **`rest_count`**: Number of rest positions
- **`rest_percentage`**: Percentage of frames resting

### Fatigue Indicator

- **`fatigue_score`**: Movement quality degradation score
  - 0.0 = No degradation
  - 0.5 = Moderate fatigue
  - > 1.0 = Significant fatigue

### Joint Angle Averages

- **`avg_joint_angles`**: Dictionary of average angles for all 8 joints

---

## Time-Series History

Available via `analyzer.get_history()`:

Returns a dictionary with complete frame-by-frame history:

```python
{
    'hip_heights': [0.8, 0.79, 0.78, ...],
    'velocities': [0.05, 0.06, 0.04, ...],
    'sways': [0.01, 0.015, 0.012, ...],
    'jerks': [0.0005, 0.0008, 0.0004, ...],
    'body_angles': [15.2, 18.5, 12.3, ...],
    'hand_spans': [0.35, 0.38, 0.42, ...],
    'foot_spans': [0.28, 0.30, 0.29, ...],
    'movement_economy': [0.85, 0.82, 0.88, ...],
    'lock_offs': [False, False, True, ...],
    'rest_positions': [False, True, True, ...],
    'joint_angles': {
        'left_elbow': [145, 148, 142, ...],
        'right_elbow': [150, 152, 149, ...],
        # ... all 8 joints
    }
}
```

---

## Usage Examples

### Analyzing Real-Time Performance

```python
metrics = analyzer.analyze_frame(landmarks)

# Check if climber is efficient
if metrics['movement_economy'] > 0.8:
    print("Excellent technique!")

# Detect fatigue
if metrics['jerk'] > 0.005:
    print("Movement becoming jerky - possible fatigue")

# Monitor stability
if metrics['com_sway'] > 0.03:
    print("Unstable - focus on control")
```

### Comparing Climb Attempts

```python
# After each climb
summary = analyzer.get_summary()

climbs = []
climbs.append({
    'avg_velocity': summary['avg_velocity'],
    'movement_economy': summary['avg_movement_economy'],
    'lock_off_percentage': summary['lock_off_percentage']
})

# Compare best attempt
best = max(climbs, key=lambda x: x['movement_economy'])
```
