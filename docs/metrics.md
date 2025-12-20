# Metrics Reference

Complete documentation of all 25+ climbing metrics including calculation methods, scientific rationale, and practical interpretation.

## Understanding the Coordinate System

All metrics use **normalized coordinates** (0.0 - 1.0) from MediaPipe Pose estimation:

- **X-axis**: 0.0 = left edge, 1.0 = right edge
- **Y-axis**: 0.0 = top edge, 1.0 = bottom edge (inverted from typical graphics)

This normalization ensures metrics work across different video resolutions and climbing wall sizes.

**Time-based metrics** use the video frame rate (default 30 fps) to convert frame-based measurements into per-second values.

---

## Core Movement Metrics

### Hip Height

**Key**: `hip_height`
**Type**: `float`
**Range**: 0.0 (top) to 1.0 (bottom)

#### Calculation

```python
left_hip_y = landmarks[LEFT_HIP].y
right_hip_y = landmarks[RIGHT_HIP].y
hip_height = (left_hip_y + right_hip_y) / 2.0
```

#### Rationale

The hips represent the body's center of mass more consistently than other landmarks (shoulders move with arm positions, knees with leg positions). Averaging both hips reduces noise from asymmetric positions like hip flags or body rotation.

#### Interpretation

- **Lower values** (→ 0.0) = higher in frame = climbing up
- **Higher values** (→ 1.0) = lower in frame = descending
- **Decreasing trend** = vertical progress
- **Constant value** = static position (rest/lock-off/stuck)

#### What You Can Learn

✅ **Total ascent**: `initial_hip_height - final_hip_height`
✅ **Climb efficiency**: Smooth decrease = good beta, fluctuations = wasted height
✅ **Rest positions**: Plateaus in the graph
✅ **Crux sections**: Prolonged plateaus or increases (downclimbing)

---

### Center of Mass Velocity

**Key**: `com_velocity`
**Type**: `float`
**Units**: Normalized units per second

#### Calculation

```python
# Center of mass from shoulders and hips
core_points = [left_shoulder, right_shoulder, left_hip, right_hip]
com_current = weighted_average(core_points)
com_previous = com_at_previous_frame

# Euclidean distance
dx = com_current.x - com_previous.x
dy = com_current.y - com_previous.y
distance = sqrt(dx² + dy²)

# Convert to velocity
velocity = distance / (1/fps)  # Distance per second
```

#### Rationale

Center of mass (COM) provides a single point representing the entire body's movement. Using core points (shoulders + hips) excludes limb movements that don't represent whole-body displacement. Velocity captures movement speed regardless of direction (up, down, or lateral).

#### Interpretation

- **0.01 - 0.1**: Normal climbing pace
- **> 0.15**: Very fast/dynamic moves
- **< 0.005**: Near stationary (rest, lock-off, stuck)
- **Spikes**: Dynamic movements, jumps, quick repositioning

#### What You Can Learn

✅ **Climbing style**: Consistent velocity = static climbing, spikes = dynamic
✅ **Fatigue onset**: Decreasing average velocity over time
✅ **Route difficulty**: Sections with low velocity = technical/difficult
✅ **Movement efficiency**: High velocity with low energy expenditure = skilled

---

### Center of Mass Sway

**Key**: `com_sway`
**Type**: `float`
**Units**: Normalized units (standard deviation)

#### Calculation

```python
# Collect horizontal positions over window (default 30 frames = 1 second)
com_x_positions = [com.x for com in last_30_frames]

# Standard deviation of lateral position
sway = std_dev(com_x_positions)
```

#### Rationale

Lateral stability is a key indicator of control and technique. Experienced climbers minimize unnecessary horizontal movement, keeping their center of mass aligned with the line of ascent. Standard deviation captures the variability in side-to-side movement over a sliding time window.

#### Interpretation

- **< 0.01**: Excellent stability (competition-level technique)
- **0.01 - 0.02**: Good stability (experienced climber)
- **0.02 - 0.03**: Moderate stability (intermediate)
- **> 0.03**: Poor stability (beginner, pumped, or very overhung)

#### What You Can Learn

✅ **Technical skill**: Lower sway = better body positioning
✅ **Route character**: High sway on slab = poor footwork, on overhang = expected
✅ **Fatigue indicator**: Increasing sway = losing control
✅ **Specific weaknesses**: Compare sway on different hold types (crimps vs slopers)

---

### Jerk

**Key**: `jerk`
**Type**: `float`
**Units**: Normalized units per second²

#### Calculation

```python
# Jerk is the rate of change of acceleration
# Requires at least 4 frames for numerical differentiation

# Velocities
v1 = distance(frame[n-3], frame[n-2]) * fps
v2 = distance(frame[n-2], frame[n-1]) * fps
v3 = distance(frame[n-1], frame[n]) * fps

# Accelerations
a1 = (v2 - v1) * fps
a2 = (v3 - v2) * fps

# Jerk
jerk = abs(a2 - a1) * fps
```

#### Rationale

Jerk measures movement smoothness. Smooth, controlled climbing has low jerk (gradual accelerations). Jerky movement indicates:

- Overgripping (tension-release cycles)
- Poor technique (fighting the wall instead of flowing)
- Fatigue (muscle control degradation)
- Fear/hesitation (stop-start movements)

This is a **second-order derivative**, making it sensitive to movement quality.

#### Interpretation

- **< 0.001**: Very smooth (expert technique)
- **0.001 - 0.005**: Normal climbing smoothness
- **> 0.005**: Jerky movement (technical issues or fatigue)
- **High variance**: Inconsistent movement quality

#### What You Can Learn

✅ **Technical proficiency**: Lower jerk = better movement control
✅ **Fatigue detection**: Increasing jerk = muscle control degradation
✅ **Learning progress**: Jerk decreases as climber learns route
✅ **Crux identification**: Jerk spikes at difficult sections

---

### Body Angle

**Key**: `body_angle`
**Type**: `float`
**Units**: Degrees (0° - 90°)

#### Calculation

```python
# Vector from hips to shoulders
hip_center = (left_hip + right_hip) / 2
shoulder_center = (left_shoulder + right_shoulder) / 2

dx = abs(shoulder_center.x - hip_center.x)  # Horizontal component
dy = abs(shoulder_center.y - hip_center.y)  # Vertical component

# Angle from vertical
body_angle = atan(dx / dy) * (180 / π)
```

#### Rationale

Body angle indicates lean from vertical. This metric reveals:

- **Wall angle compensation**: More lean on overhangs
- **Rest positions**: Near-vertical (0-15°) positions are less strenuous
- **Dynamic loading**: Large angles = more force on arms
- **Technique choices**: Keeping hips close (low angle) vs. away from wall

Using `atan(dx/dy)` gives angle from vertical (not horizontal), making 0° = vertical stance, which is more intuitive for climbing.

#### Interpretation

- **0-15°**: Vertical/slab stance (efficient, restable)
- **15-30°**: Moderate lean (typical vert climbing)
- **30-50°**: Significant lean (overhang, roof approach)
- **50-90°**: Extreme lean (steep overhang, horizontal roof)

#### What You Can Learn

✅ **Wall angle estimation**: Average body angle correlates with route steepness
✅ **Rest technique**: Low angles indicate good rest positions
✅ **Energy expenditure**: Higher angles = more arm loading = faster fatigue
✅ **Style analysis**: Consistent vs. variable angles = different climbing approaches

---

### Hand Span

**Key**: `hand_span`
**Type**: `float`
**Units**: Normalized distance

#### Calculation

```python
left_wrist = landmarks[LEFT_WRIST]
right_wrist = landmarks[RIGHT_WRIST]

dx = right_wrist.x - left_wrist.x
dy = right_wrist.y - left_wrist.y
hand_span = sqrt(dx² + dy²)
```

#### Rationale

Hand span indicates reach width and body compression. Wide spans suggest:

- Underclings or gastons
- Mantling or compression moves
- Open body position

Narrow spans suggest:

- Tight hand positions (chimneys, arêtes)
- Crossed hands (sideways movement)
- Compressed body position

#### Interpretation

- **> 0.4**: Very wide (full extension, underclings)
- **0.2 - 0.4**: Normal climbing reach
- **< 0.2**: Narrow (mantling, compression, or sideways)

#### What You Can Learn

✅ **Movement type**: Rapid changes = dynamic repositioning
✅ **Route character**: Consistently wide = horizontal traverse, narrow = vertical crack
✅ **Reach efficiency**: Compare span to vertical progress (wide ≠ always better)
✅ **Technique variety**: High variance = diverse move types

---

### Foot Span

**Key**: `foot_span`
**Type**: `float`
**Units**: Normalized distance

#### Calculation

```python
left_ankle = landmarks[LEFT_ANKLE]
right_ankle = landmarks[RIGHT_ANKLE]

dx = right_ankle.x - left_ankle.x
dy = right_ankle.y - left_ankle.y
foot_span = sqrt(dx² + dy²)
```

#### Rationale

Foot span reveals base of support and leg positioning:

- **Wide stance**: Stability, slab technique, stemming
- **Narrow stance**: Vertical climbing, drop knees, heel hooks

Changes in foot span indicate footwork adjustments and weight shifts.

#### Interpretation

- **> 0.3**: Wide stance (slab, stemming, stability)
- **0.1 - 0.3**: Normal footwork
- **< 0.1**: Feet together (flagging, drop knee, barn door prevention)

#### What You Can Learn

✅ **Base of support**: Wider = more stable but less mobile
✅ **Footwork quality**: Smooth transitions = good technique
✅ **Route type**: Wide spans = slab/chimney, narrow = overhang
✅ **Efficiency**: Excessive foot movement = wasted energy

---

### Vertical Progress

**Key**: `vertical_progress`
**Type**: `float`
**Units**: Normalized distance (cumulative)

#### Calculation

```python
initial_hip_height = hip_height_at_frame_0
current_hip_height = hip_height_at_current_frame

vertical_progress = initial_hip_height - current_hip_height
```

Since Y-axis is inverted (0=top), _decreasing_ hip height = climbing up, so progress is initial minus current.

#### Rationale

Cumulative metric tracking total height gained from start position. Unlike instantaneous hip height, this provides:

- Absolute climb progress
- Ability to detect downclimbing (negative progress)
- Performance comparison across attempts

#### Interpretation

- **Increasing**: Making upward progress
- **Plateauing**: Static position or horizontal traversing
- **Decreasing**: Downclimbing or lowering
- **Final value**: Total vertical distance climbed

#### What You Can Learn

✅ **Climb completion**: Compare to route height
✅ **Efficiency metric**: Use in economy calculation
✅ **Attempt comparison**: Normalize performance across tries
✅ **Crux identification**: Progress stalls = difficult sections

---

## Efficiency & Technique Metrics

### Movement Economy

**Key**: `movement_economy`
**Type**: `float`
**Units**: Ratio (0.0 - 1.0)

#### Calculation

```python
# Track cumulative distance traveled by COM
total_distance = sum(distance_between_consecutive_frames)

# Vertical progress (from Hip Height metric)
vertical_progress = initial_hip_height - current_hip_height

# Economy ratio
movement_economy = vertical_progress / total_distance
```

#### Rationale

Perfect efficiency would be moving straight up (economy = 1.0). In reality, climbers move laterally to reach holds, creating a less efficient path. Movement economy quantifies this:

- **High economy** (0.7-1.0): Efficient movement, good route reading, minimal deviation
- **Low economy** (< 0.5): Excessive lateral/vertical wandering, poor beta, wasted energy

This metric combines spatial efficiency with strategic climbing choices.

#### Interpretation

- **0.9 - 1.0**: Nearly perfect (impossible on real routes with holds)
- **0.7 - 0.9**: Excellent efficiency (competition level, well-known route)
- **0.5 - 0.7**: Good efficiency (normal recreational climbing)
- **< 0.5**: Poor efficiency (onsight, poor beta, beginner)

#### What You Can Learn

✅ **Route knowledge**: Increases with practice (better beta)
✅ **Climb grade impact**: More efficient on easier grades
✅ **Style differences**: Boulder (lower) vs. route (higher)
✅ **Energy expenditure**: Lower economy = faster fatigue

---

### Lock-off Detection

**Keys**: `is_lock_off`, `left_lock_off`, `right_lock_off`
**Type**: `bool`
**Units**: Boolean

#### Calculation

```python
# For each arm independently
elbow_angle = calculate_joint_angle(shoulder, elbow, wrist)
velocity = current_com_velocity

# Detection criteria
is_locked_off = (elbow_angle < 110°) AND (velocity < 0.005)

# Both arms
is_lock_off = left_lock_off OR right_lock_off
```

#### Rationale

Lock-offs are static strength positions where climbers hold themselves with bent arms while reaching for the next hold. Detection requires:

1. **Bent elbow** (< 110°): Holding with arm flexion, not straight-arm hanging
2. **Low velocity** (< 0.005): Static position, not dynamic movement

This identifies high-intensity strength moments that contribute to fatigue.

#### Interpretation

- **True**: Currently in static strength position
- **High percentage**: Route requires significant static strength
- **Frequent transitions**: Dynamic climbing with many lock-offs
- **Duration**: Time in lock-off = strength demand

#### What You Can Learn

✅ **Strength requirements**: High lock-off % = power-endurance route
✅ **Technique quality**: Fewer lock-offs = better efficiency (where possible)
✅ **Fatigue contribution**: Lock-offs are metabolically expensive
✅ **Training needs**: Frequent lock-offs = train static strength

---

### Rest Position Detection

**Key**: `is_rest_position`
**Type**: `bool`
**Units**: Boolean

#### Calculation

```python
body_angle = calculate_body_angle(landmarks)
velocity = current_com_velocity

# Detection criteria
is_rest_position = (body_angle < 20°) AND (velocity < 0.005)
```

#### Rationale

Rest positions are characterized by:

1. **Near-vertical body** (< 20°): Minimal arm loading, weight on skeleton
2. **Static position**: Not moving, allowing recovery

Identifying rests reveals route reading and pacing strategy. Good climbers find and use rests even on hard routes.

#### Interpretation

- **True**: Currently in recovery position
- **High percentage**: Route has many rests (easier) or climber is pacing well
- **Low percentage**: Sustained difficulty or poor route reading
- **Rest duration**: Longer = more recovery, but slower time

#### What You Can Learn

✅ **Route character**: Rest availability indicates sustained vs. intermittent difficulty
✅ **Pacing strategy**: Using rests = good tactics, rushing through = poor pacing
✅ **Fitness level**: Need for rests on easier routes = endurance limitation
✅ **Competitive analysis**: Rest usage in competition (risk vs. recovery trade-off)

---

## Joint Angle Metrics

All joint angles use the same calculation method but different landmark triplets. Understanding one explains all eight.

### Calculation Method (Generic)

```python
def calculate_joint_angle(point_a, point_b, point_c):
    """
    Calculate angle at point_b formed by points a-b-c.
    Uses law of cosines via dot product.
    """
    # Vectors from joint to adjacent landmarks
    ba = point_a - point_b
    bc = point_c - point_b

    # Dot product
    dot_product = ba.x * bc.x + ba.y * bc.y

    # Magnitudes
    mag_ba = sqrt(ba.x² + ba.y²)
    mag_bc = sqrt(bc.x² + bc.y²)

    # Angle from dot product formula: cos(θ) = (a·b)/(|a||b|)
    cos_angle = dot_product / (mag_ba * mag_bc)
    cos_angle = clamp(cos_angle, -1.0, 1.0)  # Numerical safety

    angle = arccos(cos_angle) * (180/π)  # Convert to degrees
    return angle
```

This returns the **interior angle** (0° - 180°) at the middle point.

---

### Elbow Angles

**Keys**: `left_elbow`, `right_elbow`
**Type**: `float`
**Units**: Degrees (0° - 180°)

#### Specific Calculation

```python
left_elbow_angle = calculate_joint_angle(
    left_shoulder,   # Point A
    left_elbow,      # Point B (vertex)
    left_wrist       # Point C
)
```

#### Interpretation

- **170° - 180°**: Nearly straight (passive hanging, efficient)
- **120° - 170°**: Slightly bent (active engagement)
- **90° - 120°**: Moderate flexion (holding position)
- **< 90°**: Deep flexion (lock-off, gaston, high power demand)

#### What You Can Learn

✅ **Arm efficiency**: More time with straight arms (>160°) = better technique
✅ **Lock-off identification**: Values < 110° during static holds
✅ **Fatigue progression**: Increasing average angle = arms tiring, straightening out
✅ **Move type**: Rapid changes = dynamic, stable values = static

---

### Shoulder Angles

**Keys**: `left_shoulder`, `right_shoulder`
**Type**: `float`
**Units**: Degrees (0° - 180°)

#### Specific Calculation

```python
left_shoulder_angle = calculate_joint_angle(
    left_hip,        # Point A
    left_shoulder,   # Point B (vertex)
    left_elbow       # Point C
)
```

#### Interpretation

- **< 90°**: Arm below shoulder (low reach, underclings)
- **90°**: Arm perpendicular to body (neutral)
- **> 90°**: Arm above shoulder (overhead reaches, typical climbing)
- **> 120°**: High reach (extended overhead, dyno preparation)

#### What You Can Learn

✅ **Reach height**: Higher angles = reaching high
✅ **Overhang adaptation**: Consistently high angles on steep terrain
✅ **Move variety**: Angle range shows movement diversity
✅ **Shoulder stress**: Prolonged extreme angles (>150° or <60°) = injury risk

---

### Knee Angles

**Keys**: `left_knee`, `right_knee`
**Type**: `float`
**Units**: Degrees (0° - 180°)

#### Specific Calculation

```python
left_knee_angle = calculate_joint_angle(
    left_hip,        # Point A
    left_knee,       # Point B (vertex)
    left_ankle       # Point C
)
```

#### Interpretation

- **170° - 180°**: Straight leg (standing on holds, heel hooks)
- **120° - 170°**: Slight bend (normal standing/stepping)
- **90° - 120°**: Moderate flexion (squatting, high step preparation)
- **< 90°**: Deep flexion (high steps, knee bars, drop knees)

#### What You Can Learn

✅ **Footwork type**: Deep flexion = high steps, straight = standing
✅ **Leg strength demand**: More time in 90-120° = quad-intensive
✅ **Drop knee detection**: One knee deep, other straight
✅ **Efficiency**: Excessive knee bend = inefficient leg use

---

### Hip Angles

**Keys**: `left_hip`, `right_hip`
**Type**: `float`
**Units**: Degrees (0° - 180°)

#### Specific Calculation

```python
left_hip_angle = calculate_joint_angle(
    left_shoulder,   # Point A
    left_hip,        # Point B (vertex)
    left_knee        # Point C
)
```

#### Interpretation

- **170° - 180°**: Fully extended (straight body, slab)
- **140° - 170°**: Normal climbing position
- **120° - 140°**: Moderate bend (bringing hips to wall)
- **< 120°**: Deep bend (compression moves, bicycles, knee-chest positions)

#### What You Can Learn

✅ **Body position**: Lower angles = hips closer to wall = better technique on overhangs
✅ **Compression moves**: Very low angles (< 120°) during compression
✅ **Flexibility requirements**: Sustained low angles = hip flexibility needed
✅ **Efficiency indicator**: Keeping hips in optimal for wall angle

---

## Summary Statistics

These are computed from the full history of frame-by-frame metrics.

### Average Metrics

All metrics prefixed with `avg_` are simple means:

```python
avg_velocity = mean(all_velocity_values)
avg_sway = mean(all_sway_values)
# etc.
```

**Use**: Characterize overall climb performance, compare attempts, track improvement.

### Maximum Metrics

Metrics like `max_velocity`, `max_sway` identify peak values:

```python
max_velocity = max(all_velocity_values)
max_jerk = max(all_jerk_values)
```

**Use**: Identify hardest moments, dynamic peaks, loss of control incidents.

### Count Metrics

`lock_off_count` and `rest_count` sum boolean frames:

```python
lock_off_count = sum(1 for frame in frames if frame.is_lock_off)
```

**Use**: Quantify specific technique occurrences.

### Percentage Metrics

Convert counts to percentages of total climb time:

```python
lock_off_percentage = (lock_off_count / total_frames) * 100
```

**Use**: Normalize for climb duration, enable cross-climb comparison.

### Fatigue Score

**Calculation**:

```python
# Split climb into thirds
first_third_quality = mean(movement_smoothness[:N/3])
last_third_quality = mean(movement_smoothness[-N/3:])

# Degradation ratio
fatigue_score = (first_third_quality - last_third_quality) / first_third_quality
```

**Interpretation**:

- **0.0**: No quality degradation
- **0.3**: Moderate fatigue
- **> 0.5**: Significant fatigue
- **Negative**: Actually improving (learning route during climb)

**Use**: Detect endurance limitations, compare climb pacing strategies.

---

## Practical Application Examples

### Analyzing a Training Session

```python
summary = analyzer.get_summary()

# Technique efficiency
if summary['avg_movement_economy'] > 0.7:
    print("✅ Good route reading and efficiency")
else:
    print("❌ Work on beta optimization")

# Endurance assessment
if summary['fatigue_score'] > 0.4:
    print("❌ Endurance limitation - train power endurance")

# Strength requirements
if summary['lock_off_percentage'] > 30:
    print("💪 Route requires significant static strength")
```

### Comparing Climb Attempts

```python
attempt1_economy = 0.65
attempt2_economy = 0.73

improvement = ((attempt2_economy - attempt1_economy) / attempt1_economy) * 100
print(f"Efficiency improved by {improvement:.1f}%")
```

### Identifying Weaknesses

```python
history = analyzer.get_history()

# Find high-sway sections
high_sway_frames = [i for i, s in enumerate(history['sways']) if s > 0.03]

# Cross-reference with video timestamps
problem_timestamps = [frame/fps for frame in high_sway_frames]
print(f"Loss of control at: {problem_timestamps} seconds")
```

### Route Characterization

```python
summary = analyzer.get_summary()

avg_body_angle = summary['avg_body_angle']
lock_off_pct = summary['lock_off_percentage']
rest_pct = summary['rest_percentage']

if avg_body_angle < 20 and rest_pct > 20:
    print("Route: Vertical, technical, good rests")
elif avg_body_angle > 40 and lock_off_pct > 25:
    print("Route: Overhung, powerful, sustained")
```

---

## Metric Relationships and Dependencies

### Dependent Metrics

Some metrics are calculated from others:

- **movement_economy** requires: `vertical_progress`, `total_distance_traveled`
- **jerk** requires: at least 4 frames of `velocity` history
- **is_lock_off** requires: `elbow_angle`, `velocity`
- **fatigue_score** requires: full climb history

### Inversely Correlated

- **velocity** ↑ → **sway** ↑ (faster = less stable)
- **body_angle** ↑ → **rest_percentage** ↓ (lean = harder)
- **movement_economy** ↑ → **total_distance** ↓ (efficient = less wandering)

### Independently Varying

- **hand_span** vs **foot_span**: Different body configurations
- **elbow_angle** vs **shoulder_angle**: Joint-specific positioning

---

## Limitations and Considerations

### MediaPipe Accuracy

- Occluded landmarks (behind wall/body): May cause brief metric errors
- 2D projection from 3D space: Distance metrics less accurate on angle to camera
- Confidence thresholds: Low-confidence frames should be filtered

### Metric Reliability

- **Most reliable**: Hip height, velocity, joint angles (use core visible landmarks)
- **Moderate**: Sway, body angle (sensitive to camera angle)
- **Least reliable**: Spans on routes with camera not perpendicular to wall

### Interpretation Context

Always consider:

- **Wall angle**: Overhangs naturally have higher body angles
- **Route style**: Slab vs. overhang climbing have different normal ranges
- **Climber experience**: Beginners have higher variance in all metrics
- **Video quality**: Poor lighting/resolution affects MediaPipe accuracy

---

## Future Metric Development

Potential additions:

- **3D depth estimation**: More accurate distance calculations
- **Reach efficiency**: Hand movement relative to height gained
- **Breathing rate**: From shoulder movement patterns
- **Chalk usage**: Hand stops indicating shake-outs
- **Route-specific learning**: Metrics improvement over multiple attempts
