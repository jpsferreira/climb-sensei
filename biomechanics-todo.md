# Biomechanics & Tracking — Todo List

## Critical Bugs

### BUG1: Body angle calculation loses lean direction
- [ ] Replace `arctan(abs(dx) / abs(dy))` with `atan2(dx, dy)` in TechniqueCalculator
- [ ] Return signed angle so left lean vs right lean are distinguishable
- [ ] Update tests to verify signed output
- **File**: `src/climb_sensei/technique.py` (or equivalent `_calculate_body_angle`)
- **Severity**: HIGH — current output is ambiguous

### BUG2: Fatigue score unbounded
- [ ] Clamp fatigue score to [0.0, 1.0] with `np.clip`
- [ ] Handle edge case where `early_avg = 0` (currently returns 0, should flag)
- **File**: `src/climb_sensei/fatigue.py` (or equivalent `_calculate_fatigue_score`)
- **Severity**: HIGH — inconsistent with 0-1 scoring expected by UI

### BUG3: Jerk window too short (4 frames = 0.13s)
- [ ] Replace hardcoded 4-frame window with configurable `window_size` parameter
- [ ] Default to 15-20 frames (~0.5s @ 30fps) for meaningful signal
- [ ] Smooth jerk values with moving average before returning
- **File**: `src/climb_sensei/stability.py` (`_calculate_jerk`)
- **Severity**: HIGH — 4-frame jerk is noise, not signal

### BUG4: No NaN/Inf validation after pose extraction
- [ ] Add `validate_landmarks()` function checking all x/y/z in [0, 1] and not NaN
- [ ] Call after `extract_landmarks()` — skip invalid frames
- [ ] Prevent NaN propagation through metrics pipeline
- **File**: `app/services/upload.py` + new validator
- **Severity**: HIGH — silent corruption of all downstream metrics

### BUG5: Pre-extracted landmarks fake confidence data
- [ ] Remove hardcoded `confidence = 0.8` and `visibility = 100%`
- [ ] Either pass real confidence data through or mark quality as "unvalidated"
- [ ] Add warning in tracking quality report when using pre-extracted data
- **File**: `src/climb_sensei/tracking_quality.py` (`_analyze_landmarks_sequence`)
- **Severity**: MEDIUM — misleading quality reports

### BUG6: Smoothness formula uses unexplained magic number
- [ ] Document or empirically validate the `exp(-jitter * 20)` decay constant
- [ ] Consider replacing with `1 - (std/mean)` coefficient of variation approach
- [ ] Add unit tests with known-good/bad jitter ranges
- **File**: `src/climb_sensei/tracking_quality.py` (`_calculate_smoothness`)
- **Severity**: MEDIUM — untested formula, results unreliable

## Efficiency Improvements

### EFF1: Vectorize tracking smoothness calculation (10-20x speedup)
- [ ] Replace nested Python loops over (frames × landmarks) with numpy array ops
- [ ] `np.diff(positions_array, axis=0)` + `np.linalg.norm(diffs, axis=2)`
- **File**: `src/climb_sensei/tracking_quality.py` (`_calculate_smoothness`)

### EFF2: Batch joint angle calculation (2-3x speedup)
- [ ] Vectorize all 8 joint angles in single numpy operation
- [ ] Replace 8 sequential `calculate_joint_angle()` calls
- **File**: `src/climb_sensei/biomechanics.py` + `climbing_analysis_service.py`

### EFF3: Eliminate redundant data copies (2x memory)
- [ ] Remove `list(self._com_positions)[-4:]` copies from deque
- [ ] Use numpy array slicing instead of list comprehensions for history
- [ ] Filter types at insertion rather than aggregation
- **Files**: `stability.py`, `technique.py`, all calculators

### EFF4: Running statistics instead of full-history aggregation (1.5x)
- [ ] Implement Welford's online algorithm for mean/variance
- [ ] Maintain running min/max instead of recomputing from history
- [ ] Remove `get_summary()` recalculation pattern
- **Files**: All calculator `get_summary()` methods

## Minor Improvements

### MINOR1: Lock-off detection oversimplified
- [ ] Consider velocity context (arm should be stationary during lock-off)
- [ ] Add configurable angle threshold (default 90, clinical range 60-70)
- [ ] Document that detection is positional only, not force-based

### MINOR2: Rest detection ignores velocity
- [ ] Add velocity < threshold check alongside arm extension
- [ ] Consider body angle (near-vertical = hanging rest vs horizontal = mantle)
- [ ] Add configurable thresholds

### MINOR3: Movement economy unbounded
- [ ] Clamp to [0, 1] or document that values > 1 indicate tracking loss
- [ ] Add tracking-loss detection (large jumps in COM position)

### MINOR4: Z-coordinate unused
- [ ] Either utilize MediaPipe z-depth for 3D metrics or explicitly drop it
- [ ] Document that all distance metrics are 2D projections

### MINOR5: Sway not normalized
- [ ] Normalize sway by shoulder width to make values body-size-independent
- [ ] Makes cross-climber comparison meaningful
