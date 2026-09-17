# python-filter-smoothing

Python library for smoothing and filtering time series data.  Supports offline
batch processing, online sample-by-sample filtering, overlapping-chunk
processing, and thread-safe asynchronous filtering.

## Installation

```bash
# Core (numpy + scipy)
uv pip install -e .

# With development tools (pytest)
uv pip install -e ".[dev]"

# With rerun.io visualization
uv pip install -e ".[viz]"

# Everything
uv pip install -e ".[dev,viz,mpc]"
```

## Quick Start

```python
import numpy as np
from python_filter_smoothing import OfflineFilter, OnlineFilter, ChunkFilter, AsyncFilter

# --- Offline: process an entire time series at once ---
t = np.linspace(0, 1, 200)
x = np.sin(2 * np.pi * t) + np.random.randn(200) * 0.3
filt = OfflineFilter(t, x)
x_smooth = filt.savgol_filter(window_length=21, polyorder=3)

# --- Online: one sample at a time ---
filt = OnlineFilter(method="ema", alpha=0.2)
for ti, xi in zip(t, x):
    filt.update(ti, xi)
print(filt.get_value())

# --- Chunk: overlapping chunks ---
filt = ChunkFilter(method="spline", overlap_strategy="cosine_blend")
filt.add_chunk(x[:60], t[:60])
filt.add_chunk(x[40:120], t[40:120])  # overlaps [40:60]
result = filt.get_filtered()

# --- Async: thread-safe, query at any time ---
filt = AsyncFilter(method="ema", buffer_size=100, alpha=0.3)
filt.update_chunk(t[:50], x[:50])      # add a whole chunk
output = filt.get_output(t=0.25)       # query from another thread
```

## Long-horizon cuRobo MPC application

The maintained MPC example generates a long collision-aware cuRobo path and
resamples it to 5 ms servo commands. Time-scaled cubic Hermite is the standard;
state-to-state Ruckig remains selectable with `resampling.method: ruckig` in
`python_filter_smoothing/configs/long_mpc_application.yml`.
`application.path_generation.mode: direct_then_mpc` first tries a jerk-limited
two-state Ruckig path to the resolver's IK solution, validates every sampled
state, and falls back to collision-aware MPC when the direct path is unsafe.
Use `mpc` to always plan with MPC or `direct_ruckig` to reject unsafe direct paths
without fallback. Direct Ruckig output is validated without the MPC path's
Savitzky–Golay post-filter because Ruckig already enforces derivative limits.
It requires a CUDA-enabled cuRobo installation; the commands below assume its
virtual environment is in the sibling `../curobo` checkout.

```bash
../curobo/.venv/bin/python long_mpc_example.py --duration 65

# Viser playback (open http://localhost:8080)
../curobo/.venv/bin/python visualize_mpc_trajectory.py \
  artifacts/long_mpc \
  --mpc-config python_filter_smoothing/configs/long_mpc.yml
```

Viser binds to `127.0.0.1` by default. Use `--host 0.0.0.0` only on a trusted
network.

### Adapting the example to another mechanism

The example code has no fixed joint count or joint names. Start by copying
`configs/long_mpc.yml` as the mechanism profile and, when the start pose or
targets differ, copy `configs/long_mpc_application.yml` as the application
profile. The mechanism profile references the repository-owned
`configs/long_mpc_optimizer.yml`; copy that file only when the cuRobo optimizer
implementation settings also need tuning. Point `long_mpc_config` at the
mechanism profile, or override that path on the command line:

```bash
../curobo/.venv/bin/python long_mpc_example.py \
  --config path/to/my_robot_application.yml \
  --mpc-config path/to/my_robot_mpc.yml \
  --output artifacts/my_robot_mpc

# summary.json records a portable MPC YAML reference, so no robot argument is needed.
../curobo/.venv/bin/python visualize_mpc_trajectory.py \
  artifacts/my_robot_mpc
```

Most adaptations should only need these YAML groups:

| Group | Tune for a new mechanism |
|---|---|
| `robot`, `scene` | cuRobo robot YAML, collision spheres, world, and optional conservative limit scales |
| `example.initial_joint_positions_rad` | `null`, a partial joint-name mapping, or a full joint-order sequence; choose a bent, nonsingular, collision-free pose |
| `example.target_*` | reachable XYZ offsets and relative rot6D orientations, both based on that initial tool pose |
| `timing` | optimization dt and control points; preserve enough real-time horizon for obstacle detours |
| `optimizer.base_config` | complete cuRobo task/solver YAML; the provided app-owned example exposes L-BFGS history, line search, and CUDA-kernel settings |
| `tool_pose_weight` | task accuracy priority for translation and rotation |
| `scene_collision_weight`, `self_collision_weight` | raise until additional iterations do not trade penetration for pose error |
| `cspace_bound_weight` | soft costs for position/velocity/acceleration/jerk/effort bounds |
| `squared_l2_regularization_weight` | velocity/acceleration/jerk/torque/energy smoothness; weaken cautiously because reversals can increase |
| `target_update` | independent iteration checkpoints, IK reference/seed, fallback seed count, and nearby IK poses |
| `application` | automatic target resolution, queue connection delay, physical acceptance, and progress gate |
| `resampling` | servo-rate interpolation and filter duration; keep generation limits at or below 1.0 initially |

Joint derivative limits come from the cuRobo robot YAML `cspace` section; the
three `robot.*_limit_scale` values only scale them. Weight magnitudes are not
portable by themselves: pose, cspace, and collision costs have different units
and robot-dependent normalization. A practical tuning order is (1) validate the
start pose and IK, (2) choose horizon and target offsets, (3) make collision strong
relative to pose tracking, (4) tune motion regularization, and (5) measure the
planner-time distribution before setting `planning_connection_delay_s`. Keep the
5 ms application-side collision and physical-limit checks enabled throughout.

The default optimizer example uses `history: 3`. Here `history` is the number of
L-BFGS curvature pairs; it is unrelated to saved MPC trajectories. cuRobo's
fused CUDA step-direction kernel requires approximately
`(((2 * control_points * active_dof) + 2) * history + 33) * 4` bytes of shared
memory. If that exceeds the device/kernel limit, runtime can rise sharply.
Increase it only after measuring both solve quality and tail latency. Motion and
collision weights remain in `long_mpc.yml`, where the application overrides the
corresponding defaults in the complete optimizer YAML.

`ContinuousMpcTrajectory.solve_horizon()` only returns a complete cuRobo
`q/dq/ddq` rollout. For each Cartesian target, the adapter first solves IK from
the connection state, installs the result as the joint reference and optimizer
seed, and falls back to several global IK seeds and configured nearby XYZ targets
when needed. Nearby candidates retain the requested orientation and are tried in
`ik_position_offsets_m` order. The straight joint seed is allowed to intersect the
scene: only the optimized output is required to pass the strict collision gate.
The iteration counts
in `optimizer.target_update.candidate_iterations` are independent cold solves.
The application archives every feasible result and ranks feasible candidates by
terminal pose error; a later infeasible solve therefore cannot erase an earlier
safe result.

`CartesianTargetResolver` keeps target fallback out of mechanism-specific calling
code. It first validates the requested pose. If collision-aware IK rejects it, the
resolver searches the segment from the requested pose toward the current,
collision-free tool pose, refines the first feasible boundary, and retreats by
`application.target_resolution.clearance_m`. Requested orientation is retained
when possible and relaxed toward the current orientation only in the configured
`orientation_fractions` order. The selected pose and already validated joint
reference are installed together, so IK is not repeated. `long_mpc_example.py`
uses this resolver by default and records the selected proxy, retreat distance,
orientation fraction, and IK attempt count in its artifacts.

```python
resolution = target_resolver.set_target(
    controller, current_state, requested_position, requested_quaternion
)
```

This is a safe fallback policy, not a global Cartesian planner: it searches one
line segment and may conservatively return a point near the current pose. Disable
it with `application.target_resolution.enabled: false` when upstream guarantees
that every requested pose must be reached exactly.

Selection, progress checks, final validation, and command publication remain in
the application. `application.execution_mode` selects `future_queue` (the verified
old path continues during calculation) or `immediate` (planning time is ignored,
for offline/blocking use). The same choice is available from the command line:

```bash
../curobo/.venv/bin/python long_mpc_example.py \
  --execution-mode future_queue
../curobo/.venv/bin/python long_mpc_example.py \
  --execution-mode immediate
```

With `future_queue`, the example plans from the state already queued at the
configured connection time. A new path is published only if it arrives before
that boundary, improves the target error, respects derivative limits, and passes
cspace, self-collision, and scene-collision checks at every 5 ms output point.
If a check fails, the queue is left untouched and the target is retried from a
newly reserved future state after the old queue advances. A stationary terminal
state may be held after exhaustion, but a moving terminal state is never extended.

The reusable application-side API is intentionally independent of the solver:

```python
from python_filter_smoothing.mpc_application import (
    MpcCommandApplication, PoseError, PoseProgressPolicy
)

app = MpcCommandApplication(initial_path, mode="future_queue", connection_samples=50)
request = app.begin_plan()                 # future q/dq/ddq supplied to MPC
candidate = solve(request.initial_state)   # application-specific MPC call
old_commands = app.consume_during_planning(elapsed_samples)
quality = PoseProgressPolicy().evaluate(initial_error, terminal_error)
if candidate.feasible and quality.accepted and validate(candidate):
    app.publish(request, candidate.state)  # failure leaves the old path intact
command = app.consume(1)                   # called by the servo side
```

The progress gate defaults to at least 2 mm positional improvement, or acceptance
inside 15 mm. Orientation progress is optional. All thresholds are in
`application.progress` in `long_mpc_application.yml`; this gate supplements rather
than replaces cuRobo feasibility and full-path constraint validation. It compares
only the start and endpoint, so detours inside one horizon are allowed; disable or
relax it when a valid plan must end its current horizon farther from the goal.

`application.constraint_acceptance.cspace_mode` uses `physical_limits`. This mode
ignores only cuRobo's aggregate `cspace` result,
then independently checks joint position and the sampled velocity, acceleration,
and discrete jerk. Self- and scene-collision constraints are never ignored. The
maintained profile keeps acceleration effectively strict and allows a 5%
application-side jerk margin:

```bash
../curobo/.venv/bin/python long_mpc_example.py \
  --cspace-acceptance physical_limits \
  --maximum-acceleration-ratio 1.001 \
  --maximum-jerk-ratio 1.05
```

These flags change application acceptance only. Trajectory-generation limits stay
independent in `resampling.*_limit_scale`; their maintained values are 1.0. With
nominal resampling limits, 1000-case testing used none of that extra jerk margin:
the maximum executed ratio remained below 1.0. The 1.05 value is application
headroom, not a generation target. Raising both generation and acceptance to 1.25
added little task progress and increased excessive velocity-reversal phases, so it
is not recommended. This is an application policy, not a change to cuRobo or the
robot model; select `--cspace-acceptance strict` to require the aggregate result.

Both resamplers use the same short centered Savitzky–Golay position filter, restore
the first and last `q/dq/ddq`, and pass through the same final validation. The filter
uses future samples from an already planned trajectory, so it adds no phase shift to
offline queue generation; it would require explicit buffering in a causal streaming
implementation. Discrete 5 ms collision checks do not constitute a continuous swept
collision proof.

Optimizer execution policy is explicit in `long_mpc.yml`. Setup uses
`optimizer.cold_start_iterations`; target updates use
`target_update.candidate_iterations`, `use_ik_joint_reference`, `seed_from_ik`,
`ik_fallback_seeds`, and `ik_position_offsets_m`. Each independent solve resets
the old optimizer cache after the target change. `fixed_iterations` and
`return_best_action` map directly to the cuRobo optimizer configuration. The synchronous example is an asynchronous
planner/servo model: production code should run the same producer in a worker and
let the servo loop consume the verified queue without waiting.

Targets accept XYZ offsets and Zhou 6D rotation offsets (first two rotation-matrix
columns). The adapter projects rot6D to SO(3), composes it with the initial tool
orientation, and passes a wxyz quaternion to cuRobo. Machine- or mechanism-specific
profiles and large benchmark artifacts belong under the ignored `local/` and
`artifacts/` directories.

---

## Filter Types

### OfflineFilter

Processes an entire time series at once.  All methods return the smoothed
data array.

```python
filt = OfflineFilter(t, x)   # x: shape (N,) or (N, D)
```

| Method | Key Parameters | Description |
|--------|---------------|-------------|
| `linear_interpolate(t_query)` | `t_query` | Piecewise-linear interpolation |
| `lowpass_filter(cutoff_freq, sample_rate, order=4)` | `cutoff_freq`, `sample_rate` | Zero-phase Butterworth lowpass |
| `polynomial_fit(degree, t_query=None)` | `degree` | Least-squares polynomial fit |
| `spline_interpolate(t_query, kind="cubic")` | `kind` | Scipy `interp1d` spline |
| `savgol_filter(window_length, polyorder)` | `window_length`, `polyorder` | Savitzky-Golay (preserves peaks) |
| `gaussian_filter(sigma)` | `sigma` | Gaussian kernel smoothing |
| `median_filter(kernel_size)` | `kernel_size` | Robust to outlier spikes |
| `moving_average(window_size)` | `window_size` | Sliding-window box-car average |
| `fir_filter(numtaps, cutoff_freq, sample_rate, ...)` | `numtaps`, `cutoff_freq`, `sample_rate`, `window="hamming"`, `pass_zero=True` | Zero-phase FIR filter |
| `iir_filter(cutoff_freq, sample_rate, ...)` | `iir_type="butterworth"`, `btype="low"`, `order=4`, `rp`, `rs` | Zero-phase IIR (Butterworth, Chebyshev I/II, Elliptic, Bessel) |
| `kalman_smooth(...)` | `process_noise=0.01`, `measurement_noise=0.1`, `state_model="position"` | Kalman smoother (RTS) with optional custom F/H/Q/R |

### OnlineFilter

Causal, sample-by-sample processing.  Created via factory or direct subclass.

```python
filt = OnlineFilter(method="ema", alpha=0.3)
filt.update(t, x)       # feed one sample
y = filt.get_value()     # read current state
filt.reset()             # clear state
```

| Method | Class | Key Parameters |
|--------|-------|---------------|
| `ema` | `OnlineFilterEMA` | `alpha=0.3` — smoothing factor ∈ (0, 1] |
| `moving_average` | `OnlineFilterMovingAverage` | `window=10` — sliding window length |
| `lowpass` | `OnlineFilterLowpass` | `cutoff_freq=0.1`, `sample_rate=1.0`, `order=2` |
| `one_euro` | `OnlineFilterOneEuro` | `min_cutoff=1.0`, `beta=0.0`, `d_cutoff=1.0` |
| `fir` | `OnlineFilterFIR` | `numtaps=31`, `cutoff_freq=5.0`, `sample_rate=100.0`, `window="hamming"` |
| `iir` | `OnlineFilterIIR` | `cutoff_freq=5.0`, `sample_rate=100.0`, `iir_type="butterworth"`, `btype="low"`, `rp`, `rs` |
| `kalman` | `OnlineFilterKalman` | `process_noise=0.01`, `measurement_noise=0.1`, `state_model="position"`, `dt=0.01` |

**One Euro Filter** ([Casiez et al., CHI 2012](https://doi.org/10.1145/2207676.2208639)):
adaptively adjusts the lowpass cutoff based on signal speed — low cutoff for
slow movements (more smoothing), high cutoff for fast movements (less lag).

**IIR Filter** — supports 5 filter families, each with different characteristics:

| `iir_type` | Description | Extra Parameters |
|------------|-------------|------------------|
| `butterworth` | Maximally flat passband (default) | — |
| `chebyshev1` | Sharper roll-off, passband ripple | `rp` (ripple dB) |
| `chebyshev2` | Equiripple stopband | `rs` (attenuation dB) |
| `elliptic` | Sharpest roll-off, ripple in both bands | `rp`, `rs` |
| `bessel` | Best phase linearity (minimal group delay distortion) | — |

All IIR types support `btype` = `"low"`, `"high"`, `"bandpass"`, `"bandstop"`.
For bandpass/bandstop, pass `cutoff_freq` as a list of two frequencies.

**Kalman Filter** — two built-in state models, or fully custom matrices:

| `state_model` | State | Description |
|---------------|-------|-------------|
| `position` | `[pos]` | Random-walk model. Good for general denoising. |
| `position_velocity` | `[pos, vel]` | Constant-velocity model. Better for tracking. Requires `dt`. |

For custom models, pass `F` (transition), `H` (observation), `Q` (process noise),
`R` (measurement noise) matrices directly.

### ChunkFilter

Accepts overlapping chunks of data, merges them, and returns a globally
smoothed result.

```python
filt = ChunkFilter(method="spline", overlap_strategy="cosine_blend")
filt.add_chunk(x_chunk, t_chunk)    # add chunks incrementally
result = filt.get_filtered(t_query) # query smoothed output
```

| Method | Class | Key Parameters |
|--------|-------|---------------|
| `linear` | `ChunkFilterLinear` | — |
| `spline` | `ChunkFilterSpline` | `kind="cubic"` |
| `polynomial` | `ChunkFilterPolynomial` | `degree=3` |
| `savgol` | `ChunkFilterSavgol` | `window_length=11`, `polyorder=3` |
| `gaussian` | `ChunkFilterGaussian` | `sigma=3.0` |
| `lowpass` | `ChunkFilterLowpass` | `cutoff_freq=5.0`, `sample_rate=100.0`, `order=4` |
| `median` | `ChunkFilterMedian` | `kernel_size=5` |
| `fir` | `ChunkFilterFIR` | `numtaps=31`, `cutoff_freq=5.0`, `sample_rate=100.0`, `window="hamming"` |
| `iir` | `ChunkFilterIIR` | `cutoff_freq=5.0`, `sample_rate=100.0`, `iir_type="butterworth"`, `btype="low"` |
| `kalman` | `ChunkFilterKalman` | `process_noise=0.01`, `measurement_noise=0.1`, `state_model="position"` |

**Overlap strategies** — how samples with overlapping timestamps are merged:

| Strategy | Description |
|----------|-------------|
| `latest` | Keep the most recently added sample (default) |
| `mean` | Average all samples at overlapping timestamps |
| `blend` | Linear crossfade in the overlap region |
| `cosine_blend` | Cosine (C¹-smooth) crossfade in the overlap region |

### AsyncFilter

Thread-safe filter for data arriving asynchronously.  Maintains a circular
buffer of recent samples and supports interpolated output at arbitrary query
times.  Output between updates is smoothly interpolated via PCHIP.

```python
filt = AsyncFilter(method="ema", buffer_size=100, alpha=0.3)

# Thread A: feed data
filt.update(t, x)                  # single sample
filt.update_chunk(t_array, x_array) # whole chunk

# Thread B: query output at any time
y = filt.get_output(t=current_time)
```

| Method | Class | Key Parameters | Description |
|--------|-------|---------------|-------------|
| `ema` | `AsyncFilterEMA` | `alpha=0.3` | Exponential moving average |
| `linear` | `AsyncFilterLinear` | — | Linear interpolation over buffer |
| `spline` | `AsyncFilterSpline` | — | Cubic spline over buffer |
| `one_euro` | `AsyncFilterOneEuro` | `min_cutoff`, `beta`, `d_cutoff` | Adaptive lowpass |
| `moving_average` | `AsyncFilterMovingAverage` | `window=10` | Sliding-window average |
| `act` | `AsyncFilterACT` | `k=0.01`, `max_chunks=10` | ACT temporal ensembling |
| `rail` | `AsyncFilterRAIL` | `poly_degree=3`, `blend_duration=None`, `dual_quintic=True`, `auto_align=False` | RAIL trajectory post-processing |

#### Action Chunk Methods

The `act` and `rail` methods are designed for **action chunk** filtering in
VLA (Vision-Language-Action) robot control pipelines, where a model
asynchronously predicts multi-step action trajectories and a control loop
must output smooth commands at a fixed rate.

**ACT Temporal Ensembling** ([Zhao et al., RSS 2023](https://tonyzhaozh.github.io/aloha/);
[LeRobot](https://huggingface.co/docs/lerobot/async)):

Maintains a window of recent action chunks.  At query time, computes a weighted
average across all chunks that cover the query timestamp.  Newer chunks receive
exponentially higher weight: `w = exp(-k × age)`.

```python
filt = AsyncFilter(method="act", k=0.01, max_chunks=10)
filt.update_chunk(t_chunk, x_chunk)
y = filt.get_output(t=now)
```

**RAIL** ([Cheng et al., arXiv:2512.24673](https://arxiv.org/abs/2512.24673)):

Two-stage trajectory post-processing ensuring C² continuity:
1. **Intra-chunk smoothing**: polynomial fit per dimension to filter prediction noise.
2. **Inter-chunk fusion**: quintic polynomial blend
   (`scipy.interpolate.BPoly.from_derivatives`) matching position, velocity,
   and acceleration at chunk boundaries.

Additional parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dual_quintic` | `True` | Use dual-quintic spline (Eq. 11-13 of the paper), splitting the blend into two halves to prevent overshoot from Runge's phenomenon.  Set `False` for single quintic. |
| `auto_align` | `False` | Automatically correct new-chunk timestamps via temporal alignment (Eq. 10) that maximises motion-direction consistency.  Requires `set_current_time()`. |
| `align_window` | `None` | Search window (seconds) for auto-alignment.  `None` = 50 % of chunk duration. |

Use `set_current_time(t)` in your control loop so that the blend region starts
at the actual switch time (as in Algorithm 1 of the paper):

```python
filt = AsyncFilter(method="rail", poly_degree=3, dual_quintic=True, auto_align=True)

# Control loop (runs at high frequency)
while running:
    t_now = get_clock()
    filt.set_current_time(t_now)      # update before chunk arrival
    # ... (inference thread calls filt.update_chunk(t_chunk, x_chunk))
    y = filt.get_output(t=t_now)      # smooth output
    send_to_robot(y)
```

---

## Visualization

Time series input/output can be visualized using [rerun.io](https://rerun.io/).

### Visualize during manual testing

```bash
uv pip install -e ".[viz]"

# Spawn the rerun viewer and stream data during tests
pytest --visualize

# Or save to an .rrd file for later viewing
pytest --visualize --rrd-path=recording.rrd
```

### Visualization API

```python
from python_filter_smoothing.visualize import (
    init_recording,
    log_time_series,
    log_scalar,
    send_dim_blueprint,
    configure_series_style,
)

init_recording(spawn=True)

# Log batch data
log_time_series("input/noisy", t, x, dim_names=["X", "Y", "Z"], dim_first=True)

# Log single sample (online filters)
log_scalar("output/ema", t=0.5, x=np.array([1.0, 2.0, 3.0]),
           dim_names=["X", "Y", "Z"], dim_first=True)

# Create per-dimension panel layout
send_dim_blueprint(["X", "Y", "Z"])

# Style a series
configure_series_style("output/ema", color=(255, 0, 0), name="EMA", width=2.0)
```

The `dim_first=True` option creates separate panels for each dimension (X, Y, Z),
with all filter outputs overlaid in each panel — making it easy to compare
methods per dimension.

---

## Examples

The `examples/` folder contains runnable scripts with rerun visualization:

```bash
python examples/offline_example.py    # All offline methods on noisy XYZ data
python examples/online_example.py     # Real-time sample-by-sample filtering
python examples/chunk_example.py      # Streaming chunks with overlap handling
python examples/async_example.py      # Multi-threaded async filtering (all methods)
python examples/async_example2.py     # Simulated inference with ACT & RAIL
python examples/async_example3.py     # Single-filter reactive chunk generation
```

Each script accepts `--save <path>.rrd` to write a recording file instead of
spawning the viewer.

| Example | What it demonstrates |
|---------|---------------------|
| `offline_example.py` | All offline methods compared side-by-side |
| `online_example.py` | EMA, moving average, lowpass, one-euro on streaming data |
| `chunk_example.py` | Chunk-based processing with overlap strategies |
| `async_example.py` | Producer/consumer threads with all async methods |
| `async_example2.py` | Simulated async inference with variable latency, ACT and RAIL |
| `async_example3.py` | Single-filter simulation with reactive (state-dependent) chunk generation |

---

## Architecture

The library uses a **base class + subclass** pattern.  Each filter type has an
abstract base class defining the interface and common logic, with concrete
subclasses implementing specific algorithms.  Factory functions provide
backward-compatible creation:

```
OnlineFilterBase (ABC)          → OnlineFilter("ema")     → OnlineFilterEMA
                                → OnlineFilter("lowpass")  → OnlineFilterLowpass
                                → OnlineFilter("kalman")   → OnlineFilterKalman
                                  ...

ChunkFilterBase (ABC)           → ChunkFilter("spline")   → ChunkFilterSpline
  ├─ overlap merging              ChunkFilter("kalman")    → ChunkFilterKalman
  └─ get_filtered()               ...

AsyncFilterBase (ABC)           → AsyncFilter("ema")      → AsyncFilterEMA
  ├─ thread-safe locking          AsyncFilter("act")       → AsyncFilterACT
  ├─ circular buffer              AsyncFilter("rail")      → AsyncFilterRAIL
  └─ PCHIP output interpolation   ...
```

---

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with visualization (spawns rerun viewer)
uv run pytest tests/ --visualize

# Run specific module
uv run pytest tests/test_async_filter.py -v
```

210 tests covering all filter types, edge cases, and thread safety.

---

## ⚠️ Disclaimer

**This project's code is substantially generated by AI (GitHub Copilot / Claude)
and has NOT undergone sufficient verification or formal review.** Use at your own
risk.  The authors make no guarantees regarding correctness, numerical stability,
or suitability for safety-critical applications (including but not limited to
real robot control).  Users are strongly encouraged to independently verify
filter behaviour for their specific use case before deployment.

---

## References & Acknowledgements

Several filter algorithms in this library are independent reimplementations
based on concepts described in the following published works:

- **One Euro Filter**: G. Casiez, N. Roussel, D. Vogel.
  "1€ Filter: A Simple Speed-Based Low-Pass Filter for Noisy Input in
  Interactive Systems." *CHI 2012*.
  [DOI:10.1145/2207676.2208639](https://doi.org/10.1145/2207676.2208639) —
  Original reference implementations are BSD/MIT licensed.

- **ACT Temporal Ensembling**: T. Z. Zhao, V. Kumar, S. Levine, C. Finn.
  "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware."
  *RSS 2023*.
  [Project page](https://tonyzhaozh.github.io/aloha/) —
  Original code is MIT licensed.

- **LeRobot Async Inference**: Hugging Face.
  [Documentation](https://huggingface.co/docs/lerobot/async) —
  Apache-2.0 licensed.

- **RAIL**: B. Cheng et al.
  "VLA-RAIL: A Real-Time Asynchronous Inference Linker for VLA Models and
  Robots." [arXiv:2512.24673](https://arxiv.org/abs/2512.24673) —
  Our implementation is based on the mathematical methods (polynomial
  smoothing, quintic C² blending) described in the paper, not derived from
  the authors' code.

This library does **not** contain copied source code from any of the above
projects.  All implementations were written independently from the published
algorithmic descriptions.

---

## License

This project is licensed under the [MIT License](LICENSE).

### Dependency Licenses

| Package | License |
|---------|---------|
| [NumPy](https://numpy.org/) | BSD-3-Clause |
| [SciPy](https://scipy.org/) | BSD-3-Clause |
| [rerun-sdk](https://rerun.io/) (optional) | MIT / Apache-2.0 |
| [pytest](https://pytest.org/) (dev) | MIT |
